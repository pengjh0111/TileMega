#!/usr/bin/env python3
"""Build immutable MX-5 body-test libraries and record their source identity."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tarfile


def inputs(source=Path('test/unit/mx/decode_bodies_v2.cu')):
    import re
    files = set()
    pending = [source]
    while pending:
        path = pending.pop()
        if path in files: continue
        files.add(path)
        for name in re.findall(r'^\s*#include\s*[<"]([^>"]+)[>"]', path.read_text(), re.M):
            dep = Path('include') / name if name.startswith('tilemega/') else path.parent / name
            if dep.is_file(): pending.append(dep)
    return sorted(files)


def identity(source=Path('test/unit/mx/decode_bodies_v2.cu')):
    digest = hashlib.sha256()
    for path in inputs(source):
        digest.update(str(path).encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tn', type=int, nargs='+', default=[16, 32, 64, 128])
    p.add_argument('--depth', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--threads', type=int, choices=[512, 576], default=576)
    p.add_argument('--body', choices=['gemm', 'head', 'attention', 'transform', 'chain'], default='gemm')
    p.add_argument('--dim', type=int, nargs='+', default=[64, 128])
    p.add_argument('--query', type=int, nargs='+', default=[2, 3, 4, 8])
    p.add_argument('--bench', action='store_true')
    p.add_argument('--null-idle', action='store_true', help='test unused-wave prefetch with null weight pointers')
    p.add_argument('--pw', type=int)
    p.add_argument('--prefetch-kind', type=int, choices=[0, 1, 2])
    a = p.parse_args(argv)
    if a.null_idle and (a.bench or a.body not in ('gemm', 'head')):
        p.error('--null-idle supports only GEMM/head unit libraries')
    if a.bench and a.body not in ('gemm', 'head', 'attention'):
        p.error('repeated-stage benches support GEMM, head and attention')
    if a.prefetch_kind is not None and not (a.bench and a.body == 'attention'):
        p.error('--prefetch-kind requires --bench --body attention')
    if a.bench and a.body == 'head':
        if any(tn not in (16, 32, 64, 128) for tn in a.tn) or any(d not in (1, 2, 3, 4) for d in a.depth):
            p.error('head bench requires TN in {16,32,64,128} and depth in {1,2,3,4}')
        if a.pw is not None and a.pw < 0:
            p.error('head bench --pw must be nonnegative; zero measures the body without prefetch')
    if a.bench and a.body == 'attention':
        if any(dim not in (64, 128) for dim in a.dim) or any(q not in (2, 3, 4, 8) for q in a.query):
            p.error('attention bench requires dim in {64,128} and query in {2,3,4,8}')
        if a.pw is not None:
            p.error('attention bench uses --prefetch-kind instead of --pw')
        if a.prefetch_kind is None:
            a.prefetch_kind = 2
    if a.bench and a.body == 'head':
        keys = build_inputs(a)[1]
        if not keys:
            p.error('no legal head bench candidates for the requested TN/depth')
        if a.pw not in (None, 0) and any(coordinates['depth'] > a.pw for _, _, coordinates in keys):
            p.error('head bench --pw must be zero or at least every selected depth')
    return a


def build_inputs(a):
    name = 'bodies' if a.body == 'gemm' else a.body
    if a.bench:
        filename = 'stage_bench.cu' if a.body == 'gemm' else f'{a.body}_stage_bench.cu'
        source_path = Path('experiments/mx5') / filename
    else:
        suffix = '_null_idle' if a.null_idle else ''
        source_path = Path(f'test/unit/mx/decode_{name}{suffix}_v2.cu')
    if a.body == 'gemm' or (a.body == 'head' and a.bench):
        keys = [(f'tn{tn}_d{d}', dict(MX5_TEST_TN=tn, MX5_TEST_DEPTH=d), dict(tn=tn, depth=d))
                for tn in a.tn for d in a.depth if (tn // 16) * 4 + d * (tn // 16 + 1) * 4 <= 160]
    elif a.body == 'head':
        keys = [(f'tn{tn}', dict(MX5_TEST_TN=tn), dict(tn=tn)) for tn in a.tn]
    elif a.body == 'attention':
        extra_def = dict(MX5_BENCH_PREFETCH_KIND=a.prefetch_kind) if a.bench else {}
        extra_coord = dict(prefetch_kind=a.prefetch_kind) if a.bench else {}
        keys = [(f'd{dim}_q{q}', dict(MX5_TEST_DIM=dim, MX5_TEST_Q=q, **extra_def),
                 dict(dim=dim, query=q, **extra_coord)) for dim in a.dim for q in a.query]
    else:
        keys = [(a.body, {}, {})]
    return source_path, keys


def compiler_command(a, definitions, source_path, output):
    return ['/opt/maca/tools/cu-bridge/bin/cucc', '-std=c++17', '-O3', '-shared',
            '-Xcompiler', '-fPIC,-resource-usage', '-fno-strict-aliasing',
            '-DTILEMEGA_SYNC_V3=1', '-DTILEMEGA_WATCHDOG=0',
            f'-DMX5_TEST_THREADS={a.threads}', *[f'-D{k}={v}' for k,v in definitions.items()],
            *([] if a.pw is None else [f'-DMX5_BENCH_PW={a.pw}']), '-Iinclude',
            str(source_path), '-o', str(output)]


def main():
    a = parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    source_path, keys = build_inputs(a)
    source = identity(source_path)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    archive = a.out / 'sources.tar.gz'
    if archive.exists(): raise RuntimeError('choose a new output directory: ' + str(a.out))
    with tarfile.open(archive, 'w:gz') as tar:
        for path in inputs(source_path): tar.add(path)

    def build(key):
        stem, definitions, coordinates = key
        output = a.out / f'{stem}.so'
        record = output.with_suffix('.json')
        if output.exists() or record.exists():
            raise RuntimeError('choose a new output directory: ' + str(output))
        command = compiler_command(a, definitions, source_path, output)
        start = time.monotonic()
        with output.with_suffix('.log').open('wb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        item = dict(body=a.body, bench=a.bench, null_idle=a.null_idle, pw=a.pw, **coordinates, threads=a.threads, base_head=head, source_sha256=source,
                    seconds=time.monotonic() - start, returncode=result.returncode, command=command)
        if output.exists(): item['binary_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        record.write_text(json.dumps(item, indent=2) + '\n')
        print(f'{stem} rc={result.returncode} seconds={item["seconds"]:.1f}', flush=True)
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(build, keys))
    if identity(source_path) != source: raise RuntimeError('source changed while compiling the body libraries')
    if any(results): raise SystemExit(1)


if __name__ == '__main__': main()
