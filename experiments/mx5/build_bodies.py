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


def main():
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
    p.add_argument('--pw', type=int)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    name = 'bodies' if a.body == 'gemm' else a.body
    if a.bench and a.body != 'gemm': p.error('the current repeated-stage bench supports GEMM only')
    source_path = Path('experiments/mx5/stage_bench.cu') if a.bench else Path(f'test/unit/mx/decode_{name}_v2.cu')
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
        command = ['/opt/maca/tools/cu-bridge/bin/cucc', '-std=c++17', '-O3', '-shared',
                   '-Xcompiler', '-fPIC,-resource-usage', '-fno-strict-aliasing',
                   '-DTILEMEGA_SYNC_V3=1', '-DTILEMEGA_WATCHDOG=0',
                   f'-DMX5_TEST_THREADS={a.threads}', *[f'-D{k}={v}' for k,v in definitions.items()],
                   *([] if a.pw is None else [f'-DMX5_BENCH_PW={a.pw}']), '-Iinclude',
                   str(source_path), '-o', str(output)]
        start = time.monotonic()
        with output.with_suffix('.log').open('wb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        item = dict(body=a.body, bench=a.bench, pw=a.pw, **coordinates, threads=a.threads, base_head=head, source_sha256=source,
                    seconds=time.monotonic() - start, returncode=result.returncode, command=command)
        if output.exists(): item['binary_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        record.write_text(json.dumps(item, indent=2) + '\n')
        print(f'{stem} rc={result.returncode} seconds={item["seconds"]:.1f}', flush=True)
        return result.returncode

    if a.body == 'gemm':
        keys = [(f'tn{tn}_d{d}', dict(MX5_TEST_TN=tn, MX5_TEST_DEPTH=d), dict(tn=tn, depth=d))
                for tn in a.tn for d in a.depth if (tn // 16) * 4 + d * (tn // 16 + 1) * 4 <= 160]
    elif a.body == 'head':
        keys = [(f'tn{tn}', dict(MX5_TEST_TN=tn), dict(tn=tn)) for tn in a.tn]
    elif a.body == 'attention':
        keys = [(f'd{dim}_q{q}', dict(MX5_TEST_DIM=dim, MX5_TEST_Q=q), dict(dim=dim, query=q))
                for dim in a.dim for q in a.query]
    else:
        keys = [(a.body, {}, {})]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(build, keys))
    if identity(source_path) != source: raise RuntimeError('source changed while compiling the body libraries')
    if any(results): raise SystemExit(1)


if __name__ == '__main__': main()
