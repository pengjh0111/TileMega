#!/usr/bin/env python3
"""Fixed-weight-prefix BF16 depth/noise-floor diagnostic; tolerance unchanged."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def fields(line):
    return dict(word.split('=', 1) for word in line.split()[1:] if '=' in word)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 4, 6, 8, 12, 16])
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--phases', default='build,run')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    here = Path(__file__).resolve().parent
    work, out = here/'depth_work', (args.out or here/'depth_results').resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = Path(os.environ.get('BUILD_DIR', repo/'build-portable'))
    nvcc = Path(os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc'))
    commands = json.loads((out/'commands.json').read_text()) if (out/'commands.json').exists() else []
    def command(cmd, log):
        start = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(list(map(str, cmd)), stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=7200, env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'))
        commands.append({'command': list(map(str, cmd)), 'returncode': result.returncode,
                         'seconds': time.monotonic()-start, 'log': os.path.relpath(log,repo)})
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        result.check_returncode()
    (out/'status.txt').write_text('RUNNING\n')
    try:
        if 'build' in args.phases.split(','):
            for depth in args.depths:
                directory = work/f'l{depth}'
                directory.mkdir(parents=True, exist_ok=True)
                command([sys.executable, here/'export_real.py', '--repo', repo,
                         '--out', directory/'export', '--layers', depth, '--prefix-depth', 16,
                         '--fp32-reference'], out/f'export_l{depth}.txt')
                command([sys.executable, repo/'python/tilemega/export_bridge.py',
                         directory/'export/exported_program.pt2', '--out', directory/'model.json'],
                        out/f'bridge_l{depth}.txt')
                command([build/'tools/tilemega-compile', directory/'model.json', directory/'model.cu',
                         '--variants', repo/'docs/experiments/OWNERSHIP/plan_structured.json'],
                        out/f'codegen_l{depth}.txt')
                command([nvcc, '-std=c++17', '-O2', '-arch=native', '-lineinfo',
                         '-Xptxas=-v,--warn-on-spills', '--expt-relaxed-constexpr',
                         '-DTILEMEGA_FP32_PARTIALS=1',
                         *[f'-I{repo/d}' for d in ('include', 'third_party/cutlass/include',
                           'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
                         directory/'model.cu', build/'libtilemega.a',
                         f'-L{nvcc.parent.parent/"lib64"}', '-lcudart', '-o', directory/'model'],
                        out/f'ptxas_l{depth}.txt')
                print(f'BUILT depth={depth}', flush=True)
        # Independently verify every shared input/parameter, not just a seed.
        seen, hashes = {}, []
        for depth in args.depths:
            fixture = work/f'l{depth}/export/fixture'
            manifest = json.loads((fixture/'manifest.json').read_text())
            if manifest['prefix_depth'] != 16 or not manifest['fp32_reference']:
                raise RuntimeError('depth fixtures lack fixed-prefix / FP32 reference provenance')
            for path in sorted(fixture.glob('*.bin')):
                if not path.name.startswith(('input_', 'state_')): continue
                sha = digest(path)
                if path.name in seen and seen[path.name] != sha:
                    raise RuntimeError(f'nonidentical shared input/weight across depth: {path.name}')
                seen[path.name] = sha
                hashes.append({'depth': depth, 'file': path.name, 'sha256': sha})
        (out/'fixture_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
        if 'run' not in args.phases.split(','):
            (out/'status.txt').write_text('BUILT_ONLY\n')
            return
        command(['nvidia-smi', '-q'], out/'gpu_before.txt')
        rows, first = [], {}
        for round_id in range(args.runs):
            order = args.depths[round_id % len(args.depths):]+args.depths[:round_id % len(args.depths)]
            for execution_index, depth in enumerate(order):
                directory = work/f'l{depth}'
                dump = directory/'dump'
                dump.mkdir(exist_ok=True)
                env = dict(os.environ, TILEMEGA_WARMUP='5', TILEMEGA_REPEAT='11')
                if round_id == 0: env['TILEMEGA_DUMP_BUFFERS'] = str(dump)
                result = subprocess.run([str(directory/'model'), str(directory/'export/fixture')],
                                        env=env, capture_output=True, text=True, timeout=120)
                text = result.stdout+result.stderr
                (out/f'l{depth}_r{round_id}.txt').write_text(text)
                diff = re.search(r'l05_vs_l0_mismatch=(\d+) max_abs=(\S+) max_rel=(\S+)', text)
                if not diff or not re.search(r'RESULT status=(PASS|MISMATCH)', text):
                    raise RuntimeError(f'incomplete depth run: {depth}, {round_id}')
                if 'l1_vs_l05_mismatch=0 max_abs=0' not in text or 'l2_vs_l1_mismatch=0 max_abs=0' not in text:
                    raise RuntimeError(f'inter-level correctness gate failed: {depth}, {round_id}')
                if 'l2_iter1_vs_iter0_mismatch=0 max_abs=0' not in text:
                    raise RuntimeError(f'iteration disagreement: {depth}, {round_id}')
                hashes = fields(next(line for line in text.splitlines() if line.startswith('E2E_HASH ')))
                if len(set(hashes.values())) != 1:
                    raise RuntimeError(f'inter-level hash mismatch: {depth}, {round_id}')
                values = (diff[1], diff[2], diff[3], hashes['l05'])
                if depth in first and first[depth] != values:
                    raise RuntimeError(f'fresh-process nondeterminism: {depth}, {round_id}')
                first[depth] = values
                timing = fields(next(line for line in text.splitlines() if line.startswith('E2E_TIME ')))
                rows.append({'depth': depth, 'round': round_id, 'execution_index': execution_index,
                             'returncode': result.returncode, 'mismatch': diff[1], 'max_abs': diff[2],
                             'max_rel': diff[3], 'hash': hashes['l05'],
                             'l05_ms': timing['l05_ms'], 'l1_ms': timing['l1_ms'], 'l2_ms': timing['l2_ms']})
                with (out/'depth.tsv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0], delimiter='\t', lineterminator='\n')
                    writer.writeheader(); writer.writerows(rows)
            print(f'ROUND {round_id+1}/{args.runs}', flush=True)
        # Read only final hidden, as distinct from the harness's all-output maximum.
        import torch
        noise = []
        for depth in args.depths:
            directory = work/f'l{depth}'
            def load(path, dtype):
                return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).double()
            fixture = directory/'export/fixture'
            py = load(fixture/'reference_0.bin', torch.bfloat16)
            wide = load(fixture/'reference_fp32_0.bin', torch.float32)
            log = (out/f'l{depth}_r0.txt').read_text()
            buffer = re.search(r'E2E_OUTPUT_DIFF index=0 buffer=(\d+)', log)[1]
            tm = load(directory/f'dump/buffer_{buffer}.bin', torch.bfloat16)
            errors = {}
            for name, a, b in (('pytorch_bf16_vs_fp32', py, wide),
                               ('tilemega_bf16_vs_fp32', tm, wide), ('tilemega_vs_pytorch_bf16', tm, py)):
                error = a-b
                norm = torch.linalg.vector_norm(error).item()
                errors[name] = norm
                noise.append({'depth': depth, 'comparison': name, 'l2_error': norm,
                              'relative_l2': norm/torch.linalg.vector_norm(b).item(),
                              'max_abs': error.abs().max().item()})
            print(f'NOISE depth={depth} k_l2={errors["tilemega_bf16_vs_fp32"]/errors["pytorch_bf16_vs_fp32"]}', flush=True)
        with (out/'noise.tsv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=noise[0], delimiter='\t', lineterminator='\n')
            writer.writeheader(); writer.writerows(noise)
        (out/'status.txt').write_text(f'DIAGNOSTIC_COMPLETE: {len(rows)} fresh processes; inspect numerical depth gate\n')
    except BaseException:
        (out/'status.txt').write_text('STOPPED: inspect last command or process log\n')
        raise


if __name__ == '__main__':
    main()
