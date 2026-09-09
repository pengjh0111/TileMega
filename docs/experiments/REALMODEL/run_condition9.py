#!/usr/bin/env python3
"""Rerun the two original 4x4096 cost leaders with FP32 partials."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phases', default='export,build,run')
    p.add_argument('--runs', type=int, default=50)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    work = here/'raw/work/condition9'
    out = here/'condition9'
    work.mkdir(parents=True, exist_ok=True); out.mkdir(exist_ok=True)
    build = Path(os.environ.get('BUILD_DIR', repo/'build-portable'))
    nvcc = Path(os.environ.get('CUDACXX', '/usr/local/cuda/bin/nvcc'))
    commands = json.loads((out/'commands.json').read_text()) if (out/'commands.json').exists() else []
    def run(command, log):
        commands.append(list(map(str, command)))
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (out/log).open('w') as stream:
            subprocess.run(commands[-1], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
    phases = args.phases.split(',')
    (out/'status.txt').write_text('RUNNING\n')
    try:
        if 'export' in phases:
            run([sys.executable, here/'export_real.py', '--repo', repo, '--out', work/'export',
                 '--layers', 4, '--hidden', 4096, '--intermediate', 14336,
                 '--heads', 32, '--kv-heads', 8], 'export.txt')
        manifest = json.loads((work/'export/fixture/manifest.json').read_text())
        for field, value in dict(layers=4, hidden=4096, intermediate=14336, heads=32,
                                 kv_heads=8, seq=4, past=3).items():
            if manifest[field] != value: raise RuntimeError(f'wrong original scene: {field}')
        if manifest['dtype'] != 'torch.bfloat16': raise RuntimeError('condition 9 requires BF16')
        (out/'fixture_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if 'build' in phases:
            run([sys.executable, repo/'python/tilemega/export_bridge.py',
                 work/'export/exported_program.pt2', '--out', work/'model.json'], 'bridge.txt')
            for split in (8, 16):
                run([build/'tools/tilemega-compile', work/'model.json', work/f'k{split}.cu',
                     '--variants', here/f'plan_split{split}.json'], f'codegen_k{split}.txt')
                run([nvcc, '-std=c++17', '-O2', '-arch=native', '-lineinfo',
                     '-Xptxas=-v,--warn-on-spills', '--expt-relaxed-constexpr',
                     '-DTILEMEGA_FP32_PARTIALS=1',
                     *[f'-I{repo/d}' for d in ('include', 'third_party/cutlass/include',
                       'third_party/cutlass/tools/util/include', 'third_party/cutlass/test')],
                     work/f'k{split}.cu', build/'libtilemega.a',
                     f'-L{nvcc.parent.parent/"lib64"}', '-lcudart', '-o', work/f'k{split}'], f'ptxas_k{split}.txt')
                print('BUILT', split, flush=True)
        if 'run' not in phases:
            (out/'status.txt').write_text('BUILT_ONLY\n')
            return
        with (out/'correctness.tsv').open('w') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['split', 'round', 'execution_index', 'pass', 'mismatch', 'max_abs',
                             'max_rel', 'hash', 'l05_ms', 'l1_ms', 'l2_ms'])
            for r in range(args.runs):
                for execution_index, split in enumerate((8,16) if r % 2 == 0 else (16,8)):
                    result = subprocess.run([str(work/f'k{split}'), str(work/'export/fixture')],
                        capture_output=True, text=True, timeout=120,
                        env=dict(os.environ, TILEMEGA_WARMUP='5', TILEMEGA_REPEAT='11'))
                    text = result.stdout+result.stderr
                    log = out/f'k{split}_r{r}.txt'; log.write_text(text)
                    diff = re.search(r'l05_vs_l0_mismatch=(\d+) max_abs=(\S+) max_rel=(\S+)', text)
                    hashes = re.search(r'E2E_HASH l05=(\S+) l1=(\S+) l2=(\S+)', text)
                    if not diff or not hashes: raise RuntimeError(f'incomplete run: {log}')
                    timing = dict(x.split('=',1) for x in next(
                        line for line in text.splitlines() if line.startswith('E2E_TIME ')).split()[1:])
                    passed = result.returncode == 0 and 'RESULT status=PASS' in text
                    writer.writerow([split,r,execution_index,int(passed),*diff.groups(),hashes[1],
                                     timing['l05_ms'],timing['l1_ms'],timing['l2_ms']])
                    stream.flush()
                    if not passed or len(set(hashes.groups())) != 1:
                        raise RuntimeError(f'condition 9 still fails; stop, do not relax tolerance: {log}')
                print('ROUND', r+1, flush=True)
        (out/'status.txt').write_text(f'PASS {2*args.runs}/{2*args.runs} fresh processes\n')
    except BaseException:
        (out/'status.txt').write_text('STOPPED: inspect logs\n')
        raise


if __name__ == '__main__':
    main()
