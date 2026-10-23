#!/usr/bin/env python3
"""Check an instrumented native C-ABI chain and its exported phase records."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from analyze_sm90r2_native_phases import analyze
from sm90d1_common import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fresh-processes', type=int, default=1)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('a guarded GPU lease is required')
    a.out.mkdir(parents=True, exist_ok=False)
    if a.fresh_processes<1:
        raise ValueError('positive fresh process count is required')
    if a.fresh_processes>1:
        for fresh in range(a.fresh_processes):
            subprocess.run([sys.executable,__file__,'--binary',str(a.binary.resolve()),
                '--out',str(a.out/f'fresh{fresh}')],check=True)
        write(a.out/'receipt.json',dict(pass_=True,fresh_gpu_processes=a.fresh_processes,
            purpose='functional diagnostic chain; not a performance result'))
        return
    with (a.out / 'program.log').open('w') as stream:
        result = subprocess.run([str(a.binary.resolve())], stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, TILEMEGA_NATIVE_TEST_TRACE_DIR=str(a.out.resolve())))
    if result.returncode:
        raise RuntimeError('instrumented C-ABI chain failed numerical/step-loop checks')
    summary = analyze(a.out)
    classes = {r['kind']: r for r in summary['classes']}
    if set(classes) != {'qkv', 'attention', 'o'}:
        raise ValueError('all three native stages must emit diagnostic records')
    if any(classes[k]['first_mma_from_release_ns'] is None for k in ('qkv','o')):
        raise ValueError('missing native GEMM first-MMA timestamp')
    if not classes['attention']['new_row_tasks'] or not classes['attention']['merge_tasks']:
        raise ValueError('missing current-row or last-arriver records')
    write(a.out / 'receipt.json', dict(pass_=True, diagnostic=summary,
        purpose='functional validation of diagnostic plumbing; not a performance result'))
    print('native diagnostic export and numerical chain pass')


if __name__ == '__main__':
    main()
