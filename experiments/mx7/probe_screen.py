"""Prioritize unmeasured small-packet cohorts without replacing any evidence."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=500)
    args = parser.parse_args()
    if os.getenv('TILEMEGA_GPU_LOCK_HELD') != '1' or os.getenv('TILEMEGA_DEVICE_GROUP') != '6,7':
        raise RuntimeError('screening requires the same guarded pair as the original matrix')
    started = time.monotonic()
    pending = False
    for allocation, store, load in itertools.product(range(-1, 8), range(2), range(7)):
        folder = args.out / f'w8_a{allocation}_s{store}_l{load}_p1024'
        summary = folder / 'summary.json'
        if summary.exists():
            if json.loads(summary.read_text())['fresh_processes'] != 5:
                raise ValueError('prior cohort lacks five fresh processes')
            continue
        if time.monotonic() - started > args.budget_s:
            pending = True
            break
        folder.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, 'experiments/mx7/run_probe.py', '--exe', 'runs/mx7/ll_litmus',
                '--source', 'experiments/mx7/ll_litmus.cu', '--out', str(folder), '--',
                '8', '1024', '2000', str(allocation), str(store), str(load), '0']
        with (folder / 'runner.log').open('x') as output:
            child = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT)
        print(folder.name, 'exit', child.returncode, flush=True)
        if not summary.exists():
            raise RuntimeError('screening tool failure; retain partial cohort')
    (args.out / 'screen_pending.json').write_text(json.dumps(dict(pending=pending)) + '\n')


if __name__ == '__main__':
    main()
