"""Resume C0 combinations in bounded, already guarded batches."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allocations', default='-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--stores', default='0,1')
    parser.add_argument('--loads', default='0,1,2,3,4,5,6')
    parser.add_argument('--widths', default='8')
    parser.add_argument('--packets', default='1024,262144')
    parser.add_argument('--all-to-all', type=int, choices=(0, 1), required=True)
    parser.add_argument('--budget-s', type=float, default=500)
    parser.add_argument('--exclude', type=Path)
    args = parser.parse_args()
    excluded = set()
    if args.exclude:
        proof = json.loads(args.exclude.read_text())
        excluded = {tuple(item) for item in proof['excluded_alloc_store_load']}
        if not proof.get('previous_failures'):
            raise ValueError('C0 exclusion must retain prior failed-cohort evidence')
    if os.getenv('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('matrix requires the group guard')
    values = lambda text: list(map(int, text.split(',')))
    start = time.monotonic()
    pending = False
    for width, allocation, store, load, packets in itertools.product(
            values(args.widths), values(args.allocations), values(args.stores),
            values(args.loads), values(args.packets)):
        if (allocation, store, load) in excluded:
            continue
        out = args.out / f'w{width}_a{allocation}_s{store}_l{load}_p{packets}'
        summary = out / 'summary.json'
        if summary.exists():
            if json.loads(summary.read_text())['fresh_processes'] != 5:
                raise RuntimeError('incomplete combination summary')
            continue
        if time.monotonic() - start > args.budget_s:
            pending = True
            break
        command = [sys.executable, 'experiments/mx7/run_probe.py', '--exe', 'runs/mx7/ll_litmus',
                   '--source', 'experiments/mx7/ll_litmus.cu', '--out', str(out), '--',
                   str(width), str(packets), '2000', str(allocation), str(store), str(load),
                   str(args.all_to_all)]
        out.mkdir(parents=True, exist_ok=True)
        print('combination', str(out), flush=True)
        with (out / 'runner.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print('combination_exit', result.returncode, flush=True)
        if not summary.exists():
            raise RuntimeError('probe infrastructure failed; retain partial records: ' + str(out))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'pending.json').write_text(json.dumps(dict(pending=pending)) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
