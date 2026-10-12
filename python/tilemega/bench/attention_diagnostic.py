"""Collect clock64 task diagnostics from the validated attention test harness.

This command must be a guarded timing step, after its numerical dependency.
It never writes a candidate-selection score.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .arms import sha256

IMPLEMENTATIONS = ('mma16', 'pvswap', 'pvswap2', 'simt')


def collect(build_dir, out, rounds=3):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise RuntimeError('attention diagnostics require the GPU guard')
    if rounds < 1:
        raise ValueError('rounds must be positive')
    out.mkdir(parents=True, exist_ok=True)
    binaries = {name: build_dir / ('attention_layout_test' if name == 'mma16'
                else 'attention_layout_' + name + '_test') for name in IMPLEMENTATIONS}
    identities = {name: sha256(path) for name, path in binaries.items()}
    receipts = []
    for rnd in range(rounds):
        names = IMPLEMENTATIONS[rnd % 4:] + IMPLEMENTATIONS[:rnd % 4]
        for name in names:
            folder = out / f'round{rnd}' / name
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / 'tasks.jsonl'
            env = dict(os.environ, TILEMEGA_ATTENTION_TEST_LATENCY=str(target))
            with (folder / 'numeric.log').open('w') as stream:
                subprocess.run([str(binaries[name])], env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            if sha256(binaries[name]) != identities[name]:
                raise ValueError('attention diagnostic binary changed')
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            if not rows or any(not row['pass'] or row['impl'] != IMPLEMENTATIONS.index(name) for row in rows):
                raise ValueError('invalid numerical path cannot provide latency diagnostics')
            receipts.append(dict(round=rnd, impl=name, tasks=str(target),
                                 sha256=sha256(target), binary_sha256=identities[name]))
    (out / 'diagnostics.json').write_text(json.dumps(dict(
        scope='single-CTA clock64 diagnostics, not end-to-end acceptance',
        units='cycles over all recorded steps; reported clock_khz is device metadata',
        runs=receipts), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    a = p.parse_args()
    collect(a.build_dir.resolve(), a.out.resolve(), a.rounds)

if __name__ == '__main__':
    main()
