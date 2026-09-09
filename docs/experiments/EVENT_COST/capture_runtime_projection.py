#!/usr/bin/env python3
"""Capture independent A2 runtime counters with archived binaries, once per cell.

This is not a >=50-process synchronization test or a performance experiment.
The existing binary's numerical check remains enabled and any failure stops us.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', type=int, default=[1,2,4,8,16])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise RuntimeError('refusing to overwrite an existing capture')
    manifest = dict(purpose='one-process-per-cell runtime counters; not synchronization acceptance',
                    environment={k:v for k,v in os.environ.items() if k.startswith('TILEMEGA_')},
                    binaries={}, fixtures={}, runs=[])
    if manifest['environment']:
        raise RuntimeError('run without inherited TILEMEGA overrides')
    for model in ('gqa2','mha4'):
        for split in args.splits:
            binary = repo/f'docs/experiments/BF16/raw_splitk/bin/{model}_k{split}_fp321'
            manifest['binaries'][str(binary.relative_to(repo))] = digest(binary)
        for seq in (1,4,128,512,2048):
            for past in (0,3,512):
                fixture = repo/f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}'
                manifest['fixtures'][str(fixture.relative_to(repo))] = {
                    str(p.relative_to(fixture)):digest(p) for p in sorted(fixture.rglob('*')) if p.is_file()}
    (args.out/'status.txt').write_text('RUNNING counter capture; no performance claim\n')
    try:
        for seq in (1,4,128,512,2048):
            for past in (0,3,512):
                for model in ('gqa2','mha4'):
                    for split in args.splits:
                        binary = repo/f'docs/experiments/BF16/raw_splitk/bin/{model}_k{split}_fp321'
                        if digest(binary) != manifest['binaries'][str(binary.relative_to(repo))]:
                            raise RuntimeError('binary changed during capture')
                        fixture = repo/f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p{past}'
                        command = [str(binary),str(fixture)]
                        started = time.monotonic()
                        run = subprocess.run(command,text=True,stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,timeout=120)
                        log = args.out/f'{model}_s{seq}_p{past}_k{split}.txt'
                        log.write_text(run.stdout)
                        manifest['runs'].append(dict(command=command,log=log.name,
                            returncode=run.returncode,seconds=time.monotonic()-started))
                        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                        if run.returncode or 'RESULT status=PASS' not in run.stdout:
                            raise RuntimeError(f'runtime gate failed: {log}')
                        for prefix in ('E2E_RESOURCE ','E2E_SCHEDULE '):
                            if sum(line.startswith(prefix) for line in run.stdout.splitlines()) != 1:
                                raise RuntimeError(f'missing/ambiguous counter record: {log}')
                print(f'captured seq={seq} past={past}',flush=True)
        (args.out/'status.txt').write_text(f'CAPTURE PASS {len(manifest["runs"])} cells; '
            'one process each, not >=50-process synchronization acceptance\n')
    except BaseException:
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (args.out/'status.txt').write_text('STOPPED: inspect manifest and last log\n')
        raise


if __name__ == '__main__':
    main()
