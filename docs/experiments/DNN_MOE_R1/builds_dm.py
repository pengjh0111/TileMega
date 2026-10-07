#!/usr/bin/env python3
"""Build DM-1 pinned plans under the shared lock, with source identities."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from identity_dm import generate, source_snapshot
from pin_case import classes, pin

HERE = Path(__file__).resolve().parent


def one(job):
    root = Path(job['root'])
    compiler = root / 'build-dm/tools/tilemega'
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=True)
    so = out / 'plan.so'
    snapshot = source_snapshot(root, compiler)
    (out / 'source.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    recipe = pin(job['manifest'], job['classes'], job['classes'], out, job['overrides'])
    command = [str(compiler), 'compile', job['export'], str(so),
               '--serving', job['phase'], '--batch', str(job['batch']),
               '--past-range', '0:0' if job['phase'] == 'prefill' else '64:1086',
               '--capacity', '1088', '--solver', 'skeleton', '--solve', job['target'],
               '--runtime-target', job['target'], '--emit', 'serving',
               '--search-passes', '1', '--top-m', '1', '--search-jobs', '2',
               '--search-budget-ms', '60000', '--dump-cg', str(out / 'selected.mlir'),
               '--candidate-mode', 'L1', '--candidate-loop', '0',
               '--measure-cmd', shlex.join([sys.executable, str(HERE / 'measure_stub.py')])]
    command += recipe['options']
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    lock = os.environ.get('TILEMEGA_GPU_LOCK', '/root/r14_work/gpu.lock')
    with (out / 'build.log').open('w') as stream:
        code = subprocess.run(['flock', lock, *command], cwd=root,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError('compile failed; see ' + str(out / 'build.log'))
    if sorted(classes(str(so) + '.classes.tsv').values()) != recipe['expected_partition']:
        raise ValueError('rebuilt class partition changed')
    identity = generate(so, snapshot)
    record = dict(job, exit_code=0, artifact_id=identity['artifact_id'],
                  so=str(so), placeholder_measurement=True)
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(event='build_complete', model=job['model'], phase=job['phase'],
                          batch=job['batch'], artifact_id=identity['artifact_id'])), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for job in json.loads(args.jobs.read_text()):
        rows.append(one(job))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
