#!/usr/bin/env python3
"""Validate an immutable correctness receipt before timing on another GPU."""
import argparse
from pathlib import Path

from sm90d1_common import read, sha, write


def verify_receipt(state, job, binary, config):
    row = read(state)[job]
    if row['status'] != 'done':
        raise ValueError('external correctness job did not complete successfully')
    folder = Path(row['out'])
    guard = read(folder / 'guard_result.json')
    selected = read(folder / 'job/selected-config.json')
    command = read(folder / 'job/physical-device-command.json')
    c1 = read(folder / 'job/measurements/c1.json')
    c2 = read(folder / 'job/measurements/c2.json')
    configuration = read(config)
    if guard['code'] or command['actual_returncode']:
        raise ValueError('external correctness guard or child rejected the run')
    if (selected['stage'] != 'correctness' or selected['decode_sha256'] != sha(binary)
            or selected['config_sha256'] != sha(config)):
        raise ValueError('correctness receipt belongs to another artifact or workload')
    args = selected['command']
    if args[args.index('--max-tokens') + 1] != '1024':
        raise ValueError('full-length step/loop correctness is required')
    if not c1['pass'] or c1['positions'] < 256 * configuration['batch']:
        raise ValueError('teacher-forced development check is incomplete')
    if (not c2['pass'] or not c2['same_binary'] or len(c2['checks']) != 2
            or not all(v['pass'] for v in c2['checks'])
            or {v['decode_loop_used'] for v in c2['checks']} != {False, True}):
        raise ValueError('both execution modes must pass with the same binary')
    paths = ['guard_result.json', 'job/selected-config.json',
             'job/physical-device-command.json', 'job/physical-device-identity.json',
             'job/measurements/c1.json', 'job/measurements/c2.json']
    return dict(job=job, source_state=str(state.resolve()), source_attempt=str(folder),
        binary_sha256=sha(binary), config_sha256=sha(config),
        receipt_sha256={name: sha(folder / name) for name in paths},
        ownership_verified=guard['ownership_verified'], pass_=True,
        evidence='verified correctness dependency; no timing was performed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = verify_receipt(a.state, a.job, a.binary, a.config)
    result['pass'] = result.pop('pass_')
    write(a.out, result)
    print('accepted external correctness receipt:', a.job)


if __name__ == '__main__':
    main()
