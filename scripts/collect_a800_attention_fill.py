#!/usr/bin/env python3
"""Add the measured 13-chunk SM80 attention merge to a new target."""
import argparse
import math
from pathlib import Path
from statistics import median

from collect_sm90r3_constants import process_rows
from sm90d1_common import ROOT, read, sha, write


def merge_samples(processes):
    samples = {8: [], 13: []}
    for rows in processes:
        latest = {}
        for row in rows:
            if row['name'] != 't_merge':
                continue
            fields = dict(item.split('=') for item in row['key'].split(':'))
            chunks = int(fields.pop('s'))
            duration, tail = float(row['median']), float(row['p99'])
            if (fields != {'d': '128', 'q': '4', 'norm': '0'} or chunks not in samples
                    or chunks in latest or row['unit'] != 'ns' or int(row['samples']) != 248
                    or not 0 < duration <= tail < math.inf):
                raise ValueError('invalid attention merge probe row')
            latest[chunks] = duration, tail
        if latest.keys() != samples.keys():
            raise ValueError('each process must measure the eight- and thirteen-chunk merges')
        for chunks, value in latest.items():
            samples[chunks].append(value)
    if len(samples[8]) != 5:
        raise ValueError('five fresh attention processes are required')
    return {str(chunks): {'median_ns': median(v[0] for v in values),
                          'p99_ns': median(v[1] for v in values)} for chunks, values in samples.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ('run', 'allocation-target', 'target', 'artifact', 'out'):
        p.add_argument('--' + opt, required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error('the calibrated target needs a new output path')
    target, allocation = read(a.target), read(a.allocation_target)
    original, local = (t['device_provenance'] for t in (target, allocation))
    constants = target['sm80_native_constants']
    if (not all(t['arch_tag'] == 'sm_80' for t in (target, allocation)) or 'A800' not in local['name']
            or original['uuid'] != constants['gpu_uuid'] or target['resources'] != allocation['resources']
            or any(local[k] != original[k] for k in ('name', 'memory_mib', 'capability', 'mig', 'driver'))):
        p.error('the allocation must match the calibrated A800 hardware class and driver')
    jobs = read(a.run/'queue/queue_calibration.json')
    section = 'sm80-attention-fill'
    artifact, probe = sha(a.artifact), sha(a.allocation_target)
    if (len(jobs) != 5 or {j['name'] for j in jobs} != {f'{section}-f{i}' for i in range(5)} or
            any((j['artifact_sha256'], j['target_sha256'], j['expected_uuid']) !=
                (artifact, probe, local['uuid']) for j in jobs)):
        p.error('attention jobs must identify the exact probe, binary and local device')
    results = a.run/'results'
    measured = merge_samples(process_rows(read(results/'state.json'), section, 5, results, local['uuid']))
    costs = constants['attention_ns']['d128:q4:norm0']['merge']
    if '13' in costs or costs['8'] <= 0:
        p.error('the parent must contain the measured eight-chunk reference and have a vacant 13-chunk key')
    constants.setdefault('attention_supplements', []).append(dict(
        shape='d128:q4:norm0', measurements=measured, original_eight_ns=costs['8'],
        parent_sha256=sha(a.target), binary_sha256=artifact, allocation_sha256=probe,
        gpu_uuid=local['uuid'], processes=5, evidence=str(a.run.resolve().relative_to(ROOT))))
    costs['13'] = measured['13']['median_ns']
    write(a.out, target)


if __name__ == '__main__':
    main()
