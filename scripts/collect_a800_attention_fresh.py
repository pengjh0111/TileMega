#!/usr/bin/env python3
"""Append guarded new-row work for the SM80 attention stage."""
import argparse
import json
import math
from pathlib import Path

from collect_a800_dense_service import read, sha


def guarded_probe(target, run, binary):
    base = read(target)
    artifact = read(run/'artifact.json')
    identity = binary.with_suffix('.identity.json')
    guard = read(run/'guard/guard_result.json')
    device = read(run/'job/physical-device-identity.json')
    policy = run/'policy.json'
    frozen = read(run/'frozen.json')
    if (base['arch_tag'] != 'sm_80' or 'A800' not in base['device_provenance']['name']
            or device['sm_count'] != base['resources']['num_sms'] or device['compute_capability'] != [8, 0]
            or device['nvml_uuid'] != read(policy)['gpu_uuid'] or device['guard_policy_sha256'] != sha(policy)
            or guard['code'] or guard['acceptance'] != 'strict' or not guard['ownership_verified']
            or frozen['target_sha256'] != sha(target) or frozen['identity_sha256'] != sha(identity)
            or sha(binary) != artifact['binary_sha256'] or artifact != read(identity)
            or artifact['contract'] != 'sm80-attention-fresh-service-v1'):
        raise ValueError('attention new-row service lacks matching guarded evidence')
    return base, artifact, device


def collect(target, run, binary):
    base, artifact, device = guarded_probe(target, run, binary)
    constants = base['sm80_native_constants']
    measurements = {}
    for case in artifact['cases']:
        if Path(case).name != case:
            raise ValueError('attention case label is not local')
        sample = run/f'{case}.json'
        row = read(sample)
        key = f'd{row["dim"]}:q{row["queries"]}:norm{row["norm"]}'
        if (case != key.replace(':', '-') or key not in constants['attention_ns'] or key in measurements
                or row['status'] != 'measured' or row['samples'] != 256
                or any(not math.isfinite(row[name]) or row[name] <= 0 for name in ('full_ns', 'fresh_ns'))):
            raise ValueError(f'invalid attention observation for {case}')
        measurements[key] = dict(row, measurement_sha256=sha(sample))
    if measurements.keys() != constants['attention_ns'].keys():
        raise ValueError('the attention supplement did not cover the calibrated shapes')
    if 'attention_fresh_service' in constants:
        raise ValueError('retain the existing matching attention supplement')
    constants['attention_fresh_service'] = dict(schema=1, measurements=measurements, evidence=str(run),
        source_sha=artifact['source_sha'], body_headers=artifact['body_headers'],
        binary_sha256=artifact['binary_sha256'], reference_target_sha256=sha(target),
        gpu_uuid=device['nvml_uuid'], guard_sha256=sha(run/'guard/guard_result.json'))
    return base


def collect_compact(target, run, binary):
    base, artifact, device = guarded_probe(target, run, binary)
    profiles, seen = [], set()
    sms = device['sm_count']
    for case in artifact['cases']:
        if Path(case).name != case or not case.startswith('compact-workspace-barrier'):
            raise ValueError('compact service requires a frozen rendezvous shape')
        sample = run/f'{case}.json'
        row = read(sample)
        shape = row['dim'], row['queries'], row['norm']
        if (shape in seen or shape[0] != 128 or not 1 <= shape[1] <= 8 or shape[2] not in (0, 1, 2)
                or row['grid'] != 2*sms):
            raise ValueError(f'invalid compact attention scope {case}')
        seen.add(shape)
        if row['status'] == 'resource_rejected':
            if row['residency'] >= 2 or row['max_shared_per_sm'] != device['shared_memory_per_sm']:
                raise ValueError(f'inconsistent compact attention capacity {case}')
            continue
        if (row['status'] != 'measured' or row['residency'] != 2 or row['overlapping_sms'] != sms
                or row['consumer_threads'] != 128 or row['iterations'] != 1024 or row['samples'] != 18*sms
                or not math.isfinite(row['barrier_ns']) or row['barrier_ns'] <= 0):
            raise ValueError(f'compact attention lacked observed concurrent residency for {case}')
        profiles.append(dict(schema=1, measurement=row, body_headers=artifact['body_headers'],
            source_sha=artifact['source_sha'], binary_sha256=artifact['binary_sha256'],
            gpu_uuid=device['nvml_uuid'], guard_sha256=sha(run/'guard/guard_result.json'),
            measurement_sha256=sha(sample), evidence=str(run)))
    constants = base['sm80_native_constants']
    if 'compact_attention_service' in constants or not profiles:
        raise ValueError('require a fresh compact attention profile set')
    constants['compact_attention_service'] = dict(schema=2, profiles=profiles)
    return base


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('target', 'run', 'binary', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    collect_service = collect_compact if args.compact else collect
    result = collect_service(args.target, args.run, args.binary)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
