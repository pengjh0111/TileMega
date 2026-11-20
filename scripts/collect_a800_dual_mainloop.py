#!/usr/bin/env python3
"""Bind guarded two-CTA SM80 MMA measurements to a derived calibration."""
import argparse
import json
import math
from pathlib import Path

from sm90d1_common import read, sha


def collect(run, target_path, binary, production_path):
    target = read(target_path)
    resources = target['resources']
    device_path = run / 'job/physical-device-identity.json'
    guard_path = run / 'guard/guard_result.json'
    policy_path = run / 'policy.json'
    probe = read(run / 'artifact.json')
    production = read(production_path)
    device, guard, policy = map(read, (device_path, guard_path, policy_path))
    grid = 2 * resources['num_sms']
    shared = production['shared_memory_bytes']
    if (target['arch_tag'] != 'sm_80' or 'A800' not in target['device_provenance']['name']
            or device['compute_capability'] != [8, 0]
            or guard['code'] != 0 or guard['acceptance'] != 'strict'
            or guard['ownership_verified'] is not True
            or policy['gpu_uuid'] != device['nvml_uuid']
            or device['guard_policy_sha256'] != sha(policy_path)
            or device['sm_count'] != resources['num_sms']
            or device['shared_memory_per_sm'] != resources['max_smem_per_sm']
            or device['shared_memory_per_block_optin'] != resources['max_dynamic_smem_per_cta']
            or device['registers_per_sm'] != resources['regs_per_sm']
            or device['l2_bytes'] != resources['l2_bytes']
            or probe['architecture'] != 'sm_80' or probe['binary_sha256'] != sha(binary)
            or probe['production_identity_sha256'] != sha(production_path)
            or probe['production_binary_sha256'] != production['binary_sha256']
            or probe['observed_residency_required'] != 2
            or production['arch'] != 'sm_80' or production['plan']['residency'] != 2
            or production['plan']['grid'] != grid or probe['production_grid'] != grid
            or probe['dynamic_shared'] != shared
            or not resources['max_smem_per_sm'] // 3 < shared <= resources['max_smem_per_sm'] // 2):
        raise ValueError('dual calibration differs from the guarded device or linked production model')

    constants = target['sm80_native_constants']
    table = constants['mma_mainloop']
    expected = set()
    for gemm in production['plan']['native_dense']['gemms'].values():
        tn = gemm['tile_n']
        tk = gemm['slot_bytes'] // (2 * tn)
        expected.add((tn, tk, gemm['warps_n'], gemm['register_stages']))
    if set(probe['cases']) != {'x'.join(map(str, shape)) for shape in expected}:
        raise ValueError('the dual calibration cases must match the linked GEMM coordinates')

    measurements = {}
    for shape in sorted(expected):
        case = 'x'.join(map(str, shape))
        row_path = run / f'{case}.json'
        row = read(row_path)
        key = ':'.join(map(str, (grid, *shape)))
        if (row['status'] != 'measured' or row['grid'] != grid
                or tuple(row[k] for k in ('tile_n', 'tile_k', 'warps_n', 'register_stages')) != shape
                or row['warps_n'] * row['warps_k'] != 4
                or row['input_norm'] is not True or row['samples'] <= 0
                or row['overlap_sms'] != device['sm_count'] or row['dynamic_shared'] != shared
                or not 0 < row['median_ns'] <= row['p99_ns'] < math.inf
                or key in table['median_ns'] or key in table['p99_ns']):
            raise ValueError(f'invalid or duplicate dual-CTA measurement {case}')
        table['median_ns'][key] = row['median_ns']
        table['p99_ns'][key] = row['p99_ns']
        measurements[key] = dict(row, receipt_sha256=sha(row_path))

    table.setdefault('supplements', []).append(dict(
        schema=1, parent_profile_sha256=sha(target_path),
        method='isolated MmaSlice K16 service with physically overlapping CTAs',
        source_sha=probe['source_sha'], binary_sha256=probe['binary_sha256'],
        gpu_uuid=device['nvml_uuid'], guard_result_sha256=sha(guard_path),
        device_receipt_sha256=sha(device_path), production_identity_sha256=sha(production_path),
        production_binary_sha256=production['binary_sha256'], processes_per_shape=1,
        evidence=run.name, measurements=measurements))
    target['sm80_native_decisions']['reuse_payload'] = True
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'target', 'binary', 'production', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    profile = collect(args.run, args.target, args.binary, args.production)
    with args.out.open('x') as stream:
        json.dump(profile, stream, indent=2)
        stream.write('\n')
