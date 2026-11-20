"""Bind dense service measurements to the calibrated SM80 target and body."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run, target, binary):
    base = read(target)
    resources = base['resources']
    guard = read(run/'guard/guard_result.json')
    device = read(run/'job/physical-device-identity.json')
    policy = read(run/'policy.json')
    artifact = read(run/'artifact.json')
    if (base['arch_tag'] != 'sm_80' or 'A800' not in base['device_provenance']['name']
            or guard['acceptance'] != 'strict' or guard['code'] != 0
            or policy['gpu_uuid'] != device['nvml_uuid']
            or device['guard_policy_sha256'] != sha(run/'policy.json')
            or device['sm_count'] != resources['num_sms']
            or device['shared_memory_per_sm'] != resources['max_smem_per_sm']
            or device['shared_memory_per_block_optin'] != resources['max_dynamic_smem_per_cta']
            or device['registers_per_sm'] != resources['regs_per_sm']
            or device['l2_bytes'] != resources['l2_bytes']
            or artifact['binary_sha256'] != sha(binary)
            or artifact['architecture'] != 'sm_80' or artifact['roles'] != [32, 128]):
        raise ValueError('calibration receipt differs from the guarded SM80 body or target')
    geometry = ('grid', 'n', 'k', 'batch', 'tn', 'tk', 'slots', 'warps_n', 'register_stages')
    metrics = ('producer_issue_ns_per_tile', 'consumer_mma_ns_per_tile',
               'consumer_epilogue_ns_per_task', 'untraced_span_ns', 'traced_span_ns')
    measurements = {}
    for case in artifact['cases']:
        if Path(case).name != case:
            raise ValueError('calibration case must be a local case label')
        row = read(run/f'{case}.json')
        if row['status'] == 'resource_rejected':
            continue
        if (row['status'] != 'measured' or not row['output_bitwise_equal']
                or row['input_norm'] is not True or row['weight_layout'] != 'contiguous'
                or row['grid'] != device['sm_count'] * row['requested_residency']
                or min(row['plain_residency'], row['traced_residency']) < row['requested_residency']
                or row['warps_n'] * row['warps_k'] != 4
                or any(not math.isfinite(row[m]) or row[m] <= 0 for m in metrics)):
            raise ValueError(f'invalid production GEMM service result for {case}')
        u = int(row['epilogue'].removeprefix('swiglu-u'))
        if u not in (4, 8, 16):
            raise ValueError('unknown dense gate epilogue')
        key = ':'.join(str(row[k]) for k in geometry) + f':{u}'
        if key in measurements:
            raise ValueError(f'duplicate service geometry {key}')
        measurements[key] = dict(row, case=case)
    if not measurements:
        raise ValueError('calibration has no accepted service geometry')
    constants = base['sm80_native_constants']
    if 'dense_gate_service' in constants:
        raise ValueError('provide the original target without a dense gate service profile')
    constants['dense_gate_service'] = dict(
        schema=1, base_target_sha256=sha(target), source_sha=artifact['source_sha'],
        body_headers=artifact['body_headers'], binary_sha256=artifact['binary_sha256'],
        gpu_uuid=device['nvml_uuid'], guard_result_sha256=sha(run/'guard/guard_result.json'),
        evidence=run.name, measurements=measurements)
    return base


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'target', 'binary', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    profile = collect(args.run, args.target, args.binary)
    with args.out.open('x') as stream:
        json.dump(profile, stream, indent=2)
        stream.write('\n')
