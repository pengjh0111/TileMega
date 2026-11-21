"""Bind dense service measurements to the calibrated SM80 target and body."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from collect_a800_registry import append_service


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


PRODUCTION_KEY_FIELDS = ('grid', 'n', 'k', 'batch', 'tn', 'tk', 'slots', 'warps_n', 'warps_k',
                         'register_stages', 'u', 'residual_tiles', 'input_norm', 'live_rows',
                         'copy_issue_group')


def production_key(row):
    return ':'.join((row['epilogue'], *(str(int(row[key])) for key in PRODUCTION_KEY_FIELDS)))


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
               'untraced_span_ns', 'traced_span_ns')
    measurements = {}
    rejected = {}
    typed = artifact.get('contract') == 'sm80-dense-gemm-service-v1'
    for case in artifact['cases']:
        if Path(case).name != case:
            raise ValueError('calibration case must be a local case label')
        row = read(run/f'{case}.json')
        if row['status'] == 'resource_rejected':
            rejected[case] = 'linked resource capacity'
            continue
        if (row['status'] != 'measured' or not row['output_bitwise_equal']
                or row['weight_layout'] != 'contiguous'
                or row['grid'] != device['sm_count'] * row['requested_residency']
                or min(row['plain_residency'], row['traced_residency']) < row['requested_residency']
                or row['warps_n'] * row['warps_k'] != 4
                or not math.isfinite(row['consumer_epilogue_ns_per_task'])
                or row['consumer_epilogue_ns_per_task'] < 0
                or not typed and row['consumer_epilogue_ns_per_task'] == 0
                or any(not math.isfinite(row[m]) or row[m] <= 0 for m in metrics)):
            raise ValueError(f'invalid production GEMM service result for {case}')
        if typed:
            # This probe isolates a stage. Occupancy capacity does not by itself
            # establish co-residency for the enclosing persistent program.
            if row['requested_residency'] != 1:
                rejected[case] = 'requires linked full-program residency and overlap'
                continue
            ratio = row['traced_span_ns'] / row['untraced_span_ns']
            if not 0.75 <= ratio <= 1.25:
                rejected[case] = f'traced/untraced span {ratio:.3f}'
                continue
            if row['epilogue'] not in ('swiglu', 'store', 'residual'):
                raise ValueError(f'unknown production epilogue for {case}')
            if (row['input_norm'] != (row['epilogue'] != 'residual')
                    or row['u'] not in (4, 8, 16) or row['copy_issue_group'] not in (0, 4, 8)
                    or row['residual_tiles'] != (
                        max(2, (row['n']//row['tn'] + row['grid']-1)//row['grid'])
                        if row['epilogue'] == 'residual' else 0)):
                raise ValueError(f'inconsistent production geometry for {case}')
            key = production_key(row)
            if key != artifact['geometry_by_case'][case]:
                raise ValueError(f'measured implementation differs from queued shape for {case}')
        else:
            if row['input_norm'] is not True:
                raise ValueError('legacy gate pricing requires input normalization')
            u = row.get('u') or int(row['epilogue'].removeprefix('swiglu-u'))
            if u not in (4, 8, 16):
                raise ValueError('unknown dense gate epilogue')
            key = ':'.join(str(row[k]) for k in geometry) + f':{u}'
        if key in measurements:
            raise ValueError(f'duplicate service geometry {key}')
        measurements[key] = dict(row, case=case)
    constants = base['sm80_native_constants']
    profile = dict(schema=1, base_target_sha256=sha(target), source_sha=artifact['source_sha'],
        body_headers=artifact['body_headers'], binary_sha256=artifact['binary_sha256'],
        gpu_uuid=device['nvml_uuid'], guard_result_sha256=sha(run/'guard/guard_result.json'),
        evidence=str(run), measurements=measurements, rejected=rejected)
    if typed:
        frozen = read(run/'frozen.json')
        registry = constants.setdefault('dense_gemm_service', dict(
            schema=1, reference_target_sha256=sha(target), profiles=[]))
        if (registry['schema'] != 1 or
                registry['reference_target_sha256'] != frozen['target_sha256'] or
                sha(binary.with_suffix('.identity.json')) != frozen['identity_sha256'] or
                artifact != read(binary.with_suffix('.identity.json'))):
            raise ValueError('service evidence differs from the frozen probe or reference target')
        if any(profile['guard_result_sha256'] == previous['guard_result_sha256']
               for previous in registry['profiles']):
            return base
        compatible = {key: previous['binary_sha256']
                      for previous in registry['profiles']
                      if previous['body_headers'] == profile['body_headers']
                      for key in previous['measurements']}
        profile['additional_observations'] = {}
        for key in tuple(measurements):
            if key in compatible:
                profile['additional_observations'][key] = dict(
                    row=measurements.pop(key), selected_probe_sha256=compatible[key])
        registry['profiles'].append(profile)
    elif measurements:
        append_service(constants, [profile])
    else:
        raise ValueError('calibration has no accepted service geometry')
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
