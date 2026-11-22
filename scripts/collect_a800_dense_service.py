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


def collect(run, target, binary, calibration_target=None, maximum_perturbation=0.25):
    base = read(target)
    resources = base['resources']
    if calibration_target is not None:
        probe = read(calibration_target)
        if {k:v for k,v in base.items() if k != 'sm80_native_constants'} != {
                k:v for k,v in probe.items() if k != 'sm80_native_constants'}:
            raise ValueError('the calibration target changes the target outside its native prices')
    guard = read(run/'guard/guard_result.json')
    device = read(run/'job/physical-device-identity.json')
    policy = read(run/'policy.json')
    artifact = read(run/'artifact.json')
    if (base['arch_tag'] != 'sm_80' or 'A800' not in base['device_provenance']['name']
            or guard['acceptance'] != 'strict' or guard['code'] != 0 or not guard['ownership_verified']
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
    epilogue_only = '-DTILEMEGA_SM80_DENSE_EPILOGUE_CLOCK=1' in artifact.get('command', [])
    if epilogue_only and not typed:
        raise ValueError('isolated epilogue sampling requires a production service key')
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
                or any(not math.isfinite(row[m]) or row[m] <= 0 for m in metrics[2:])):
            raise ValueError(f'invalid production GEMM service result for {case}')
        if epilogue_only:
            epilogue = row['consumer_epilogue_ns_per_task']
            compute = row.get('consumer_residual_compute_ns_per_task', 0)
            sm_ghz = row.get('epilogue_calibration_sm_ghz', 0)
            if (row.get('epilogue_only') is not True or row['epilogue'] != 'residual'
                    or not math.isfinite(sm_ghz) or sm_ghz <= 0 or not 0 < compute <= epilogue
                    or any(row[m] != 0 for m in metrics[:2])):
                raise ValueError(f'invalid isolated epilogue service result for {case}')
        elif any(not math.isfinite(row[m]) or row[m] <= 0 for m in metrics[:2]):
            raise ValueError(f'invalid mainloop service result for {case}')
        if typed:
            if row['n'] % row['tn'] and (row['epilogue'] != 'swiglu'
                    or row.get('padded_n') != (row['n']+row['tn']-1)//row['tn']*row['tn']
                    or not row.get('clipped_output_bitwise_equal')):
                raise ValueError(f'{case} lacks a matching clipped-output reference')
            if row['dynamic_shared'] < artifact.get('reserved_program_shared', 0):
                raise ValueError(f'{case} did not reserve the frozen program arena')
            # Count overlap between CTAs with work in every plain launch.
            # The enclosing program still needs a separate occupancy query.
            if row['requested_residency'] != 1 and (
                    not 0 < row.get('overlapping_sms', 0) <= device['sm_count']
                    or row['dynamic_shared'] != artifact.get('reserved_program_shared')):
                rejected[case] = 'no active-CTA overlap in the frozen arena'
                continue
            ratio = row['traced_span_ns'] / row['untraced_span_ns']
            threshold = min(maximum_perturbation, 0.05) if epilogue_only else maximum_perturbation
            if abs(ratio-1) > threshold:
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
        known_targets = {registry['reference_target_sha256'], sha(target)}
        if calibration_target is not None:
            known_targets.add(sha(calibration_target))
        if (registry['schema'] != 1 or frozen['target_sha256'] not in known_targets or
                sha(binary.with_suffix('.identity.json')) != frozen['identity_sha256'] or
                artifact != read(binary.with_suffix('.identity.json'))):
            raise ValueError('service evidence differs from the frozen probe or reference target')
        if any((profile['evidence'], profile['binary_sha256']) ==
               (previous['evidence'], previous['binary_sha256']) for previous in registry['profiles']):
            return base
        if epilogue_only:
            profile['measurements'] = {}
            profile['epilogue_measurements'] = measurements
            registry['profiles'].append(profile)
            return base

        def compatible_headers(headers):
            current = profile['body_headers']
            header = 'Backend/Sm90DenseGemm.h'
            proof = constants.get('residual_copy_header_equivalence')
            if (proof and headers.get(header) == proof['inactive_sha256']
                    and current.get(header) == proof['compiled_sha256']):
                headers = headers | {header: proof['compiled_sha256']}
            return headers == current

        compatible = {key: previous['binary_sha256']
                      for previous in registry['profiles']
                      if compatible_headers(previous['body_headers'])
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
    parser.add_argument('--calibration-target', type=Path)
    parser.add_argument('--maximum-perturbation', type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.maximum_perturbation <= 0.25:
        parser.error('choose a perturbation bound between 0 and 0.25')
    profile = collect(args.run, args.target, args.binary,
                      args.calibration_target, args.maximum_perturbation)
    with args.out.open('x') as stream:
        json.dump(profile, stream, indent=2)
        stream.write('\n')
