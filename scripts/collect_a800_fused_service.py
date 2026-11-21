#!/usr/bin/env python3
"""Bind low-perturbation Gu/Dn local service to its guarded production geometry."""
import argparse
import json
import math
from pathlib import Path

from collect_a800_dense_service import read, sha

MODIFIERS = ('TILEMEGA_SM80_MOE_PAGE_SEGMENTS', 'TILEMEGA_SM80_ACTIVE_EPILOGUE',
             'TILEMEGA_SM80_COPY_ISSUE_GROUP')
SHAPE = ('batch', 'gate_up_n', 'down_n', 'hidden', 'gate_up_tn', 'gate_up_tk',
         'down_tn', 'down_tk', 'slots', 'gate_up_warps_n', 'down_warps_n', 'experts',
         'float_routes', 'page', 'register_stages', 'softmax_all', 'segments', 'live_rows', 'copy_group')


def validate_payload_anchor(path, artifact, shape, measurement):
    from tilemega.build.identity import verify

    anchor = read(path)
    binary = Path(anchor['decode'])
    identity = verify(binary)
    coordinates = identity['plan']['native_dense']
    if (sha(path) != artifact['anchor_sha256'] or anchor['decode_sha256'] != identity['binary_sha256']
            or anchor['coordinates'] != coordinates or coordinates.get('payload') != 'reuse'
            or not coordinates.get('expert_fuse') or coordinates['residency'] != 1
            or coordinates['weight_stream'] or coordinates.get('route', 'local') != 'local'
            or coordinates.get('combine', 'la') != 'la'
            or identity['plan']['runtime_batch'] != shape['batch']
            or identity['shared_memory_bytes'] != measurement['dynamic_shared']
            or bool(shape['segments']) != (coordinates.get('expert_page_copy') == 'segments')
            or shape['copy_group'] != coordinates.get('copy_issue_group', 0)
            or bool(shape['live_rows']) != (coordinates.get('epilogue_rows') == 'live')
            or bool(shape['gu_route_bases']) != (coordinates.get('expert_gu_routes') == 'bases')):
        raise ValueError('payload calibration must match the linked production layout')
    for family, key in (('expert_gate_up', 'gate_up'), ('expert_down', 'down')):
        tile = coordinates['gemms'][family]
        if (tile['tile_n'] != shape[f'{key}_tn']
                or tile['slot_bytes'] != 2 * shape[f'{key}_tn'] * shape[f'{key}_tk']
                or tile['warps_n'] != shape[f'{key}_warps_n']
                or tile['warps_k'] != shape[f'{key}_warps_k']
                or tile['slots'] != shape['slots'] or tile['register_stages'] != shape['register_stages']):
            raise ValueError('payload calibration probe retiled the linked expert stage')
    return dict(anchor_sha256=sha(path), binary_sha256=identity['binary_sha256'],
                manifest_sha256=identity['manifest_sha256'],
                dynamic_shared=identity['shared_memory_bytes'])


def append(base, target_sha, run, binary, payload_anchor=None):
    artifact = read(run/'artifact.json')
    if artifact.get('production_payload') == 'reuse' and payload_anchor is None:
        raise ValueError('shared-payload calibration requires its linked production anchor')
    identity = binary.with_suffix('.identity.json')
    guard = read(run/'guard/guard_result.json')
    device = read(run/'job/physical-device-identity.json')
    policy = run/'policy.json'
    frozen = read(run/'frozen.json')
    resources = base['resources']
    registry = base['sm80_native_constants'].setdefault(
        'fused_expert_service', dict(schema=1, reference_target_sha256=target_sha, profiles=[]))
    device_class = {
        'sm_count': 'num_sms', 'shared_memory_per_sm': 'max_smem_per_sm',
        'shared_memory_per_block_optin': 'max_dynamic_smem_per_cta',
        'registers_per_sm': 'regs_per_sm', 'l2_bytes': 'l2_bytes',
    }
    if (base['arch_tag'] != 'sm_80' or 'A800' not in base['device_provenance']['name']
            or any(device[k] != resources[v] for k, v in device_class.items())
            or device['compute_capability'] != [8, 0]
            or guard['code'] or guard['acceptance'] != 'strict' or not guard['ownership_verified']
            or device['nvml_uuid'] != read(policy)['gpu_uuid']
            or device['guard_policy_sha256'] != sha(policy)
            or sha(identity) != frozen['identity_sha256'] or artifact != read(identity)
            or frozen['target_sha256'] not in (registry['reference_target_sha256'], target_sha)
            or sha(binary) != artifact['binary_sha256']
            or artifact['contract'] != 'sm80-fused-expert-service-v1'
            or artifact['architecture'] != 'sm_80' or artifact['roles'] != [32, 128]):
        raise ValueError('fused service differs from the guarded production geometry')
    case, = artifact['cases']
    if Path(case).name != case:
        raise ValueError('fused service case must be a local path')
    measurement = run/f'{case}.json'
    row = read(measurement)
    fields = ('gu_issue_ns_per_tile', 'gu_mma_ns_per_tile', 'gu_epilogue_ns_per_task',
              'dn_mma_ns_per_tile', 'dn_epilogue_ns_per_task',
              'dn_preissue_ns_per_task', 'dn_activation_ns_per_task')
    params = artifact['body_parameters']
    if (row['status'] != 'measured' or not row['bitwise_equal'] or len(params) != 17
            or row['sampled_ctas_per_launch'] != 1
            or row['grid'] != resources['num_sms'] or row['batch'] != params[0]
            or not params[-1] or row['dynamic_shared'] != params[-1]
            or any(row[f'{mode}_residency'] != 1 for mode in ('plain', 'producer', 'consumer'))
            or any(row[f'{mode}_local_bytes'] != row['plain_local_bytes']
                   for mode in ('producer', 'consumer'))
            or any(not math.isfinite(row[field]) or row[field] <= 0 for field in fields)
            or not math.isfinite(row['dn_issue_ns_per_tile']) or row['dn_issue_ns_per_tile'] < 0):
        raise ValueError('fused service has different residency, arena, or invalid local times')
    # Bound both the probe dispatch effect and the additional timing writes.
    modes = ('plain', 'producer', 'consumer', 'producer_off', 'consumer_off')
    observation = {mode: row[f'{mode}_kernel_ns'] for mode in modes}
    timing = row.get('phase_timing_ns')
    if artifact['source_modifiers'].get('TILEMEGA_FUSED_SERVICE_SPLIT_PHASES'):
        if not isinstance(timing, list) or len(timing) != 4 or row['launches'] != 128:
            raise ValueError('split-phase calibration requires four independent 32-launch comparisons')
    for sample in [observation, *(timing or [])]:
        plain = sample['plain']
        for role in ('producer', 'consumer'):
            measured, dispatch = (sample[mode] for mode in (role, f'{role}_off'))
            if (any(not math.isfinite(value) or value <= 0 for value in (plain, dispatch, measured))
                    or max(abs(dispatch/plain - 1), abs(measured/dispatch - 1)) > 0.05):
                raise ValueError(f'{role} service instrumentation exceeds 5% of the stage')
    mapping = {'true': 1, 'false': 0}
    source = artifact['source_modifiers']
    dimensions = [mapping[value] if value in mapping else int(value) for value in params[:-1]]
    dimensions.extend(source.get(name, 0) for name in MODIFIERS)
    shape = dict(zip(SHAPE, dimensions, strict=True))
    shape['gu_route_bases'] = source.get('TILEMEGA_SM80_MOE_GU_ROUTE_BASES', 0)
    if payload_anchor is not None:
        shape['reuse_payload'] = 1
    shape['gate_up_warps_k'] = 4 // shape['gate_up_warps_n']
    shape['down_warps_k'] = 4 // shape['down_warps_n']
    if registry['schema'] != 1:
        raise ValueError('unknown fused service registry schema')
    profile = dict(shape=shape, measurement=row, source_sha=artifact['source_sha'],
                   binary_sha256=artifact['binary_sha256'], source_modifiers=source,
                   body_headers=artifact['body_headers'], reference_target_sha256=frozen['target_sha256'],
                   measurement_sha256=sha(measurement), gpu_uuid=device['nvml_uuid'],
                   guard_sha256=sha(run/'guard/guard_result.json'), evidence=str(run))
    if payload_anchor is not None:
        profile['payload_anchor'] = validate_payload_anchor(payload_anchor, artifact, shape, row)
    for previous in registry['profiles']:
        if all(previous.get(key) == value for key, value in profile.items()):
            return
        previous_shape = dict(gu_route_bases=0, reuse_payload=0)
        previous_shape.update(previous['shape'])
        if previous_shape != {'reuse_payload': 0, **shape}:
            continue
        old_headers = previous['body_headers']
        headers = profile['body_headers']
        moe_header = 'Backend/Sm90DenseMoe.cuh'
        proof = previous.get('moe_header_equivalence')
        if (proof and proof['reference_sha256'] == old_headers[moe_header]
                and proof['equivalent_sha256'] == headers[moe_header]
                and len(proof['preprocessed_sha256']) == 64):
            old_headers = old_headers | {moe_header: proof['equivalent_sha256']}
        if old_headers == headers:
            raise ValueError('retain the selected observation for this compatible fused geometry')
    registry['profiles'].append(profile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--observation', nargs=2, action='append', metavar=('RUN', 'BINARY'), required=True)
    parser.add_argument('--payload-anchor', type=Path,
                        help='linked production anchor for one shared-payload observation')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.payload_anchor and len(args.observation) != 1:
        parser.error('a shared-payload anchor binds one observation')
    target = read(args.target)
    for run, binary in args.observation:
        append(target, sha(args.target), Path(run), Path(binary), args.payload_anchor)
    with args.out.open('x') as stream:
        json.dump(target, stream, indent=2)
        stream.write('\n')
