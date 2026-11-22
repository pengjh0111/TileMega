#!/usr/bin/env python3
"""Measure exposed startup and tail inside a guarded SM80 fused stage."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics

from collect_a800_dense_service import read, sha
from collect_a800_fused_service import SHAPE
from prepare_a800_fused_service import BODY_HEADERS


def guarded(run):
    result = read(run/'guard/guard_result.json')
    plan = read(run/'planned.json')
    if (result['code'] != 0 or result['acceptance'] != 'strict' or not result['ownership_verified']
            or plan['device']['name'] != 'NVIDIA A800-SXM4-80GB'):
        raise ValueError('phase pricing requires a strictly admitted A800 observation')
    return plan


def phase_widths(run, grid, first, steps, body_only=False):
    path = run/'ledger/stage_trace.tsv'
    with (run/'ledger/runtime_stages.tsv').open() as stream:
        fused = {int(row['stage']) for row in csv.DictReader(stream, delimiter='\t')
                 if 'mlp.experts.gate_up' in row['name']}
    if not fused or not steps:
        raise ValueError('trace has no fused expert stages or sampled steps')
    points = ('worker', 't_begin', 't_tasks_end')
    if not body_only:
        points += ('route_ready', 'active_experts', 'm2', 'm4', 'm5', 'first_dn_preissue', 'last_dn_epilogue')
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            past, stage = int(row['past']), int(row['stage'])
            if first <= past < first+steps and (stage in fused or stage-1 in fused or stage+1 in fused):
                groups[past, stage].append(tuple(int(row[key]) for key in points))
    index = {key: i for i, key in enumerate(points)}
    samples = defaultdict(list)
    for past in range(first, first+steps):
        for stage in sorted(fused):
            rows = groups[past, stage]
            if len(rows) != grid or {r[0] for r in rows} != set(range(grid)):
                raise ValueError(f'incomplete CTA trace at {past=}, {stage=}')

            def marked(name):
                values = [r[index[name]] for r in rows if r[index[name]]]
                if not values:
                    raise ValueError(f'missing {name} at {past=}, {stage=}')
                return values

            begin, end = min(marked('t_begin')), max(marked('t_tasks_end'))
            previous, following = groups[past, stage-1], groups[past, stage+1]
            if (len(previous) != grid or len(following) != grid
                    or max(r[index['t_tasks_end']] for r in previous) > begin
                    or min(r[index['t_begin']] for r in following) < end):
                raise ValueError(f'outside task overlaps fused region at {past=}, {stage=}')
            samples['body_ns'].append(end-begin)
            if body_only:
                continue
            route = marked('route_ready')
            active = marked('active_experts')
            if len(route) != grid or len(active) != grid or len(set(active)) != 1:
                raise ValueError('route state differs among CTAs')
            samples['route_startup_ns'].append(min(route)-begin)
            samples['combine_tail_ns'].append(max(marked('m5'))-max(marked('m4')))
            samples['active_experts'].append(active[0])
            samples['dn_preissue_entry_before_last_gu_ns'].append(
                max(marked('m2'))-min(marked('first_dn_preissue')))
            samples['last_dn_to_stage_end_ns'].append(end-max(marked('last_dn_epilogue')))
    return {key: dict(min=min(group), median=statistics.median(group), max=max(group),
                      mean=statistics.mean(group)) for key, group in samples.items()}, len(fused)*steps


def artifact(binary):
    identity = read(Path(f'{binary}.identity.json'))
    if sha(binary) != identity['binary_sha256']:
        raise ValueError(f'binary changed at {binary}')
    return identity


def append(target, run, control, reference, trace, body_only=False, device_proof=None):
    measured, prior = (guarded(p) for p in (run, control))
    baseline, probe = (artifact(p) for p in (reference, trace))
    recipe = read(Path(f'{trace}.diagnostic.json'))
    if body_only:
        paired = read(run/'ledger/run.json')
        if (run.resolve() != control.resolve() or measured['batch'] != 1 or not measured['independent_arms']
                or not paired['full_token_equivalence'] or recipe.get('sm80_moe_phases')
                or probe['defines'].get('TILEMEGA_TRACE_STAGE') != '1'):
            raise ValueError('direct fused-stage pricing requires same-card B1 boundary-only measurement')
    else:
        evidence = ('guard/guard_result.json', 'planned.json', 'ledger/run-reference.json',
                    'ledger/reference-tokens.json')
        if (Path(measured['reference_ledger']).parts[-3:] != (control/'ledger').resolve().parts[-3:]
                or measured['reference_evidence'] != [sha(control/name) for name in evidence]
                or not recipe['sm80_moe_phases']):
            raise ValueError('separate phase trace has no bound guarded reference')
    if (any(measured[key] != prior[key] for key in ('model', 'batch', 'reference', 'prefill', 'target'))
            or probe['binary_sha256'] != measured['trace'] or baseline['binary_sha256'] != prior['reference']
            or baseline['plan'] != probe['plan']
            or any(baseline[key] != probe[key] for key in ('manifest_sha256', 'generated_source_sha256'))
            or baseline['trace'] or not probe['trace']
            or baseline['artifact_id'] != recipe['reference_artifact_id']
            or probe['artifact_id'] != recipe['trace_artifact_id']):
        raise ValueError('trace and reference do not bind the same generated production plan')
    obs = read(run/'ledger/run-trace.json')
    base = read(control/'ledger/run-reference.json')
    if (obs['checkpoint'] != base['checkpoint']
            or sha(run/'ledger/trace-tokens.json') != sha(control/'ledger/reference-tokens.json')):
        raise ValueError('phase trace did not reproduce the original checkpoint and complete tokens')
    ratio = obs['trace']['window_mean_ms']/base['reference']['window_mean_ms']
    if abs(ratio-1) > 0.05:
        raise ValueError('trace window outside 5% of the guarded reference')

    source = Path(f'{reference}.cu')
    if sha(source) != baseline['generated_source_sha256']:
        raise ValueError('generated reference source changed')
    aliases = re.findall(r'\bMoeFused<([^>]+)>', source.read_text())
    if not aliases:
        raise ValueError('reference has no fused expert bodies')
    fields = aliases[0].split(',')
    if (fields[-3:] != ['true', 'false', 'true'] or int(fields[13]) != 16 or int(fields[15]) != 8
            or any(alias.split(',')[:18] != fields[:18] for alias in aliases)):
        raise ValueError('unsupported mixture of fused production bodies')
    integer = lambda value: {'true': 1, 'false': 0}[value] if value in ('true', 'false') else int(value)
    values = [integer(fields[i]) for i in (0,1,2,3,4,5,6,7,8,9,11,14,16,17,-6,-4)]
    defines = baseline['defines']
    values.extend(int(defines.get(key, 0)) for key in (
        'TILEMEGA_SM80_MOE_PAGE_SEGMENTS', 'TILEMEGA_SM80_ACTIVE_EPILOGUE', 'TILEMEGA_SM80_COPY_ISSUE_GROUP'))
    shape = dict(zip(SHAPE, values, strict=True))
    shape['gu_route_bases'] = int(defines.get('TILEMEGA_SM80_MOE_GU_ROUTE_BASES', 0))
    stats, count = phase_widths(run, baseline['plan']['grid'], 567, len(obs['trace']['window_step_ms']), body_only)
    provenance = dict(schema=1, unit='ns', fused_stages=count, window_over_reference=ratio,
                      same_gpu=measured['device']['uuid'] == prior['device']['uuid'],
                      trace_job=run.name, reference_job=control.name, checkpoint_sha256=obs['checkpoint'],
                      reference_artifact_id=baseline['artifact_id'], trace_artifact_id=probe['artifact_id'],
                      trace_sha256=sha(run/'ledger/stage_trace.tsv'), token_sha256=sha(run/'ledger/trace-tokens.json'),
                      overlapping_external_stages=0, ranges=stats)
    if body_only:
        if baseline['plan']['residency'] != 1:
            raise ValueError('direct fused-stage calibration requires single residency')
        profile_target = target['sm80_native_constants'].setdefault(
            'fused_expert_service', dict(schema=1, reference_target_sha256=baseline['target_sha256'],
                                         profiles=[]))
        shape.update(gate_up_warps_k=int(fields[10]), down_warps_k=int(fields[12]))
        headers = {name: baseline['source']['source_files'][f'include/tilemega/{name}']
                   for name in BODY_HEADERS}
        body = dict(provenance, active_experts=8, body_ns=stats['body_ns']['median'])
        if device_proof:
            comparison = read(device_proof)
            if (comparison['reference_artifact_id'] != baseline['artifact_id']
                    or comparison['reference_sha256'] != baseline['binary_sha256']
                    or comparison['generated_source_sha256'] != baseline['generated_source_sha256']
                    or not comparison['fused_and_entry_resource_contract']
                    or any('Sm90DenseAttentionBody' not in name
                           for name in comparison['changed_functions'] + comparison['changed_instruction_functions'])):
                raise ValueError('rebuilt production fused stage or caller is different')
            body['device_equivalence'] = dict(
                scope='fused stage and native caller', comparison_sha256=sha(device_proof),
                rebuilt_sha256=comparison['rebuilt_sha256'], instructions=comparison['instruction_functions'],
                compiled_body_headers={name: comparison['compiled_headers'][f'include/tilemega/{name}']
                                       for name in headers})
        profile = dict(shape=shape, body_headers=headers, production_body=body,
                       source_sha=baseline['source']['head'], binary_sha256=baseline['binary_sha256'],
                       measurement=dict(grid=baseline['plan']['grid'], plain_residency=1,
                                        dynamic_shared=baseline['shared_memory_bytes']))
        if baseline['plan']['native_dense'].get('payload') == 'reuse':
            shape['reuse_payload'] = 1
            profile['payload_anchor'] = dict(binary_sha256=baseline['binary_sha256'],
                                             manifest_sha256=baseline['manifest_sha256'],
                                             dynamic_shared=baseline['shared_memory_bytes'])
        moe = 'Backend/Sm90DenseMoe.cuh'
        for other in profile_target['profiles']:
            proof = other.get('moe_header_equivalence')
            if proof and proof['reference_sha256'] == headers[moe]:
                profile['moe_header_equivalence'] = proof
                break
        if profile not in profile_target['profiles']:
            profile_target['profiles'].append(profile)
        return
    profile_target = target['sm80_native_constants']['fused_expert_service']
    profile, = (p for p in profile_target['profiles']
                if all(p['shape'].get(key, 0) == value for key, value in shape.items())
                and p['measurement']['dynamic_shared'] == baseline['shared_memory_bytes']
                and 'production_body' not in p)

    phases = {key: stats[key]['median'] for key in ('route_startup_ns', 'combine_tail_ns')}
    phases.update(provenance)
    if 'production_phases' in profile and profile['production_phases'] != phases:
        raise ValueError('retain the previously selected production phase evidence')
    profile['production_phases'] = phases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'run', 'control', 'reference', 'trace', 'out'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--whole-body', action='store_true')
    parser.add_argument('--device-proof', type=Path, help='equivalent rebuilt fused device code')
    args = parser.parse_args()
    if args.device_proof and not args.whole_body:
        parser.error('a device equivalence proof is only used for the matching production body')
    document = read(args.target)
    append(document, args.run, args.control, args.reference, args.trace, args.whole_body, args.device_proof)
    with args.out.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
