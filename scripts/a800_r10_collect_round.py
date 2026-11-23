#!/usr/bin/env python3
"""Collect paired whole-request results and separately guarded stage diagnostics."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from a100_baselines import sha
from a800_native import artifact
from a800_r10_collect_e1 import TERMINAL
from a800_r10_prepare import paired_libraries, read
from tilemega.bench.scheduler import write_json


TABLES = {'options': 'options', 'splitk': 'splitk', 'mainloop': 'mainloop', 'router': 'router'}
INVARIANTS = ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256', 'config_sha256', 'prompt_sha256')
STAGES = ('active_ctas', 'overlapping_sms', 'stage_span_ns', 'effective_gbps',
          'producer_issue_fraction', 'consumer_wait_fraction', 'epilogue_fraction',
          'grid_barrier_fraction', 'trace_perturbation')


def accounting(path, job):
    slots = {}
    for line in path.read_text().splitlines():
        name, raw, state, exit_code = line.split('|')
        match = re.fullmatch(rf'{job}_(\d+)', name)
        if match:
            slots[match[1]] = (raw, state.split()[0].rstrip('+'), exit_code)
    return slots


def guarded_stats(directory, expected, gpu):
    if not (directory/'guard/guard_result.json').is_file():
        return []
    guard = read(directory/'guard/guard_result.json')
    identity = read(directory/'job/physical-device-identity.json')
    if guard['code'] or guard['acceptance'] != 'strict' or not guard['ownership_verified']:
        return []
    if (identity['nvml_uuid'] != gpu or identity['compute_capability'] != [8, 0]
            or read(directory/'frozen.json')['identity_sha256'] != expected['identity_sha256']):
        raise ValueError('stage probe differs from the guarded allocation and frozen binary')
    result = []
    for case in read(directory/'artifact.json')['cases']:
        file = directory/f'{case}.json'
        if not file.is_file():
            continue
        data = read(file)
        accepted = data['status'] == 'measured'
        if 'trace_perturbation' in data:
            accepted &= data['sample_stride'] == 16 and data['trace_perturbation'] <= .05
        if (accepted and 'effective_gbps' in data and
                not math.isclose(data['logical_weight_bytes']/data['stage_span_ns'],
                                 data['effective_gbps'], abs_tol=.002)):
            raise ValueError('bandwidth does not equal the logical byte count divided by the measured span')
        result.append(dict(stage=case, status=data['status'] if accepted else 'rejected',
                           evidence_sha256=sha(file),
                           **{key: data.get(key, '') for key in STAGES}))
    return result


def handle_cell(plan, job, state, account_code, timing, stages):
    if not (timing/'result.json').is_file():
        return [dict(model=plan['model'], batch=plan['batch'], job=job, arm='',
                     status=state.lower(), evidence_sha256=account_code)], {}
    binding = read(timing/'planned.json')
    samples = {sample['arm']: sample for sample in read(timing/'rounds.json')}
    result = read(timing/'result.json')
    if ((binding['model'], binding['batch']) != (plan['model'], plan['batch']) or
            binding['protocol_version'] != 'a800-single-v2'):
        raise ValueError('paired timing belongs to another experiment')
    control = samples['control']
    if plan['phase'] == 'router':
        routing = read(timing/'routing.json')['passed']
        route_reference = read(routing['control']['path'])['router_indices_sha256']
        binaries = paired_libraries(plan)
    else:
        routing = {}
    rows, candidates = [], {}
    for name in binding['order']:
        label = plan['labels'].get(name, name)
        base = dict(model=plan['model'], batch=plan['batch'], job=job, arm=label,
                    binary_sha256=binding['variant_sha256'][name],
                    evidence_sha256=sha(timing/'result.json'))
        sample = samples.get(name)
        if name in result.get('rejected', {}) or sample is None:
            detail = result.get('rejected', {}).get(name)
            reason = json.dumps(detail, sort_keys=True) if detail is not None else 'missing guarded timing receipt'
            rows.append(dict(base, status='rejected', reason=reason))
            continue
        if any(sample[field] != control[field] for field in INVARIANTS):
            raise ValueError('trial changed a frozen input, GPU, or output token sequence')
        if routing:
            proof = routing[name]
            observed = read(proof['path'])
            if (sha(proof['path']) != proof['sha256'] or not observed['pass']
                    or name != 'control' and not observed['hf_pass']
                    or observed['decode_sha256'] != base['binary_sha256']
                    or observed['device_uuid'] != sample['gpu_uuid']
                    or observed['router_indices_sha256'] != route_reference):
                raise ValueError('router trial lacks its guarded ordered-index proof')
            _, identity, manifest = artifact(Path(binaries[label]))
            if identity['binary_sha256'] != base['binary_sha256']:
                raise ValueError('router geometry changed after timing')
            gemm = manifest['native_dense']['gemms']['router']
            base.update(router_tn=gemm['tile_n'], router_tk=gemm['slot_bytes']//(2*gemm['tile_n']),
                        router_indices_sha256=observed['router_indices_sha256'],
                        router_proof_sha256=proof['sha256'], teacher_forced=observed['hf_pass'])
        tpot = (statistics.median(sample['n1024_seconds']) -
                statistics.median(sample['n1_seconds']))/1023
        if tpot <= 0 or not math.isclose(tpot, sample['tpot_seconds'], rel_tol=1e-7):
            raise ValueError('trial receipt does not implement the frozen TPOT protocol')
        improvement = 1 - tpot/control['tpot_seconds']
        candidates[label] = dict(tpot_seconds=tpot, beats_control=improvement > .005,
                                 decode_sha256=base['binary_sha256'], job=job)
        rows.append(dict(base, status='measured', tpot_ms=1000*tpot,
                         improvement_pct=100*improvement, gpu_uuid=sample['gpu_uuid']))
        for run_label, diagnostic in plan.get('probes', {}).items():
            if diagnostic['arm'] != label:
                continue
            for part in guarded_stats(stages/run_label, diagnostic, sample['gpu_uuid']):
                rows.append(dict(base, split_k=diagnostic['split_k'], **part))
    winners = [name for name, row in candidates.items() if row['beats_control']]
    winner = min(winners, key=lambda name: candidates[name]['tpot_seconds']) if winners else 'control'
    state = dict(winner=winner, arms=candidates, publication='screening')
    if plan['phase'] == 'options':
        enum = [row['tpot_seconds'] for key, row in candidates.items() if key.startswith('enum-')]
        seed = [row['tpot_seconds'] for key, row in candidates.items() if key.startswith('seed-')]
        state['cost_model_blind'] = bool(seed and enum and min(seed) < .995*min(enum))
    return rows, state


def option_exclusions(plan_path, plan, job):
    for label, reason in plan.get('rejected', {}).items():
        evidence = plan_path
        status = 'prepare_rejected'
        if isinstance(reason, dict) and 'same_coordinates_as' in reason:
            status = 'same_coordinates'
            reason = f"matches {reason['same_coordinates_as']}"
        elif isinstance(reason, dict):
            reason = reason.get('reason') or f"shortlist prepare exit {reason['exit_code']}"
        else:
            log = plan_path.with_name(f'{label}.log')
            if log.is_file():
                lines = [line for line in log.read_text().splitlines()
                         if line.startswith('tilemega-compile: ')]
                if lines:
                    evidence = log
                    reason = lines[-1]
        yield dict(model=plan['model'], batch=plan['batch'], job=job, arm=label,
                   status=status, evidence_sha256=sha(evidence), reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=TABLES)
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--job', required=True, type=int)
    parser.add_argument('--accounting', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sheet = read(args.ledger)
    devices = accounting(args.accounting, args.job)
    groups = sheet.get('groups', {index: [index] for index in sheet['items']})
    if any(group not in devices or devices[group][1] not in TERMINAL for group in groups):
        return 3
    rows = [dict(model=value['model'], batch=value['batch'], status='prepare_rejected')
            for value in sheet['rejected'].values()]
    summary, pools = {}, []
    for group, indices in groups.items():
        job, state, code = devices[group]
        for index in indices:
            item = sheet['items'][index]
            plan_path = Path(item['plan'])
            if sha(plan_path) != item['sha256']:
                raise ValueError('the prepared experiment changed')
            plan = read(plan_path)
            if 'groups' in sheet:
                root = args.runs/'optimization/r10-bundles'/job/index
                timing, stages = root/'timing', root/'stages'
            else:
                timing = args.runs/'optimization/decode-pairs'/job
                stages = args.runs/'calibration/r2-supplement-service'/job
            try:
                samples, result = handle_cell(plan, job, state, code, timing, stages)
            except (KeyError, OSError, ValueError) as error:
                evidence = timing/'result.json'
                samples = [dict(model=plan['model'], batch=plan['batch'], job=job,
                                status='rejected_evidence', reason=str(error),
                                evidence_sha256=sha(evidence) if evidence.is_file() else '')]
                result = {'status': 'rejected_evidence', 'reason': str(error), 'job': job}
            rows.extend(samples)
            summary[f"{plan['model']}/B{plan['batch']}"] = result
            if args.phase == 'options':
                rows.extend(option_exclusions(plan_path, plan, job))
                search = plan_path.parent/'enum.so.search/receipt.json'
                if search.is_file():
                    prediction = read(search)
                    pipeline = prediction['pipeline']
                    pools.append(dict(model=plan['model'], batch=plan['batch'], job=job,
                                      additive_body_candidates=pipeline['additive_body_candidates'],
                                      evaluated=len(pipeline['evaluated']),
                                      shortlist=len(pipeline['shortlist']),
                                      finalists=len(pipeline['finalists']),
                                      receipt_sha256=sha(search)))
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ('model', 'batch', 'arm', 'job', 'status', 'tpot_ms', 'improvement_pct',
              'gpu_uuid', 'binary_sha256', 'stage', 'split_k', *STAGES, 'evidence_sha256', 'reason')
    if args.phase == 'router':
        fields += ('router_tn', 'router_tk', 'router_indices_sha256', 'router_proof_sha256', 'teacher_forced')
    with (args.out/f'A800_R10_{TABLES[args.phase]}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.phase == 'options':
        with (args.out/'A800_R10_candidates.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, ('model', 'batch', 'job', 'additive_body_candidates',
                                             'evaluated', 'shortlist', 'finalists', 'receipt_sha256'))
            writer.writeheader()
            writer.writerows(pools)
    write_json(args.out/f'{args.phase}.screening.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
