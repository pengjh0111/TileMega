#!/usr/bin/env python3
"""Adopt faster, token-identical guarded R10 measurements without retiming."""

import argparse
import csv
import math
from pathlib import Path

from a100_baselines import ROOT, inspect_inputs, sha
from a100_results import guarded_measurement, measurement, write_csv
from a800_baselines import INVENTORY, hardware_class
from a800_efficiency import calculate
from a800_native import verify_request_tokens
from a800_r10_collect_round import INVARIANTS, accounting
from a800_r10_matrix import summarize
from a800_r10_prepare import MODELS, read
from a800_r9_matrix import IDENTITY
from a800_results import native_provenance
from tilemega.bench.scheduler import write_json


def timed_row(current, artifact, raw, provenance, uuid, checks, correctness):
    metrics = {name: raw[name] for name in ('tpot_ms', 't1_s', 't1024_s', 'tok_s', 'versions')}
    return dict(current, **artifact, **provenance, **metrics, processes=1,
                accepted_correctness_processes=checks, correctness=correctness,
                receipt_sha256=raw['receipt'], gpu_uuid=uuid,
                speedup=float(current['fastest_baseline_tpot_ms'])/raw['tpot_ms'])


def request(runs, control, sample, job, key, artifact, target, inputs):
    model, batch = key
    round_id = sample['round']
    history = runs/'native'/sample['gpu_uuid']/model/f'B{batch}'
    [receipt_path] = [path for path in history.glob(f'*/tune-decode-{round_id}.json')
                      if read(path)['decode_sha256'] == artifact['decode_sha256']]
    receipt = read(receipt_path)
    device = receipt['device']
    if (not receipt['pass'] or receipt['job'] != job or receipt['device_uuid'] != sample['gpu_uuid']
            or hardware_class(device) != hardware_class(target['device_provenance'])
            or any(receipt[key] != artifact[key] for key in IDENTITY)
            or receipt['tokens_sha256'] != control['tokens_sha256']):
        raise ValueError(f'{model} B{batch} lacks an accepted artifact-bound timing receipt')
    folder = Path(receipt['evidence'])
    if sha(folder/'reference_tokens.json') != receipt['tokens_sha256']:
        raise ValueError(f'{model} B{batch} greedy token receipt changed')
    record = measurement(dict(name=f'R10 paired {model} B{batch}', model=model,
                              batch=batch, round=round_id),
                         dict(status='done', out=str(folder.parent.parent)),
                         dict(models=inputs, device=device))
    if (record['receipt'] != sample['measurement_sha256']
            or record['checkpoint_sha256'] != control['checkpoint_sha256']
            or not math.isclose(record['tpot_ms'], 1000*sample['tpot_seconds'], rel_tol=1e-7)):
        raise ValueError(f'{model} B{batch} raw request timing changed')
    return record, native_provenance(record, receipt)


def bundles(phase, scope, slots, runs):
    for group, indices in scope['groups'].items():
        if group not in slots:
            continue
        job, state, _ = slots[group]
        if state not in ('COMPLETED', 'FAILED'):
            continue
        for index in indices:
            item = scope['items'][index]
            if sha(item['plan']) != item['sha256']:
                raise ValueError('frozen R10 phase plan changed')
            plan = read(item['plan'])
            root = runs/'optimization'
            if phase == 'e1':
                folder = root/'r10-e1-bundles'/job/index
                outcome = folder/'allocation.json'
                if not outcome.is_file() or not outcome.stat().st_size or read(outcome) != {
                        'job': job, 'cell': index, 'return_code': 0}:
                    continue
            else:
                folder = root/'r10-bundles'/job/index/'timing'
            if (folder/'result.json').is_file():
                yield job, plan, folder


def adopt_pair(phase, job, plan, folder, runs, current, inputs):
    binding = read(folder/'planned.json')
    result_path = folder/'result.json'
    result = read(result_path)
    model, batch = plan['model'], plan['batch']
    key = model, batch
    if (binding['model'], binding['batch'], binding['phase'], binding['protocol_version']) != (
            model, batch, 'decode', 'a800-single-v2'):
        raise ValueError('R10 timing does not use the frozen decode protocol')
    samples = {row['arm']: row for row in read(folder/'rounds.json')}
    control = samples['control']
    if 'arms' in result:
        eligible = result['arms']
    else:
        eligible = {'trial': {}} if result['rounds'] == 2 else {}
    best = min((samples[name] for name in eligible), key=lambda row: row['tpot_seconds'], default=None)
    if best is None or best['tpot_seconds'] >= .995*min(
            control['tpot_seconds'], float(current['tpot_ms'])/1000):
        return None
    if (any(best[name] != control[name] for name in INVARIANTS)
            or any(best[name] != current[name] for name in ('checkpoint_sha256', 'config_sha256', 'prompt_sha256'))):
        raise ValueError(f'{model} B{batch} changes the paired GPU, output, or frozen inputs')
    label = best['arm']
    target = Path(plan['command'][plan['command'].index('--target')+1])
    artifact = dict(decode_sha256=binding['variant_sha256'][label],
                    prefill_sha256=binding['shared_sha256'], target_sha256=binding['target_sha256'])
    if (sha(target) != artifact['target_sha256'] or
            any(current[name] != artifact[name] for name in ('prefill_sha256', 'target_sha256'))):
        raise ValueError(f'{model} B{batch} changes a frozen prefill or target')
    if artifact['decode_sha256'] == current['decode_sha256']:
        return None
    checks = 0
    correctness = 'guarded timing, full greedy tokens match paired control'
    if phase == 'router':
        routing = read(folder/'routing.json')['passed']
        trial, prior = routing[label], routing['control']
        proof, reference = read(Path(trial['path'])), read(Path(prior['path']))
        if (sha(trial['path']) != trial['sha256'] or sha(prior['path']) != prior['sha256']
                or not proof['pass'] or not proof['hf_pass'] or not reference['pass']
                or proof['decode_sha256'] != artifact['decode_sha256']
                or proof['device_uuid'] != best['gpu_uuid']
                or proof['tokens_sha256'] != best['tokens_sha256']
                or proof['router_indices_sha256'] != reference['router_indices_sha256']):
            raise ValueError(f'{model} B{batch} changes ordered expert routing')
        checks, correctness = 1, 'paired exact greedy tokens and ordered routes, full teacher-forced passed'
    raw, provenance = request(runs, control, best, job, key, artifact, read(target), inputs)
    row = timed_row(current, artifact, raw, provenance, best['gpu_uuid'], checks, correctness)
    tag = plan.get('labels', {}).get(label, label)
    if phase == 'e1':
        tag = Path(tag).stem
    fact = dict(model=model, batch=batch, phase=phase, arm=tag, job=job,
                paired_control_ms=1000*control['tpot_seconds'], prior_tpot_ms=current['tpot_ms'],
                tpot_ms=raw['tpot_ms'], pair_sha256=sha(result_path), timing_sha256=raw['receipt'],
                decode_sha256=artifact['decode_sha256'], tokens_sha256=best['tokens_sha256'],
                path=str(result_path), correctness=correctness)
    return row, fact


def adopt_pipeline(entry, current, inputs):
    folder, job = Path(entry['folder']), entry['job']
    model, batch = current['model'], int(current['batch'])
    spec = read(folder/'selection.binding.json')
    guard = folder/f'selection-{job}'
    observed = read(guard/'planned.json')
    device = observed.pop('device')
    identity = read(guard/'job/physical-device-identity.json')
    expected = folder/'selection.plan.json'
    profile = read(Path(spec['control']['target']))
    if (not job.isdecimal() or sha(expected) != spec['receipt_sha256'] or observed != read(expected)
            or (spec['model'], spec['batch'], observed['protocol_version']) != (model, batch, 'a800-single-v2')
            or identity['compute_capability'] != [8, 0] or identity['nvml_uuid'] != device['uuid']
            or hardware_class(device) != hardware_class(profile['device_provenance'])):
        raise ValueError(f'{model} B{batch} changes the pipeline measurement binding')
    frozen = Path(spec['shortlist'])
    archive = Path(spec['control']['timing_source'])
    if sha(archive) != spec['control']['timing_sha256']:
        raise ValueError(f'{model} B{batch} changes the archived greedy reference')
    reference = read(archive)
    directory = Path(spec['selection_results'])
    receipt_path = directory/'ready-to-finalize.json'
    selected = read(receipt_path)
    stage = selected['receipt']
    timing = stage['measurement']
    if (sha(frozen) != selected['prepared_sha256'] or selected['scope'] != read(frozen)['scope']
            or selected['status'] != 'ready_to_finalize' or timing['protocol_version'] != 'a800-single-v2'
            or timing['prior_measurements'] or timing.get('split_rank_measurement')
            or timing['gpu_uuid'] != device['uuid'] or observed['target_sha256'] != current['target_sha256']
            or observed['prefill_sha256'] != current['prefill_sha256']
            or timing['checkpoint_sha256'] != current['checkpoint_sha256']
            or set(timing['launch_order']) != set(timing['arms']) or len(timing['arms']) < 2):
        raise ValueError(f'{model} B{batch} lacks a complete guarded solver measurement')
    candidates = {f"c{row['rank']}": row for row in stage['candidates'] if row['status'] == 'compiled'}
    winner = None
    statuses = []
    for index, label in enumerate(timing['launch_order']):
        case, arm = candidates[label], timing['arms'][label]
        binary = Path(case['binary'])
        if (arm['processes'] != 1 or sha(binary) != case['binary_sha256']
                or case['occupancy']['effective_residency'] < case['occupancy']['requested_residency']):
            raise ValueError(f'{model} B{batch} changed the prepared candidate or its occupancy')
        base = directory/'measure'/f'{index:02d}-{label}'/'measurements'
        path = base/f'B{batch}'/'measurements.json'
        verify_request_tokens(path, batch, 3)
        # The guarded selector binds the physical identity to the parent
        # receipt. Its nested executor writes request timings separately.
        raw = guarded_measurement(dict(name=f'{model} B{batch} {label}', model=model, batch=batch, round=index),
                                  path, guard, dict(device=device, models=inputs), selector_bound=True)
        artifact = dict(decode_sha256=case['binary_sha256'], prefill_sha256=observed['prefill_sha256'],
                        target_sha256=observed['target_sha256'])
        if (raw['receipt'] != timing['process_receipts'][label]
                or raw['checkpoint_sha256'] != current['checkpoint_sha256']
                or not math.isclose(raw['tpot_ms'], arm['tpot_ms'], rel_tol=1e-7)):
            raise ValueError(f'{model} B{batch} changes the solver request timing')
        tokens = sha(base/'reference_tokens.json')
        equal = tokens == reference['tokens_sha256']
        unchanged = case['binary_sha256'] == current['decode_sha256']
        status = 'eligible' if equal else 'rejected_token_mismatch'
        if equal and unchanged:
            status = 'incumbent_binary'
        statuses.append(dict(model=model, batch=batch, arm=label, job=job, status=status,
                             tpot_ms=raw['tpot_ms'], incumbent_tpot_ms=current['tpot_ms'],
                             gpu_uuid=device['uuid'], decode_sha256=case['binary_sha256'],
                             tokens_sha256=tokens, timing_sha256=raw['receipt'], selection_sha256=sha(receipt_path)))
        if not equal or unchanged or raw['tpot_ms'] >= .995*float(current['tpot_ms']):
            continue
        if winner is not None and raw['tpot_ms'] >= winner[0]['tpot_ms']:
            continue
        correctness = 'guarded pipeline timing, full greedy tokens match archived control'
        row = timed_row(current, artifact, raw, native_provenance(raw, artifact), device['uuid'], 0, correctness)
        fact = dict(model=model, batch=batch, phase='pipeline', arm=label, job=job,
                    paired_control_ms='', prior_tpot_ms=current['tpot_ms'], tpot_ms=raw['tpot_ms'],
                    pair_sha256=sha(receipt_path), timing_sha256=raw['receipt'], decode_sha256=artifact['decode_sha256'],
                    tokens_sha256=tokens, path=str(receipt_path), correctness=correctness)
        winner = row, fact
    return winner, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', action='append', default=[], nargs=5,
                        metavar=('NAME', 'LEDGER', 'ACCOUNTING', 'PARENT_JOB', 'RUNS'))
    parser.add_argument('--pipeline', type=Path, help='per-cell guarded solver allocation receipts')
    args = parser.parse_args()
    if not args.phase and not args.pipeline:
        parser.error('supply a paired phase or guarded pipeline receipts')
    with args.incumbent.open(newline='') as stream:
        frozen = list(csv.DictReader(stream))
    rows = {(row['model'], int(row['batch'])): row for row in frozen}
    if len(frozen) != 40 or len(rows) != 40 or any(row['status'] != 'measured' for row in frozen):
        raise ValueError('require a complete incumbent matrix')
    inputs = inspect_inputs(INVENTORY)
    accepted = {}
    for phase, ledger, accounts, parent, runs in args.phase:
        if phase not in ('e1', 'options', 'mainloop', 'router') or not parent.isdecimal():
            raise ValueError('choose a measured R10 phase and its Slurm parent ID')
        scope, slots, runs = read(Path(ledger)), accounting(Path(accounts), int(parent)), Path(runs)
        for job, plan, folder in bundles(phase, scope, slots, runs):
            key = plan['model'], plan['batch']
            if key[0] not in MODELS:
                raise ValueError('the new candidate is outside R10')
            verdict = adopt_pair(phase, job, plan, folder, runs, rows[key], inputs)
            if verdict:
                rows[key], accepted[key] = verdict
    pipeline = []
    if args.pipeline:
        for entry in read(args.pipeline).values():
            key = entry['model'], entry['batch']
            if key[0] not in MODELS:
                raise ValueError('the solver measurement is outside R10')
            verdict, statuses = adopt_pipeline(entry, rows[key], inputs)
            pipeline.extend(statuses)
            if verdict:
                rows[key], accepted[key] = verdict
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.out/'A800_llm.csv'
    write_csv(native, list(rows.values()))
    results = ROOT/'docs/results'
    efficiency = calculate(native, results/'A800_baselines.csv', results/'A800_baselines_frozen.json',
                           results/'SM90R6_hw_bound.csv', ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json')
    write_csv(args.out/'A800_R10_efficiency.csv', efficiency)
    if accepted:
        write_csv(args.out/'A800_R10_adopted.csv', list(accepted.values()))
    if pipeline:
        write_csv(args.out/'A800_R10_pipeline.csv', pipeline)
    write_json(args.out/'a800_native_selection.json',
               {f'{model}/B{batch}': {name: row[name] for name in IDENTITY}
                for (model, batch), row in rows.items()})
    write_json(args.out/'summary.json', dict(adopted=len(accepted), all=summarize(efficiency),
               active=summarize([row for row in efficiency if row['model'] in MODELS])))


if __name__ == '__main__':
    main()
