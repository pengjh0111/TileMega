#!/usr/bin/env python3
"""Adopt faster, token-identical guarded R10 measurements without retiming."""

import argparse
import csv
import math
from pathlib import Path

from a100_baselines import ROOT, inspect_inputs, sha
from a100_results import measurement, write_csv
from a800_baselines import INVENTORY, hardware_class
from a800_efficiency import calculate
from a800_r10_collect_round import INVARIANTS, accounting
from a800_r10_matrix import summarize
from a800_r10_prepare import MODELS, read
from a800_r9_matrix import IDENTITY
from a800_results import native_provenance
from tilemega.bench.scheduler import write_json


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
    metrics = {name: raw[name] for name in ('tpot_ms', 't1_s', 't1024_s', 'tok_s', 'versions')}
    row = dict(current, **artifact, **provenance, **metrics, processes=1,
               accepted_correctness_processes=checks, correctness=correctness,
               receipt_sha256=raw['receipt'], gpu_uuid=best['gpu_uuid'],
               speedup=float(current['fastest_baseline_tpot_ms'])/raw['tpot_ms'])
    tag = plan.get('labels', {}).get(label, label)
    fact = dict(model=model, batch=batch, phase=phase, arm=tag, job=job,
                paired_control_ms=1000*control['tpot_seconds'], prior_tpot_ms=current['tpot_ms'],
                tpot_ms=raw['tpot_ms'], pair_sha256=sha(result_path), timing_sha256=raw['receipt'],
                decode_sha256=artifact['decode_sha256'], tokens_sha256=best['tokens_sha256'],
                path=str(result_path), correctness=correctness)
    return row, fact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', action='append', required=True, nargs=5,
                        metavar=('NAME', 'LEDGER', 'ACCOUNTING', 'PARENT_JOB', 'RUNS'))
    args = parser.parse_args()
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
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.out/'A800_llm.csv'
    write_csv(native, list(rows.values()))
    results = ROOT/'docs/results'
    efficiency = calculate(native, results/'A800_baselines.csv', results/'A800_baselines_frozen.json',
                           results/'SM90R6_hw_bound.csv', ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json')
    write_csv(args.out/'A800_R10_efficiency.csv', efficiency)
    if accepted:
        write_csv(args.out/'A800_R10_adopted.csv', list(accepted.values()))
    write_json(args.out/'a800_native_selection.json',
               {f'{model}/B{batch}': {name: row[name] for name in IDENTITY}
                for (model, batch), row in rows.items()})
    write_json(args.out/'summary.json', dict(adopted=len(accepted), all=summarize(efficiency),
               active=summarize([row for row in efficiency if row['model'] in MODELS])))


if __name__ == '__main__':
    main()
