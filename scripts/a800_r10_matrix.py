#!/usr/bin/env python3
"""Regenerate the frozen matrix from accepted R10 same-allocation promotions."""
import argparse
import csv
import json
import math
from pathlib import Path

from a100_baselines import BATCHES, ROOT, inspect_inputs, sha
from a100_results import measurement, write_csv
from a800_baselines import INVENTORY, RUN
from a800_efficiency import calculate
from a800_r10_prepare import MODELS, read
from a800_r9_matrix import IDENTITY, accepted_check, matching
from a800_results import native_provenance
from tilemega.bench.scheduler import write_json


def verify_pair(old, folder, job, result, inputs):
    binding = read(folder/'planned.json')
    control, trial = read(folder/'rounds.json')
    pair = {'decode_sha256': binding['variant_sha256']['trial'],
            'prefill_sha256': binding['shared_sha256'], 'target_sha256': binding['target_sha256']}
    comparisons = result['paired_tpot_ratios']
    passed = (len(comparisons) == 3 and min(comparisons) > 1
              and trial['tpot_seconds'] < .995*control['tpot_seconds'])
    if (not result['formal_timing'] or result['protocol_version'] != 'a800-single-v2'
            or result['rounds'] != 2 or [control['arm'], trial['arm']] != ['control', 'trial']
            or len(comparisons) != 3 or result['clears_tpot_screen'] != passed
            or old['decode_sha256'] != binding['variant_sha256']['control']
            or old['target_sha256'] != pair['target_sha256'] or old['prefill_sha256'] != pair['prefill_sha256']
            or any(control[key] != trial[key] for key in ('gpu_uuid', 'tokens_sha256', 'checkpoint_sha256',
                                                          'prompt_sha256', 'config_sha256'))
            or any(old[key] != trial[key] for key in ('checkpoint_sha256', 'prompt_sha256', 'config_sha256'))):
        raise ValueError('formal A/B changed an incumbent input, its acceptance gate, or the process budget')
    model, batch = old['model'], int(old['batch'])
    history = RUN/'native'/trial['gpu_uuid']/model/f'B{batch}'
    paths = list(history.glob(f"*/candidate-timing-{trial['round']}.json"))
    receipt, = [read(path) for path in paths if matching(read(path), pair)]
    proofs = [read(path) for path in history.glob('*/correctness-*.json')]
    fresh = [proof for proof in proofs if proof['job'] == job and matching(proof, pair)]
    if (len({proof['round'] for proof in fresh}) != 5 or len(fresh) != 5
            or sum(proof['hf_pass'] for proof in fresh) != 1
            or any(not proof['pass'] or proof['tokens_sha256'] != trial['tokens_sha256']
                   or not accepted_check(proof) for proof in fresh)
            or not receipt['pass'] or receipt['job'] != job or receipt['tokens_sha256'] != trial['tokens_sha256']):
        raise ValueError('formal comparison lacks five valid fresh checks and one full teacher-forced run')
    raw = measurement(dict(model=model, batch=batch, name=f'R10 {model} B{batch}'),
                      dict(status='done', out=str(Path(receipt['evidence']).parent.parent)),
                      dict(models=inputs, device=receipt['device']))
    if (raw['receipt'] != trial['measurement_sha256']
            or raw['checkpoint_sha256'] != old['checkpoint_sha256']
            or not math.isclose(raw['tpot_ms'], 1000*trial['tpot_seconds'], rel_tol=1e-7)):
        raise ValueError('the raw request receipt differs from the paired trial')
    metrics = {name: raw[name] for name in ('tpot_ms', 't1_s', 't1024_s', 'tok_s', 'versions')}
    value = dict(old, **metrics, **pair, **native_provenance(raw, receipt),
                 processes=1, accepted_correctness_processes=5,
                 correctness='5 independent checks, full HF and step/loop passed',
                 receipt_sha256=raw['receipt'], gpu_uuid=trial['gpu_uuid'],
                 speedup=float(old['fastest_baseline_tpot_ms'])/raw['tpot_ms'])
    return value if passed else old


def summarize(rows):
    keys = ('speedup', 'tilemega_efficiency', 'baseline_efficiency')
    geomean = {key: math.exp(math.fsum(math.log(row[key]) for row in rows)/len(rows)) for key in keys}
    return dict(cells=len(rows), geomean=geomean,
                wins=sum(row['comparison'] == 'win' for row in rows),
                ties=sum(row['comparison'] == 'tie' for row in rows),
                losses=sum(row['speedup'] < 1 for row in rows),
                loss_outside_tie=sum(row['comparison'] == 'loss' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('incumbent', 'accounting', 'out'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    with args.incumbent.open(newline='') as stream:
        old_rows = list(csv.DictReader(stream))
    if len(old_rows) != 40 or any(row['status'] != 'measured' for row in old_rows):
        raise ValueError('require the published 40-cell incumbent')
    scope = {f'{model}/B{batch}' for model in MODELS for batch in BATCHES}
    trials = read(args.accounting)['formal']
    if trials.keys() != scope:
        raise ValueError('the formal round has unresolved model/batch cells')
    inputs = inspect_inputs(INVENTORY)
    rows, selection, decisions = [], {}, {}
    for old in old_rows:
        cell = f"{old['model']}/B{old['batch']}"
        row = dict(old)
        if cell in scope:
            trial = trials[cell]
            path = Path(trial['receipt'])
            if (trial['state'] != 'COMPLETED' or not path.is_file() or sha(path) != trial['sha256']):
                raise ValueError(f'{cell} lacks a completed formal receipt')
            paired = read(path)
            row = verify_pair(old, path.parent, trial['job'], paired, inputs)
            decisions[cell] = dict(job=trial['job'], admitted=paired['clears_tpot_screen'],
                                   timing_sha256=trial['sha256'], prior_tpot_ms=float(old['tpot_ms']),
                                   final_tpot_ms=float(row['tpot_ms']))
        rows.append(row)
        selection[cell] = {key: row[key] for key in IDENTITY}
    if len(selection) != 40:
        raise ValueError('duplicate incumbent model/batch cells')
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.out/'A800_llm.csv'
    write_csv(native, rows)
    results = ROOT/'docs/results'
    efficiency = calculate(native, results/'A800_baselines.csv', results/'A800_baselines_frozen.json',
                           results/'SM90R6_hw_bound.csv', ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json')
    write_csv(args.out/'A800_R10_efficiency.csv', efficiency)
    write_json(args.out/'a800_native_selection.json', selection)
    active = [row for row in efficiency if row['model'] in MODELS]
    write_json(args.out/'formal-summary.json', dict(active=summarize(active), all=summarize(efficiency),
                                                   comparisons=decisions))


if __name__ == '__main__':
    main()
