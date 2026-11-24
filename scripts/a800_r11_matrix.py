#!/usr/bin/env python3
"""Publish the complete A800 matrix and its R11 paired measurements."""
import csv
import math

from a100_baselines import BATCHES, MODELS, ROOT
from a100_results import write_csv
from a800_efficiency import calculate
from a800_r10_prepare import MODELS as PRIORITY
from tilemega.bench.scheduler import write_json


IDENTITY = ('decode_sha256', 'prefill_sha256', 'target_sha256')
FROZEN_INPUTS = ('checkpoint_sha256', 'config_sha256', 'prompt_sha256')


def incumbent(path):
    with path.open(newline='') as stream:
        matrix = list(csv.DictReader(stream))
    rows = {(row['model'], int(row['batch'])): row for row in matrix}
    if len(matrix) != 40 or set(rows) != {(model, batch) for model in MODELS for batch in BATCHES}:
        raise ValueError('R11 requires the complete incumbent matrix')
    if any(row['status'] != 'measured' or float(row['tpot_ms']) <= 0 for row in matrix):
        raise ValueError('provide the complete measured incumbent matrix')
    return rows


def summary(rows, efficiencies=None, denominator=None):
    ratio = [float(row['speedup']) for row in rows]
    speed = math.exp(math.fsum(map(math.log, ratio))/len(ratio))
    wins = sum(item > 1.005 for item in ratio)
    losses = sum(item < .995 for item in ratio)
    result = dict(cells=len(rows), geomean_speedup=speed, wins=wins,
                  ties=len(rows)-wins-losses, losses=losses,
                  target_tpot_reduction=max(0, 1 - speed/1.20))
    if efficiencies is not None:
        def geomean(field):
            return math.exp(math.fsum(math.log(row[field]) for row in efficiencies)/len(efficiencies))

        baseline = geomean('baseline_efficiency')
        tilemega = geomean('tilemega_efficiency')
        if not math.isclose(baseline, denominator, rel_tol=1e-5):
            raise ValueError('the measured matrix changes the frozen efficiency denominator')
        result.update(geomean_tilemega_efficiency=tilemega, geomean_baseline_efficiency=baseline,
                      speedup_from_efficiency=tilemega/denominator)
    return result


def write_results(out, original, matrix, adoption, comparisons=None):
    out.mkdir(parents=True, exist_ok=True)
    rows = list(matrix.values())
    native = out/'A800_llm.csv'
    write_csv(native, rows)
    reference = ROOT/'docs/results'
    efficiency = calculate(native, reference/'A800_baselines.csv', reference/'A800_baselines_frozen.json',
                           reference/'SM90R6_hw_bound.csv',
                           ROOT/'configs/targets/sm_80_a800_sxm4_80gb_registry.json')
    floors = {(row['model'], row['batch']): row for row in efficiency}
    comparisons = adoption if comparisons is None else comparisons
    write_csv(out/'A800_R11_llm.csv', [dict(model=row['model'], batch=row['batch'],
        scope='priority' if row['model'] in PRIORITY else 'other',
        r10_tpot_ms=original[key]['tpot_ms'], r11_tpot_ms=row['tpot_ms'],
        fastest_baseline=row['fastest_baseline'], fastest_baseline_tpot_ms=row['fastest_baseline_tpot_ms'],
        speedup=row['speedup'], efficiency=floors[key]['tilemega_efficiency'],
        baseline_efficiency=floors[key]['baseline_efficiency'],
        same_allocation_control_ms=comparisons.get(key, {}).get('control_tpot_ms', ''),
        same_allocation_trial_ms=comparisons.get(key, {}).get('tpot_ms', ''),
        same_allocation_result=comparisons.get(key, {}).get('status', ''),
        paired_fresh_pass_rate=comparisons.get(key, {}).get('pass_rate', ''),
        paired_trial_sha256=comparisons.get(key, {}).get('decode_sha256', ''),
        paired_evidence=comparisons.get(key, {}).get('evidence', ''),
        job=comparisons.get(key, {}).get('job', ''), correctness=row['correctness'],
        decode_sha256=row['decode_sha256'], target_sha256=row['target_sha256']) for key, row in matrix.items()])
    if adoption:
        write_csv(out/'A800_R11_adopted.csv', list(adoption.values()))
    value = dict(priority=summary([row for row in rows if row['model'] in PRIORITY],
        [row for row in efficiency if row['model'] in PRIORITY], .760021),
        all=summary(rows, efficiency, .630470))
    write_json(out/'a800_native_selection.json', {f'{model}/B{batch}': {key: row[key] for key in IDENTITY}
                                                  for (model, batch), row in matrix.items()})
    write_json(out/'A800_R11_summary.json', value)
    return value
