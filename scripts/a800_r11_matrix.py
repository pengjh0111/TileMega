#!/usr/bin/env python3
"""Publish the complete A800 matrix and its R11 paired measurements."""
import csv
import math

from a100_baselines import BATCHES, MODELS
from a100_results import write_csv
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


def summary(rows):
    ratio = [float(row['speedup']) for row in rows]
    speed = math.exp(math.fsum(map(math.log, ratio))/len(ratio))
    wins = sum(item > 1.005 for item in ratio)
    losses = sum(item < .995 for item in ratio)
    return dict(cells=len(rows), geomean_speedup=speed, wins=wins,
                ties=len(rows)-wins-losses, losses=losses,
                target_tpot_reduction=max(0, 1 - speed/1.20))


def write_results(out, original, matrix, adoption):
    out.mkdir(parents=True, exist_ok=True)
    rows = list(matrix.values())
    write_csv(out/'A800_llm.csv', rows)
    write_csv(out/'A800_R11_llm.csv', [dict(model=row['model'], batch=row['batch'],
        scope='priority' if row['model'] in PRIORITY else 'other',
        r10_tpot_ms=original[key]['tpot_ms'], r11_tpot_ms=row['tpot_ms'],
        fastest_baseline=row['fastest_baseline'], fastest_baseline_tpot_ms=row['fastest_baseline_tpot_ms'],
        speedup=row['speedup'], same_allocation_control_ms=adoption.get(key, {}).get('control_tpot_ms', ''),
        same_allocation_trial_ms=adoption.get(key, {}).get('tpot_ms', ''),
        job=adoption.get(key, {}).get('job', ''), correctness=row['correctness'],
        decode_sha256=row['decode_sha256'], target_sha256=row['target_sha256']) for key, row in matrix.items()])
    if adoption:
        write_csv(out/'A800_R11_adopted.csv', list(adoption.values()))
    value = dict(priority=summary([row for row in rows if row['model'] in PRIORITY]), all=summary(rows))
    write_json(out/'A800_R11_summary.json', value)
    return value
