#!/usr/bin/env python3
"""Authorize final native selection after all seven E-3 cohorts pass."""
import argparse
import csv
import math
from pathlib import Path

from sm90d1_common import ROOT, read, sha, write

CELLS = {('qwen3_0_6b', 1), ('qwen3_0_6b', 16),
         ('llama3_2_1b', 1), ('llama3_2_1b', 16), ('llama3_2_3b', 4),
         ('llama3_1_8b', 1), ('llama3_1_8b', 16)}
KINDS = {'qkv', 'attention', 'o', 'gate_up', 'down', 'lm_head'}


def finite_bound(value, limit):
    return math.isfinite(float(value)) and abs(float(value)) <= limit


def check_metrics(ranking, receipt, classes):
    cell = ranking['model'], int(ranking['batch'])
    if cell != (receipt['model'], int(receipt['batch'])):
        raise ValueError('request and class observations name different cells')
    if (not ranking['ranking_pass'] or not ranking['complete'] or ranking['retrospective']
            or ranking['reused_training_receipts'] or ranking['prediction_alias_inconsistencies']
            or ranking['distinct_programs'] < 8 or not ranking['top3_contains_fastest']
            or not math.isfinite(float(ranking['kendall_tau'])) or ranking['kendall_tau'] < .7):
        raise ValueError('E-3 requires a fresh complete eight-program ranking and fastest in top three')
    candidates = ranking['candidates']
    required = 3 if cell[0] == 'llama3_1_8b' else 1
    if len({r['instruction_sha256'] for r in candidates}) < 8 or any(
            not r['protocol_complete'] or r['processes'] < required
            or not r['correctness']['pass_'] or not finite_bound(r['step_error_pct'], 10)
            for r in candidates):
        raise ValueError('E-3 request error or development protocol is incomplete')
    if (not receipt['class_pass'] or len(classes) != 6 or {r['kind'] for r in classes} != KINDS
            or any(not finite_bound(r['error_pct'], 15) for r in classes)
            or receipt['binary_sha256'] not in {r['binary_sha256'] for r in candidates}):
        raise ValueError('E-3 requires all six class errors within 15% on a ranked program')
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--ranking', type=Path, nargs='+', required=True)
    parser.add_argument('--classes', type=Path, nargs='+', required=True,
                        help='class directories containing receipt.json and classes.csv')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    target = read(args.target)
    calibration = target['sm90_dense_decode']
    if calibration['fit_status'] != 'initial_overlap_fitted':
        raise ValueError('promote an immutable initial calibration, never an already promoted fit')
    target_sha, probe_sha = sha(args.target), sha(args.probe)
    rankings, class_sets = {}, {}
    for path in args.ranking:
        value = read(path);cell = value['model'], int(value['batch'])
        if cell in rankings:
            raise ValueError('duplicate ranking cell')
        rankings[cell] = path, value
    for directory in args.classes:
        receipt = read(directory / 'receipt.json');cell = receipt['model'], int(receipt['batch'])
        if cell in class_sets:
            raise ValueError('duplicate class cell')
        class_sets[cell] = directory, receipt
    if set(rankings) != CELLS or set(class_sets) != CELLS:
        raise ValueError('all seven development cells are required before final selection')
    evidence = []
    for cell in sorted(CELLS):
        ranking_path, ranking = rankings[cell]
        directory, receipt = class_sets[cell]
        if ranking['target_sha256'] != target_sha or ranking['probe_sha256'] != probe_sha:
            raise ValueError('every ranking must use the same frozen target and price probe')
        for key, record in receipt['inputs'].items():
            path = Path(record['path'])
            if not path.is_absolute():
                path = ROOT / path
            if sha(path) != record['sha256']:
                raise ValueError('class evidence changed: ' + key)
        if (receipt['inputs']['target']['sha256'] != target_sha
                or receipt['inputs']['probe']['sha256'] != probe_sha):
            raise ValueError('class and request validation use different frozen prices')
        with (directory / 'classes.csv').open() as stream:
            classes = list(csv.DictReader(stream))
        check_metrics(ranking, receipt, classes)
        for row in ranking['candidates']:
            for name, digest in row['correctness']['receipt_sha256'].items():
                if sha(Path(row['correctness']['source_attempt']) / name) != digest:
                    raise ValueError('correctness evidence changed')
        evidence.append(dict(model=cell[0], batch=cell[1], ranking=str(ranking_path.resolve()),
            ranking_sha256=sha(ranking_path), classes=str(directory.resolve()),
            classes_sha256=sha(directory / 'classes.csv'), receipt_sha256=sha(directory / 'receipt.json'),
            kendall_tau=ranking['kendall_tau'],
            max_step_error_pct=max(abs(float(r['step_error_pct'])) for r in ranking['candidates']),
            max_class_error_pct=max(abs(float(r['error_pct'])) for r in classes),
            ownership_verified=all(r['ownership_verified'] for r in ranking['candidates'])))
    output = args.out.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'runs/sm90r2'):
        raise ValueError('choose a new promotion directory under runs/sm90r2')
    calibration['fit_status'] = 'fitted'
    calibration['pending'] = ['E-4 measured selection', 'final matrix and reproducibility']
    calibration['e3_validation'] = dict(source_target_sha256=target_sha,
        price_probe_sha256=probe_sha, cohorts=evidence,
        evidence='verified E-3 metrics; inferred prices; ownership scope retained per cohort')
    write(output / 'target.json', target)
    write(output / 'receipt.json', dict(source_target_sha256=target_sha,
        target_sha256=sha(output / 'target.json'), script_sha256=sha(__file__),
        measured_selection_enabled=True, final_selection_eligible=True,
        final_solver_gate_pass=False, cohorts=evidence))
    print('All seven E-3 cohorts pass; final selection eligible, final solver gate remains open')


if __name__ == '__main__':
    main()
