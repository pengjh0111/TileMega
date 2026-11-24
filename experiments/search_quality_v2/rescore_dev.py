"""Run the preregistered isolated-stage and v1 development gates."""

import csv
import itertools
import json
from pathlib import Path
import statistics
import time

import scipy.stats

from calibration_queue import CALIB, CASE
from guarded import ROOT, read, run, sha, write
from make_cost_table import VARIANTS


PHASE = CASE / 'phase2'
TOOL = CASE.parent / 'build-release/tools'
TARGET = ROOT / 'configs/targets/sm_90_h800_sq2.json'
DEV = ROOT / 'docs/results/sm90_search_quality/qwen3_1_7b_b1'
PHYSICS = {
    'overlap': 'Measured head and steady service envelopes can overlap with the same raw DRAM work.',
    'head-startup': 'Expose the device-measured single-CTA DRAM first-byte delay for the initial tile.',
    'tail-publication': 'Expose the geometry-specific measured epilogue on the final tile, when no next tile can hide it.',
}


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def csv_write(path, values):
    with path.open('w', newline='') as handle:
        w = csv.DictWriter(handle, list(values[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(values)


def matched_pairs(data, coordinates):
    families = ('qkv', 'o', 'gate_up', 'down', 'lm_head')
    by_family = {}
    for a, b in itertools.combinations(data, 2):
        ca, cb = coordinates[a['q']], coordinates[b['q']]
        change = [k for k in families if ca['gemms'][k] != cb['gemms'][k]]
        other = {k for k in ca.keys() | cb.keys() if k != 'gemms' and ca.get(k) != cb.get(k)}
        if len(change) != 1 or (other and not (other == {'swiglu_u'} and change == ['gate_up'])):
            continue
        dy = b['v1_tpot_ms'] - a['v1_tpot_ms']
        dx = (b['pool_ns'] - a['pool_ns']) / 1e6
        by_family.setdefault(change[0], []).append((dx, dy))
    result = {}
    for family in families:
        pairs = by_family.get(family, [])
        result[family] = {
            'pairs': len(pairs),
            'median_signed_delta_residual_ms': statistics.median(x - y for x, y in pairs) if pairs else None,
            'median_absolute_delta_residual_ms': statistics.median(abs(x - y) for x, y in pairs) if pairs else None,
            'same_sign': sum((x > 0) == (y > 0) and x != 0 and y != 0 for x, y in pairs),
        }
    return result


def run_variant(variant, cells_path, truth, coordinates):
    directory = PHASE / variant
    done = directory / 'metrics.json'
    if done.exists():
        return read(done)
    directory.mkdir(parents=True, exist_ok=True)
    table = PHASE / f'{variant}.json'
    metrics = {'variant': variant, 'physical_basis': PHYSICS[variant],
               'cost_table_sha256': sha(table), 'pass': False}
    isolated = directory / 'isolated'
    if (isolated / 'exit.json').exists():
        code = read(isolated / 'exit.json')['returncode']
    else:
        code = run([TOOL / 'tilemega-sq2-check-cost', TARGET, ROOT, table], isolated)
    if code:
        metrics.update(status='isolated_stage_gate_failed', evidence=str(isolated / 'run.log'))
        write(done, metrics)
        return metrics
    stage_rows = rows(isolated / 'run.log')
    table_value = read(table)
    family_of = {x['geometry_id']: x['family'] for x in table_value['gemms']}
    family_of.update({x['geometry_id']: 'attention' for x in table_value['attention']})
    assert {row['geometry_id'] for row in stage_rows} == family_of.keys()
    metrics['isolated_stage_max_abs_error'] = max(abs(float(r['relative_error'])) for r in stage_rows)
    metrics['isolated_stage_geometry_count'] = len(stage_rows)
    by_family = {}
    for family in sorted(set(family_of.values())):
        errors = [float(r['simulated_ns']) - float(r['measured_ns'])
                  for r in stage_rows if family_of[r['geometry_id']] == family]
        by_family[family] = {'median_signed_ns': statistics.median(errors), 'max_abs_ns': max(map(abs, errors))}
    metrics['isolated_residuals_ns'] = by_family

    preds = directory / 'native-evaluations.jsonl'
    if (directory / 'dev/exit.json').exists():
        code = read(directory / 'dev/exit.json')['returncode']
    else:
        code = run([TOOL / 'tilemega-native-evaluate', TARGET, ROOT, cells_path, preds,
                    '--native-cost-table', table], directory / 'dev')
    if code:
        metrics.update(status='dev_evaluation_failed', evidence=str(directory / 'dev/run.log'))
        write(done, metrics)
        return metrics
    data = []
    with preds.open() as handle:
        for line in handle:
            v = json.loads(line)
            q = v['input']['id']
            assert v['coordinates'] == coordinates[q] and v['constants_from_older_body'] == []
            data.append({'variant': variant, 'q': q, 'pool_ns': v['pool_ns'],
                         'replay_ns': v['replay_ns'], 'additive_ns': v['additive_ns'],
                         'v1_tpot_ms': truth[q], 'transferred_residual_ms': v['pool_ns'] / 1e6 - truth[q]})
    assert len(data) == len(truth) == 180 and {row['q'] for row in data} == truth.keys()
    csv_write(directory / 'dev_rescore.csv', data)
    xs = [r['pool_ns'] for r in data]
    ys = [r['v1_tpot_ms'] for r in data]
    ranked = sorted(data, key=lambda r: (r['pool_ns'], r['q']))
    best = min(ys)
    curve = [{'k': k, 'regret': min(r['v1_tpot_ms'] for r in ranked[:k]) / best - 1}
             for k in range(1, 33)]
    rho = float(scipy.stats.spearmanr(xs, ys).statistic)
    tau = float(scipy.stats.kendalltau(xs, ys).statistic)
    by_q = {r['q']: r for r in data}
    measured_sign = truth['q135'] - truth['q079']
    predicted_sign = by_q['q135']['pool_ns'] - by_q['q079']['pool_ns']
    metrics.update(
        status='dev_gate_complete', spearman=rho, kendall=tau, regret_curve=curve,
        top10_best_gap=curve[9]['regret'], top1=ranked[0]['q'],
        matched_family_deltas=matched_pairs(data, coordinates),
        q079_to_q135={'pool_delta_ns': predicted_sign, 'v1_delta_ms': measured_sign,
                      'same_sign': predicted_sign * measured_sign > 0},
        **{'pass': rho >= .7 and curve[9]['regret'] <= .01},
    )
    write(done, metrics)
    return metrics


def main():
    export = read(CASE / 'phase0_reproduction/reproduction_inputs.json')['export']
    frozen = read(DEV / 'frozen_search.json')
    assert sha(export) == read(CASE / 'phase0_reproduction/reproduction_inputs.json')['export_sha256']
    coordinates = {c['original_q_id']: c['coordinates'] for c in frozen['candidates']}
    cells = []
    for q, value in coordinates.items():
        path = PHASE / 'dev-coordinates' / f'{q}.json'
        write(path, value)
        cells.append({'id': q, 'coordinates': str(path), 'export': export, 'batch': 1})
    cells_path = PHASE / 'dev_cells.json'
    write(cells_path, cells)
    plan = PHASE / 'preregistered.json'
    predeclared = {
        'variants_in_order': VARIANTS, 'physical_basis': PHYSICS,
        'max_variants': 3, 'isolated_max_error': .03, 'dev_min_rho': .7, 'dev_top10_max_gap': .01,
        'table_hashes': {v: sha(PHASE / f'{v}.json') for v in VARIANTS},
        'target_sha256': sha(TARGET), 'dev_coordinates_sha256': sha(DEV / 'frozen_search.json'),
        'dev_tpot_sha256': sha(DEV / 'candidates.csv'),
        'compiler_sha256': sha(TOOL / 'tilemega'),
        'evaluator_sha256': sha(TOOL / 'tilemega-native-evaluate'),
        'test_set_tpot_used': False,
    }
    if plan.exists():
        assert read(plan)['settings'] == json.loads(json.dumps(predeclared))
    else:
        write(plan, {'at': time.time(), 'settings': predeclared})
    truth = {r['original_q_id']: float(r['tpot_ms']) for r in rows(DEV / 'candidates.csv')}

    outcomes = []
    selected = None
    for variant in VARIANTS:
        metrics = run_variant(variant, cells_path, truth, coordinates)
        outcomes.append(metrics)
        print(variant, metrics['status'], metrics.get('spearman'), metrics.get('top10_best_gap'), flush=True)
        if metrics['pass']:
            selected = variant
            break
    write(PHASE / 'dev_gate.json', {'selected_variant': selected, 'outcomes': outcomes,
                                   'completed_at': time.time(), 'test_set_tpot_used': False})
    if selected:
        write(CASE / 'model_freeze.json', {
            'frozen_at': time.time(), 'variant': selected, 'cost_table': str(PHASE / f'{selected}.json'),
            'cost_table_sha256': predeclared['table_hashes'][selected],
            'target_sha256': sha(TARGET), 'solver_sha256': sha(TOOL / 'tilemega'),
            'dev_gate_sha256': sha(PHASE / 'dev_gate.json'),
            'production_header_binding_sha256': sha(CALIB / 'production_source_binding.json'),
            'before_test_set_timing': True,
        })
    results = [
        row for item in outcomes for f in [PHASE / item['variant'] / 'dev_rescore.csv']
        if f.is_file() for row in rows(f)
    ]
    if results:
        csv_write(CASE / 'dev_rescore.csv', results)
    return 0 if selected else 2


if __name__ == '__main__':
    raise SystemExit(main())
