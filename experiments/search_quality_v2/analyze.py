"""Compute SQ2 paper quantities from the completed, immutable audit."""

import collections
import itertools
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import spearmanr

from calibration_queue import CASE
from guarded import read, sha, write
from measure_population import csv_dump
from rescore_dev import rows


def sample_random(values, max_k, samples=20000):
    rng = np.random.default_rng(20261124)
    order = np.array([rng.choice(len(values), max_k, replace=False) for _ in range(samples)])
    minima = np.minimum.accumulate(np.asarray(values)[order], axis=1)
    return [{'k': k, 'expected_ms': float(np.mean(minima[:, k - 1])),
             'p05_ms': float(np.percentile(minima[:, k - 1], 5)),
             'p95_ms': float(np.percentile(minima[:, k - 1], 95))}
            for k in range(1, max_k + 1)]


def rank_head(data, field):
    scored = [r for r in data if r[field] not in (None, '')]
    return sorted(scored, key=lambda r: (float(r[field]), r['id']))


def conversion(rows_with_aliases):
    alias_groups = collections.defaultdict(list)
    for r in rows_with_aliases:
        if r['tpot_ms']:
            alias_groups[r['realization_id']].append(r)
    out = []
    for entries in alias_groups.values():
        representative = next(r for r in entries if r['id'] == r['representative'])
        combined = dict(representative)
        for label in ('E', 'Q', 'S', 'U'):
            combined[label] = any(r[label] == 'True' for r in entries)
        q_scores = [float(r['replay_ns']) for r in entries if r['Q'] == 'True']
        combined['replay_ns'] = min(q_scores) if q_scores else None
        for key in ('pool_ns', 'operator_sum_ns', 'tpot_ms', 'range_ms'):
            combined[key] = float(representative[key])
        out.append(combined)
    return out


def matched_attention_pair(pop, by_id):
    measured = {
        (int(r['chunks']), int(r['block_keys']), int(r['slots']), int(r['new_row_work']), r['merge']):
            (float(r['combined_body_ns']), float(r['combined_body_ns_process_range']))
        for r in rows(CASE / 'calib_attention.csv')
        if r['model'] == 'qwen3_1_7b' and r['past'] == '575'
    }
    data = [row for row in pop['all_original_coordinates']
            if row['id'] in by_id and by_id[row['id']]['tpot_ms']]
    pair_first, checks = None, 0
    for a, b in itertools.combinations(sorted(data, key=lambda r: r['id']), 2):
        ca, cb = a['coordinates'], b['coordinates']
        if ca.get('argmax', 'la') != cb.get('argmax', 'la'):
            continue
        if any(ca.get(k) != cb.get(k) for k in ca.keys() | cb.keys()
               if k not in ('attention', 'schema', 'tp', 'argmax')):
            continue
        x, y = ca['attention'], cb['attention']
        if x == y:
            continue
        items = []
        for item, coord in [(a, x), (b, y)]:
            key = (coord['chunks'], coord['block_keys'], coord['slots'], coord['new_row_work'], coord['merge'])
            observed = by_id[item['id']]
            stage_ms, stage_range = (x / 1e6 for x in measured[key])
            items.append({'id': item['id'], 'chunks': coord['chunks'], 'merge': coord['merge'],
                          'isolated_ms': stage_ms, 'isolated_range_ms': stage_range,
                          'program_ms': float(observed['tpot_ms']),
                          'program_range_ms': float(observed['range_ms']),
                          'processes_ms': [float(observed[k]) for k in ('process0_tpot_ms', 'process1_tpot_ms')]})
        first, second = items
        xdiff = first['isolated_ms'] - second['isolated_ms']
        ydiff = first['program_ms'] - second['program_ms']
        resolved = (max(first['processes_ms']) < min(second['processes_ms']) or
                    max(second['processes_ms']) < min(first['processes_ms']))
        checks += 1
        local_resolved = abs(xdiff) > max(first['isolated_range_ms'], second['isolated_range_ms'])
        pair = {'pair': items, 'reversal': xdiff * ydiff < 0 and resolved and local_resolved}
        if pair_first is None:
            pair_first = pair
        if pair['reversal']:
            return dict(pair, scanned=checks, rule='lexicographic first matched attention reversal with disjoint process ranges')
    return {'scanned': checks, **(pair_first or {'pair': [], 'reversal': False}),
            'rule': 'no matched attention reversal outside the two-process ranges'}


def costs(pop, actual, attempts):
    phases = {phase: read(CASE / f'{phase}-materialized.json') for phase in ('finalists', 'population')}
    all_build_attempts = {Path(r['build_dir']) / 'compile/exit.json' for r in actual.values() if r.get('build_dir')}
    finalists_build_attempts = {Path(actual[q]['build_dir']) / 'compile/exit.json'
                               for q in pop['S'] if actual[q].get('build_dir')}
    build_times = [read(p)['seconds'] for p in sorted(all_build_attempts)]
    per_phase = collections.defaultdict(float)
    for attempt in attempts:
        if attempt['wall_seconds']:
            per_phase[attempt['phase']] += float(attempt['wall_seconds'])
    elapsed_by_rank = {}
    for phase in ('round0', 'round1'):
        group = [float(r['wall_seconds']) for r in attempts if r['phase'] == phase and r['status'] == 'valid']
        elapsed_by_rank[phase] = statistics.median(group)
    compiled_wall = sum(phases['finalists'][k] for k in
                        ('compile_and_generation_wall_seconds', 'occupancy_and_feedback_wall_seconds'))
    search = pop['production']['pooled_wall_seconds'] + pop['production']['eq5_replay_seconds']
    iso_search = read(CASE / 'phase3/operator-sum.jsonl')['seconds']
    stats = {
        'search_pooled_seconds': pop['production']['pooled_wall_seconds'],
        'shortlist_eq5_seconds': pop['production']['eq5_replay_seconds'],
        'operator_sum_search_seconds': iso_search,
        'compile_finalists_wall_seconds': compiled_wall,
        'compile_finalists_command_work_seconds': sum(read(p)['seconds'] for p in finalists_build_attempts),
        'compile_finalists_count': len(finalists_build_attempts),
        'production_predicted_search_and_compile_seconds': search + compiled_wall,
        'phase_artifacts': phases,
        'audit_compile_commands_work_seconds': sum(build_times),
        'audit_guarded_attempt_work_seconds_by_phase': dict(per_phase),
        'median_compile_candidate_seconds': statistics.median(build_times),
        'median_guarded_process_seconds': elapsed_by_rank,
    }
    qspace = read(CASE / 'qspace.json')
    cpu_each = stats['median_compile_candidate_seconds']
    gpu_each = sum(elapsed_by_rank.values())
    stats['extrapolated_E_compile_and_measure_s'] = pop['E_count'] * (cpu_each + gpu_each)
    stats['extrapolated_legal_space_compile_s'] = qspace['legal_count_estimate'] * cpu_each
    stats['extrapolation_scope'] = 'Candidate count times median command work. Sequential equivalent, with no parallel speedup assumed.'
    return stats


def main():
    assert (CASE / 'measurement_complete.json').is_file()
    frozen = read(CASE / 'frozen_search.json')
    plan = read(CASE / 'confirmation_plan.json')
    pop = read(CASE / 'population.json')
    manifest = read(CASE / 'materializations.json')
    candidate_table = rows(CASE / 'candidates.csv')
    by_id = {r['id']: r for r in candidate_table}
    p = conversion(candidate_table)
    assert len(p) >= 32 and any(r['U'] for r in p)
    oracle = min(p, key=lambda r: (r['tpot_ms'], r['id']))
    top = sorted(p, key=lambda r: (r['tpot_ms'], r['id']))[:math.ceil(.05 * len(p))]
    ref = oracle['tpot_ms']
    assert oracle['id'] == plan['observed_top3'][0]
    rho = float(spearmanr([r['pool_ns'] for r in p], [r['tpot_ms'] for r in p]).statistic)
    recall = sum(r['Q'] for r in top) / len(top)
    c = costs(pop, manifest, rows(CASE / 'timings.csv'))
    k = c['compile_finalists_count']
    assert 1 <= k <= 32
    random = sample_random([r['tpot_ms'] for r in p], 32)

    confirmations = collections.defaultdict(list)
    for attempt in rows(CASE / 'confirmation.csv'):
        if attempt['status'] == 'valid':
            confirmations[attempt['qid']].append(float(attempt['tpot_ms']))
    confirm_ref = confirmations.get(oracle['id'], [])
    key_methods = {
        'Oracle': oracle['id'], 'TileMega (predicted)': plan['predicted'],
        'TileMega (measured)': plan['measured'], **frozen['baseline_ids'],
    }
    selection = []
    for method, qid in key_methods.items():
        row = by_id[qid] if qid else {}
        t = float(row['tpot_ms']) if row.get('tpot_ms') else None
        values = confirmations.get(row.get('representative'), [])
        confirm_tp = statistics.median(values) if len(values) == 5 else None
        compile_count = (len(p) if method == 'Oracle' else k if method.startswith('TileMega') else 1)
        rail = c['production_predicted_search_and_compile_seconds'] if method.startswith('TileMega') else None
        if method.startswith('B-') and row.get('compile_status') == 'compiled':
            compile_wall = read(Path(manifest[qid]['build_dir']) / 'compile/exit.json')['seconds']
            rank_wall = c['search_pooled_seconds'] if method == 'B-pool' else (
                c['operator_sum_search_seconds'] if method == 'B-iso' else 0)
            rail = compile_wall + rank_wall
        selection.append({
            'method': method, 'id': qid, 'tpot_ms': t, 'gap_pct': 100 * (t / ref - 1) if t else None,
            'confirmation_ms': confirm_tp, 'confirmed_processes': len(values),
            'confirmation_gap_pct': 100 * (confirm_tp / statistics.median(confirm_ref) - 1)
            if confirm_tp and len(confirm_ref) == 5 else None, 'compiled_candidates': compile_count,
            'search_plus_compile_seconds': rail,
        })
    random_tp = random[k - 1]['expected_ms']
    selection.append({'method': 'B-rand-k (expectation)', 'id': None, 'tpot_ms': random_tp,
                      'gap_pct': 100 * (random_tp / ref - 1), 'confirmation_ms': None,
                      'confirmed_processes': None, 'compiled_candidates': k, 'search_plus_compile_seconds': None})
    csv_dump(CASE / 'selection_summary.csv', selection)
    csv_dump(CASE / 'analysis_population.csv', p)
    write(CASE / 'costs.json', c)
    curve = []
    for label, field, dataset in [
        ('TileMega replay order', 'replay_ns', [r for r in p if r['Q']]),
        ('Pooled order', 'pool_ns', p), ('Operator-sum order', 'operator_sum_ns', p),
    ]:
        rank = rank_head(dataset, field)
        curve.extend({'method': label, 'k': i, 'gap_pct': 100 * (min(r['tpot_ms'] for r in rank[:i]) / ref - 1)}
                     for i in range(1, min(32, len(rank)) + 1))
    curve += [{'method': 'Random', 'k': row['k'], 'gap_pct': 100 * (row['expected_ms'] / ref - 1),
              'p05': 100 * (row['p05_ms'] / ref - 1), 'p95': 100 * (row['p95_ms'] / ref - 1)} for row in random]
    csv_dump(CASE / 'regret_curves.csv', curve)
    pair = matched_attention_pair(pop, by_id)
    pred = selection[1]
    iso = next(r for r in selection if r['method'] == 'B-iso')
    iso_raw, pred_raw = by_id[iso['id']], by_id[pred['id']]
    isolated_beaten = bool(iso['tpot_ms'] and pred['tpot_ms'] and
        min(float(iso_raw[k]) for k in ('process0_tpot_ms', 'process1_tpot_ms')) >
        max(float(pred_raw[k]) for k in ('process0_tpot_ms', 'process1_tpot_ms')))
    outcomes = {
        'C1_selection': pred['gap_pct'] is not None and pred['gap_pct'] <= 2,
        'C2_ranking': rho >= .7 and recall >= .5,
        'C3_cost': c['production_predicted_search_and_compile_seconds'] <= 900,
        'C4_local_vs_global': isolated_beaten and pair['reversal'],
    }
    qspace = read(CASE / 'qspace.json')
    numbers = {
        'legal_space_estimate': qspace['legal_count_estimate'], 'legal_space_95': qspace['legal_count_wilson_95'],
        'funnel_counts': [qspace['legal_count_estimate'], pop['E_count'], len(pop['Q']), len(pop['S']), 1],
        'funnel_seconds': [qspace['wall_seconds'], c['search_pooled_seconds'], c['shortlist_eq5_seconds'],
                           c['compile_finalists_wall_seconds'], None],
        'population_raw': len(candidate_table), 'population_valid_unique': len(p),
        'population_U_unique': sum(r['U'] for r in p), 'spearman': rho, 'recall_at_Q': recall,
        'top_five_percent_size': len(top), 'oracle_ms': ref, 'selection': selection, 'costs': c,
        'local_matched_attention': pair, 'criteria': outcomes,
        'sources': {
            'legal_space': {'file': 'qspace.json', 'calculation': 'proposal volume times acceptance rate; Wilson interval'},
            'funnel': {'file': 'population.json and costs.json', 'calculation': 'original coordinate counts, guarded elapsed phase costs'},
            'population': {'file': 'candidates.csv', 'calculation': 'exact program identities with two valid process observations'},
            'ranking': {'file': 'analysis_population.csv', 'calculation': 'Spearman over one row per valid program; ceil(5% of P) sorted by measured median and id'},
            'selection': {'file': 'candidates.csv, confirmation.csv', 'calculation': 'two-process audit medians; 5-process confirmation medians kept independent'},
            'random': {'file': 'regret_curves.csv', 'calculation': '20,000 subsamples without replacement, seed 20261124, expected minimum and 5th/95th percentiles'},
            'local': {'file': 'calib_attention.csv, candidates.csv', 'calculation': pair['rule']},
        },
    }
    write(CASE / 'numbers.json', numbers)
    print('criteria', outcomes, 'valid programs', len(p), 'oracle', oracle['id'], ref, flush=True)


if __name__ == '__main__':
    main()
