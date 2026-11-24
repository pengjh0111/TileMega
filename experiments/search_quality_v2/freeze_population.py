"""Bind search, executable, score, and device-order receipts before TPOT."""

import json
from pathlib import Path
import random
import time

from calibration_queue import CALIB, CASE, REPRO
from guarded import ROOT, read, sha, write
from materialize_population import MATERIALIZATIONS, PHASE, TARGET, TOOL, TOOLS
from phase0 import PROMPT


def main():
    destination = CASE / 'frozen_search.json'
    if destination.exists():
        raise RuntimeError('retain the existing freeze')
    pop = read(CASE / 'population.json')
    first = read(CASE / 'finalists-materialized.json')
    second = read(CASE / 'population-materialized.json')
    actual = read(MATERIALIZATIONS)
    assert all(q in actual for q in pop['P'])
    assert first['compiler_sha256'] == second['compiler_sha256'] == sha(TOOL)
    assert not any((CASE / 'measurements').glob('**/command.json'))
    model = read(CASE / 'model_freeze.json')
    assert sha(TARGET) == model['target_sha256'] and sha(model['cost_table']) == model['cost_table_sha256']
    reg = read(REPRO / 'reproduction_inputs.json')
    binding = read(CALIB / 'production_source_binding.json')
    assert sha(CALIB / 'production_source_binding.json') == model['production_header_binding_sha256']
    for path, expected in binding['production_headers'].items():
        assert sha(ROOT / path) == expected, path
    cells = {r['id']: r for r in pop['all_original_coordinates']}
    score_path = PHASE / 'population-scores.jsonl'
    scores = {}
    for line in score_path.read_text().splitlines():
        row = json.loads(line)
        q = row['input']['id']
        assert q not in scores and row['coordinates'] == cells[q]['coordinates']
        assert row['constants_from_older_body'] == []
        assert abs(row['pool_ns'] - cells[q].get('pool_ns', row['pool_ns'])) < 0.01, q
        scores[q] = {k: row[k] for k in ('pool_ns', 'operator_sum_ns', 'replay_ns')}
    assert set(scores) == set(pop['P'])
    b_iso = pop['baselines']['B-iso']
    assert abs(scores[b_iso]['operator_sum_ns'] - pop['operator_sum_baseline_ns']) < .01

    representatives = {}
    for q in pop['P']:
        m = actual[q]
        if m['status'] == 'compiled':
            assert m['occupancy_verified'] and sha(m['binary']) == m['binary_sha256']
            representatives.setdefault(m['realization_id'], q)
    order = list(representatives.values())
    rng = random.Random(20261124)
    rng.shuffle(order)
    cards = pop['gpu_cards']
    rounds = [order, list(reversed(order))]
    lanes = [
        {q: cards[(i + round_index) % len(cards)]['uuid'] for i, q in enumerate(order)}
        for round_index in (0, 1)
    ]
    harness = ROOT / 'experiments/search_quality_v2'
    file_manifest = {
        'population': CASE / 'population.json', 'materializations': MATERIALIZATIONS,
        'stage_model_freeze': CASE / 'model_freeze.json', 'scores': score_path,
        'primary_search_receipt': PHASE / 'owner-audit.jsonl',
        'owner_verifier_output': PHASE / 'owner-verification.json',
        'finalists': CASE / 'finalists-materialized.json',
        'target': TARGET, 'cost_table': Path(model['cost_table']),
        'compiler': TOOL, 'solver': TOOLS / 'tilemega-native-search',
        'scorer': TOOLS / 'tilemega-native-evaluate', 'prompt': PROMPT,
        'guard_policy': REPRO / 'guard_policy.json',
        'guarded': harness / 'guarded.py', 'measure': harness / 'measure_population.py',
        'phase0_timing_check': harness / 'phase0.py',
        'runner': ROOT / 'python/tilemega/serving/r16_run.py',
    }
    frozen = {
        'frozen_at': time.time(), 'model_frozen_at': model['frozen_at'],
        'files': {key: {'path': str(path), 'sha256': sha(path)} for key, path in file_manifest.items()},
        'source_hashes': binding['production_headers'],
        'checkpoint': reg['checkpoint'], 'checkpoint_sha256': reg['checkpoint_sha256'],
        'protocol': {'warmup': 1, 'repeats': 3, 'tokens': [1, 1024], 'prompt_length': 64,
                     'prefill': 'prompt/L1', 'mode': 'L1', 'decode_loop': 1,
                     'C1_positions': 256, 'audit_processes': 2, 'confirmation_processes': 5,
                     'order_seed': 20261124, 'device_order': 'fixed shuffled order and reverse, filter by preassigned GPU per round; guard-rejected items rejoin the same GPU queue'},
        'round_order': rounds, 'round_gpu_uuid': lanes, 'gpu_cards': cards,
        'representatives': representatives, 'predicted_winner': first['predicted'],
        'shortlist': pop['Q'], 'finalists': pop['S'], 'population_ids': pop['P'],
        'E_count': pop['E_count'], 'legal_space_seed': pop['seed'],
        'scores': scores, 'baseline_ids': pop['baselines'],
        'algorithm': {'search_profile': 'sq2', 'budget_ms': 600000, 'random_starts': 4, 'epsilon': .015,
                      'shortlist_min': 8, 'shortlist_max': 24, 'finalists_max': 6, 'per_class_cap': 2,
                      'replay': 'Eq. 5 matched exact template owners and order for every shortlisted configuration'},
    }
    write(destination, frozen)
    print('frozen candidates', len(pop['P']), 'unique executable programs', len(representatives),
          'predicted', first['predicted'], flush=True)


if __name__ == '__main__':
    main()
