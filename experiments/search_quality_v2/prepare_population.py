"""Predeclare the measurement population from the CPU search evidence."""

from pathlib import Path
import random
import statistics
import time

from calibration_queue import CASE
from guarded import ROOT, read, sha, write
from make_gemm_probes import identity


PHASE = CASE / 'phase3'
V1 = ROOT / 'docs/results/sm90_search_quality/qwen3_1_7b_b1'
RNG = 20261124


def main():
    receipt = PHASE / 'owner-audit.jsonl'
    verification_path = PHASE / 'owner-verification.json'
    iso_path = PHASE / 'operator-sum.jsonl'
    search = read(receipt)
    iso = read(iso_path)
    plans = read(verification_path)
    assert plans['all_shortlist_plans_match']
    assert search['objective'] == 'completion' and iso['objective'] == 'operator-sum'
    assert [x['coordinates'] for x in plans['shortlist']] == [x['coordinates'] for x in search['shortlist']]
    assert all(x['eq5_replay_ns'] == q['replay_ns'] for x, q in zip(plans['shortlist'], search['shortlist'], strict=True))
    assert not (CASE / 'population.json').exists()
    coordinates = {}

    def register(c, name):
        key = identity(c)
        return coordinates.setdefault(key, {'id': name, 'coordinates': c, 'coordinate_sha256': key,
                                            'sets': [], 'baselines': []})

    evaluated = []
    for i, candidate in enumerate(search['evaluated']):
        row = register(candidate['coordinates'], f'e{i:04d}')
        assert row['sets'] == []
        row['sets'].append('E')
        row.update(pool_ns=candidate['pool_ns'], search_origin=candidate['origin'],
                   structure=candidate['structural_class'], signature=candidate['implementation_signature'],
                   evaluation_ms=candidate['evaluation_ms'], discovered_ms=candidate['discovered_ms'])
        evaluated.append(row['coordinate_sha256'])
    for stage, members in [('Q', search['shortlist']), ('S', search['finalists'])]:
        for candidate in members:
            row = coordinates[identity(candidate['coordinates'])]
            row['sets'].append(stage)
            row['replay_ns'] = candidate['replay_ns']
    ranked = sorted(evaluated, key=lambda k: (coordinates[k]['pool_ns'], coordinates[k]['id']))
    bins = []
    if len(ranked) <= 400:
        e_sample = ranked
    else:
        e_sample = []
        rng = random.Random(RNG)
        for i in range(10):
            lo, hi = i * len(ranked) // 10, (i + 1) * len(ranked) // 10
            sample = rng.sample(ranked[lo:hi], 30)
            e_sample.extend(sample)
            bins.append({'decile': i + 1, 'population': hi - lo, 'selected': sample})
    included = set(e_sample)
    included.update(k for k, r in coordinates.items() if 'Q' in r['sets'])
    sampler = CASE / 'qspace.json'
    for i, c in enumerate(read(sampler)['uniform_sample']):
        row = register(c, f'u{i:04d}')
        row['sets'].append('U')
        included.add(row['coordinate_sha256'])
    v1 = {v['original_q_id']: v['coordinates'] for v in read(V1 / 'frozen_search.json')['candidates']}
    caller = search['evaluated'][0]
    assert caller['origin'] == 'seed:0'
    seed = caller['coordinates']  # canonicalized by the native parser
    iso_rank = min(enumerate(iso['evaluated']), key=lambda item: (item[1]['pool_ns'], item[0]))[1]
    baselines = {
        'B-seed': seed, 'B-iso': iso_rank['coordinates'], 'B-pool': coordinates[ranked[0]]['coordinates'],
        'v1-q079': v1['q079'], 'v1-q135': v1['q135'],
    }
    for name, c in baselines.items():
        row = register(c, name)
        row['baselines'].append(name)
        included.add(row['coordinate_sha256'])
    order = [key for key in coordinates if key in included]
    ids = [coordinates[k]['id'] for k in order]
    assert len(set(ids)) == len(ids)
    q = [coordinates[identity(r['coordinates'])]['id'] for r in search['shortlist']]
    s = [coordinates[identity(r['coordinates'])]['id'] for r in search['finalists']]
    assert set(s) <= set(q) <= set(ids)
    inputs = read(CASE / 'phase0_reproduction/reproduction_inputs.json')
    reference = read(CASE / 'phase0_reproduction/summary.json')['rows']
    stage_wall = [statistics.median(
        read(Path(r['processes'][n]['measurement']).parents[3] / 'exit.json')['seconds']
        for r in reference.values()
    ) for n in (0, 1)]
    # Maximum of ten distinct choices for five confirmation processes each.
    estimate = (len(ids) * sum(stage_wall) + 50 * stage_wall[1]) / 3600
    output = {
        'at': time.time(), 'release_status': 'cpu-selected population; no test-set timing',
        'production': {'pooled_wall_seconds': search['pooled_seconds'],
                       'eq5_replay_seconds': plans['materialization_and_eq5_seconds'],
                       'search_receipt_sha256': sha(receipt), 'owner_verification_sha256': sha(verification_path)},
        'source_hashes': {str(f): sha(f) for f in (receipt, verification_path, iso_path, sampler, CASE / 'model_freeze.json')},
        'rule': 'E pooled-score deciles use floor-rank boundaries, sample 30 per decile, union Q/S, U and tagged baselines.',
        'seed': RNG, 'E_count': len(evaluated), 'E_sample_count': len(e_sample), 'deciles': bins,
        'Q': q, 'S': s, 'P': ids, 'all_original_coordinates': list(coordinates.values()),
        'baselines': {name: coordinates[identity(c)]['id'] for name, c in baselines.items()},
        'operator_sum_baseline_ns': iso_rank['pool_ns'],
        'guarded_reproduction_wall_seconds_per_process': stage_wall,
        'planned_tp_gpu_hours_before_dedup': estimate,
        'budget_reassessment': 'uses reproduction process medians and up to 50 confirmation processes; compile and occupancy tracked separately',
        'gpu_cards': inputs['authorized_cards'],
    }
    cells = []
    for k in order:
        row = coordinates[k]
        path = CASE / 'candidates' / row['id'] / 'coordinates.json'
        write(path, row['coordinates'])
        cells.append({'id': row['id'], 'coordinates': str(path), 'export': inputs['export'], 'batch': 1})
    write(CASE / 'population.json', output)
    write(PHASE / 'score_cells.json', cells)
    print('E, Q, S, P', len(evaluated), len(q), len(s), len(ids), 'estimated GPU hours', estimate)
    print('baseline ids', output['baselines'])
    return 0 if estimate <= 24 else 2


if __name__ == '__main__':
    raise SystemExit(main())
