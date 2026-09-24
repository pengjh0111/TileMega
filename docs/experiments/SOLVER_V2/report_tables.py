#!/usr/bin/env python3
"""Rebuild R9 report tables from process logs and compiler dumps.

No measurement.json, findings, summary, or gate verdict is consumed. Missing
arms stay pending; incomplete ten-process samples never become measurements.
--out defaults to report_tables/ beside this script. Tables are report inputs,
not independent evidence; every row names its original evidence directory.
"""
import argparse
import collections
import csv
import pathlib
import re
import statistics

from measure import parse
from plan_statistics import attention_path, interleaving, rows

HERE = pathlib.Path(__file__).resolve().parent
ARMS = ('legacy', 'skeleton-k4', 'skeleton-k8', 'skeleton-k16', 'skeleton-kW')


def table(path):
    return list(rows(path)) if path.exists() else []


def samples(directory):
    logs = sorted(directory.glob('process_*.log'))
    values = []
    passed = 0
    for log in logs:
        valid, times = parse(log.read_text())
        passed += valid and times is not None
        if times is not None:
            values.append(times)
    complete = len(logs) == passed == len(values) == 10
    median = tuple(map(statistics.median, zip(*values))) if complete else None
    return len(logs), passed, median


def write(path, fields, records):
    with path.open('w') as output:
        writer = csv.DictWriter(output, fields, delimiter='\t', extrasaction='raise')
        writer.writeheader()
        writer.writerows(records)


def collect(root, output):
    output.mkdir(parents=True, exist_ok=True)
    performance, processes, phases, resources, classes, placements = [], [], [], [], [], []
    oracles, paths, searches, statuses = [], [], [], []
    observed = {}
    for model in ('llama', 'qwen3'):
        for seq in (1, 4, 16, 64):
            for arm in ARMS:
                directory = root / ('legacy_r8_domain' if arm == 'legacy' else 'matrix') / f'{model}_s{seq}'
                if arm != 'legacy':
                    directory /= arm
                identity = dict(model=model, seq=seq, arm=arm, evidence=str(directory.relative_to(root)))
                timing = table(directory / 'selected.cu.timing.tsv')
                for row in timing:
                    phases.append(dict(identity, phase=row['phase'], count=row['count'], total_ms=row['total_ms']))
                candidates = ([directory / 'selected.cu.measurement'] if arm == 'legacy'
                              else sorted(directory.glob('selected.cu.top*.cu.measurement')))
                valid = []
                for candidate in candidates:
                    count, passed, times = samples(candidate)
                    if count:
                        processes.append(dict(identity, candidate=candidate.name, processes=count, passed=passed))
                    if times:
                        valid.append((times[2], candidate, times))
                if not valid or len(candidates) != (1 if arm == 'legacy' else 3):
                    statuses.append(dict(identity, state='pending_or_failed', detail='no complete internally equal shortlist'))
                    continue
                _, winner, times = min(valid, key=lambda x: x[0])
                performance.append(dict(identity, candidate=winner.name, l05_ms=times[0], l1_ms=times[1],
                                        l2_ms=times[2], l2_l1=times[2]/times[1], l2_l05=times[2]/times[0]))
                observed[model, seq, arm] = times
                statuses.append(dict(identity, state='measured', detail=winner.name))
                if arm == 'legacy':
                    continue
                shortlist = table(directory / 'selected.cu.top3.tsv')
                key = next(r['key'] for r in shortlist if pathlib.Path(r['source']).name == winner.name.removesuffix('.measurement'))
                metrics = []
                for file in sorted(directory.glob('selected.cu.final*.metrics.tsv')):
                    for row in table(file):
                        metrics.append((file, row))
                level_order = sorted(metrics, key=lambda x: (float(x[1]['level2_ns']), x[0].name))
                sim_order = sorted(metrics, key=lambda x: (float(x[1]['simulated_ns']), x[0].name))
                level_ranks = {r['key']: i+1 for i, (_, r) in enumerate(level_order)}
                sim_ranks = {r['key']: i+1 for i, (_, r) in enumerate(sim_order)}
                for row in table(directory / 'selected.cu.resources.tsv'):
                    resources.append(dict(identity, **row, level2_rank=level_ranks[row['key']], simulated_rank=sim_ranks[row['key']]))
                metric_file, metric = next((p, r) for p, r in metrics if r['key'] == key)
                prefix = str(metric_file).removesuffix('.metrics.tsv')
                selected_classes = table(pathlib.Path(prefix+'.classes.tsv'))
                for row in selected_classes:
                    classes.append(dict(identity, **row))
                queue = interleaving(prefix+'.tasks.tsv')
                legacy_queue = root/'legacy_r8_domain'/f'{model}_s{seq}'/'eft_queue.tsv'
                control = interleaving(legacy_queue) if legacy_queue.exists() else None
                placed = int(metric['placed'])
                source = pathlib.Path(str(winner).removesuffix('.measurement')).read_text()
                variants = int(re.search(r'^#define TILEMEGA_GEMM_VARIANT_COUNT (\d+)$', source, re.M)[1])
                placements.append(dict(identity, residency=metric['residency'], grid=metric['grid'], variants=variants,
                    affinity_share=int(metric['affinity'])/placed, home_share=int(metric['home'])/placed,
                    spread_other_share=int(metric['spread_other'])/placed,
                    average_candidates=int(metric['candidate_sum'])/placed, interleaving=queue['interleaving'],
                    legacy_eft_interleaving=control['interleaving'] if control else 'pending'))
                path = attention_path(prefix+'.tasks.tsv', prefix+'.edges.tsv')
                paths.append(dict(identity, **path))
                census = table(pathlib.Path(prefix+'.oracles.tsv'))
                for field in ('structure', 'predecessor', 'successor'):
                    for category, count in sorted(collections.Counter(row[field] for row in census).items()):
                        oracles.append(dict(identity, field=field, category=category, count=count))
                evaluated = coordinates = improvements = aligned = 0
                search_path = directory/'selected.cu.search.tsv'
                for line in search_path.read_text().splitlines():
                    parts = line.split('\t')
                    if parts[0] == 'EVALUATE': evaluated += 1
                    if parts[0] == 'COORDINATE':
                        coordinates += 1
                        improvements += int(parts[4])
                        aligned += int(parts[5])
                phase_map = {r['phase']: r for r in timing}
                hits = int(phase_map.get('cache_hit', {}).get('count', 0))
                misses = int(phase_map.get('cache_miss', {}).get('count', 0))
                searches.append(dict(identity, evaluations=evaluated, coordinates=coordinates, improvements=improvements,
                    aligned_improvements=aligned, cache_hit=hits, cache_miss=misses,
                    cache_hit_share=hits/(hits+misses) if hits+misses else 0,
                    rounds=phase_map.get('search_rounds', {}).get('count', 'pending'),
                    jobs=phase_map.get('search_jobs', {}).get('count', 1)))
    comparisons = []
    for (model, seq, arm), times in observed.items():
        if arm == 'legacy': continue
        control = observed.get((model, seq, 'legacy'))
        wide = observed.get((model, seq, 'skeleton-kW'))
        comparisons.append(dict(model=model, seq=seq, arm=arm,
            l2_legacy=times[2]/control[2] if control else 'pending',
            l2_kW=times[2]/wide[2] if wide else 'pending'))
    identity_fields = ['model', 'seq', 'arm', 'evidence']
    schemas = [
        ('status', statuses, ['state', 'detail']),
        ('performance', performance, ['candidate', 'l05_ms', 'l1_ms', 'l2_ms', 'l2_l1', 'l2_l05']),
        ('processes', processes, ['candidate', 'processes', 'passed']),
        ('phases', phases, ['phase', 'count', 'total_ms']),
        ('resources', resources, ['rank', 'key', 'estimated', 'actual', 're_solved', 'residency', 'level2_ns', 'simulated_ns', 'level2_rank', 'simulated_rank']),
        ('classes', classes, ['class', 'gemm', 'op', 'tile_m', 'tile_n', 'tile_k', 'stages', 'split_k', 'seed_m', 'seed_n', 'seed_k', 'seed_stages', 'seed_split']),
        ('placements', placements, ['residency', 'grid', 'variants', 'affinity_share', 'home_share', 'spread_other_share', 'average_candidates', 'interleaving', 'legacy_eft_interleaving']),
        ('paths', paths, ['cp_ns', 'path_nodes', 'attention_ns', 'attention_share']),
        ('oracles', oracles, ['field', 'category', 'count']),
        ('searches', searches, ['evaluations', 'coordinates', 'improvements', 'aligned_improvements', 'cache_hit', 'cache_miss', 'cache_hit_share', 'rounds', 'jobs']),
    ]
    for name, records, fields in schemas:
        write(output/(name+'.tsv'), identity_fields+fields, records)
    write(output/'comparisons.tsv', ['model', 'seq', 'arm', 'l2_legacy', 'l2_kW'], comparisons)
    print(f'REPORT measured={len(performance)}/40 pending_or_failed={40-len(performance)} output={output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=pathlib.Path, default=HERE)
    parser.add_argument('--out', type=pathlib.Path, default=HERE/'report_tables')
    args = parser.parse_args()
    collect(args.evidence, args.out)
