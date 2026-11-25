#!/usr/bin/env python3
"""Summarize A800 service coverage and finalist rank against same-GPU timing."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from a100_baselines import inspect_inputs
from a100_results import write_csv
from a800_baselines import INVENTORY
from a800_r11_collect_round import collect
from a800_r11_gate import read


def audit(path, coordinates=None):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    checked = [row for row in rows if 'modeled_services' in row and 'measured_services' in row]
    # More than one search origin can reach the same coordinate. Count each
    # candidate once while preserving the duplicated origins in raw evidence.
    unique = {json.dumps(row['coordinates'], sort_keys=True): row for row in checked}
    counts = Counter(row['status'] for row in unique.values())
    primary = (unique.get(json.dumps(coordinates, sort_keys=True)) if coordinates is not None else
               next((row for row in checked if row['origin'] == 'finalist'), None))
    return dict(modeled_services_count=len(primary['modeled_services']) if primary is not None else '',
                measured_services_count=len(primary['measured_services']) if primary is not None else '',
                modeled_all_candidates=sum(len(row['modeled_services']) for row in unique.values()),
                measured_all_candidates=sum(len(row['measured_services']) for row in unique.values()),
                discarded_unpriced=counts['legal_missing_calibration'],
                distinct_candidates=len(unique))


def axes(a, b):
    fields = set(a) | set(b)
    for field in sorted(fields-{'gemms', 'schema', 'tp'}):
        if a.get(field) != b.get(field):
            yield field
    for name in sorted(a['gemms'].keys() | b['gemms'].keys()):
        old, new = a['gemms'][name], b['gemms'][name]
        for field in sorted(old.keys() | new.keys()):
            if old.get(field) != new.get(field):
                yield f'{name}.{field}'


def rank_result(prepared, runs, job, inputs):
    binding = read(prepared)
    candidates = binding['candidates']
    control = read(binding['control'])['coordinates']
    measurements = {fact['decode_sha256']: fact for fact, _, _ in collect(runs, job, inputs)}
    output = []
    for candidate in candidates:
        rank = candidate['rank']
        fact = measurements[candidate['binary_sha256']]
        prediction = candidate['predicted_ns']
        output.append(dict(model=fact['model'], batch=fact['batch'], job=fact['job'], rank=rank,
            structure=candidate['structure'], eligible=fact['pass'], status=fact['status'],
            predicted_tpot_ms=prediction/1e6 if prediction is not None else '',
            measured_tpot_ms=fact['tpot_ms'], control_tpot_ms=fact['control_tpot_ms'],
            new_candidate=candidate['coordinates'] != control, rank1_hit='', deviations='',
            decode_sha256=fact['decode_sha256'], evidence=fact['evidence']))
    viable = [row for row in output if row['eligible']]
    if viable:
        best = min(row['measured_tpot_ms'] for row in viable)
        first = output[0]
        first['rank1_hit'] = bool(first['eligible'] and first['measured_tpot_ms'] <= best*1.005)
    differences = defaultdict(list)
    for i, (base, left) in enumerate(zip(output, candidates)):
        if not base['eligible'] or not base['predicted_tpot_ms']:
            continue
        for trial, right in zip(output[i+1:], candidates[i+1:]):
            if not trial['eligible'] or not trial['predicted_tpot_ms']:
                continue
            price = trial['predicted_tpot_ms']/base['predicted_tpot_ms']
            observed = trial['measured_tpot_ms']/base['measured_tpot_ms']
            delta = abs(math.log(price/observed))
            for axis in axes(left['coordinates'], right['coordinates']):
                differences[axis].append(delta)
    identity = dict(model=binding['model'], batch=binding['batch'], job=job, finalists=len(output),
        distinct_structural_classes=len({row['structure'] for row in output}),
        new_candidates_measured=sum(row['new_candidate'] for row in viable),
        evaluated=binding['evaluated'])
    return output, identity, differences


def recovery_result(spec, runs, inputs):
    binding = read(spec['finalists'])
    names = {row['binary_sha256']: row['label'] for row in binding['candidates']}
    for fact, observation, _ in collect(runs, spec['job'], inputs):
        yield dict(fact, label=names[fact['decode_sha256']],
                   measured=observation is not None)


def summarize(cells, out, inputs):
    coverage, counts, ranks, bias = [], [], [], defaultdict(list)
    recoveries = []
    for cell in cells:
        old, new = (audit(Path(cell[f'{round}_coverage']), cell.get(f'{round}_coordinates'))
                    for round in ('r10', 'r11'))
        coverage.append(dict(model=cell['model'], batch=cell['batch'], **new,
                             **{f'r10_{key}': value for key,value in old.items()}))
        if cell.get('job'):
            runs = Path(cell['runs'])
            rows, count, deviations = rank_result(Path(cell['finalists']), runs, cell['job'], inputs)
            specs = cell.get('recoveries', ())
            if cell.get('recovery'):
                specs = (*specs, cell['recovery'])
            recovered = [row for spec in specs for row in recovery_result(spec, runs, inputs)]
            recoveries.extend(recovered)
            measured = {row['decode_sha256'] for row in rows if row['eligible'] and row['new_candidate']}
            supplementary = {row['decode_sha256'] for row in recovered if row['measured']} - measured
            count.update(new_original_candidates_measured=len(measured),
                         recovered_candidates_measured=len(supplementary),
                         new_candidates_measured=len(measured | supplementary))
            ranks.extend(rows)
            counts.append(count)
            for key, values in deviations.items():
                bias[key].extend(values)
    out.mkdir(parents=True, exist_ok=True)
    if coverage:
        write_csv(out/'A800_R11_coverage.csv', coverage)
    if ranks:
        worst = sorted(bias, key=lambda axis: statistics.median(bias[axis]), reverse=True)[:3]
        for row in ranks:
            if row['rank'] == 1:
                row['deviations'] = ' | '.join(f'{axis}: {statistics.median(bias[axis]):.4f} log ratio ({len(bias[axis])} pairs)'
                                               for axis in worst)
        write_csv(out/'A800_R11_rank.csv', ranks)
        write_csv(out/'A800_R11_candidates.csv', counts)
    if recoveries:
        write_csv(out/'A800_R11_router_recovery.csv', recoveries)
    return dict(coverage_cells=len(coverage), no_modeled_services=sum(row['modeled_services_count']==0 for row in coverage),
                rank_cells=len(counts), rank1_hits=sum(row['rank1_hit'] is True for row in ranks),
                four_diverse_finalists=sum(row['finalists']==4 and row['distinct_structural_classes']>=2
                                           and row['new_candidates_measured']>=2 for row in counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    cells = read(args.ledger)['cells']
    inputs = inspect_inputs(INVENTORY) if any(cell.get('job') for cell in cells) else None
    summary = summarize(cells, args.out, inputs)
    print(f'{summary["no_modeled_services"]}/{summary["coverage_cells"]} cells with no modeled services')


if __name__ == '__main__':
    main()
