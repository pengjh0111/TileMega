#!/usr/bin/env python3
"""Validate complete E5-1/E5-2 evidence and summarize CPU analysis costs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES = ('reuse', 'rederive', 'materialize')
MODELS = ('qwen3_1_7b', 'llama3_1_8b', 'qwen3_30b_a3b', 'olmoe_1b_7b')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def identity(coordinates):
    return hashlib.sha256(json.dumps(coordinates, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def inputs(cell):
    for field in ('receipt', 'parent_receipt', 'seed', 'seed_source', 'export'):
        if sha(cell[field]) != cell[field+'_sha256']:
            raise ValueError('analysis source changed: '+cell[field])


def budget_rows(budget, model, batch):
    cell_id = f'{model}-B{batch}'
    curves = []
    searches = budget / cell_id / 'searches.jsonl'; seen = set()
    for result in read(searches):
        request = result['input']; inputs(request)
        key = result['analysis'], request['budget_ms']
        if key in seen or key[0] not in MODES or key[1] not in (10000,60000,600000):
            raise ValueError('duplicate or unexpected budget search')
        seen.add(key)
        if key[0] != 'reuse' and not result['bitwise_flow_verified_against_reuse']:
            raise ValueError('budget search lacks exact flow verification')
        evaluated = result['evaluated']
        if len(evaluated) != result['evaluations']:
            raise ValueError('incomplete E5-2 candidate set')
        def append(kind, c, rank=''):
            curves.append(dict(model=model,batch=batch,analysis=key[0],budget_ms=key[1],
                kind=kind,discovered_ms=c.get('discovered_ms',''),pool_ns=c['pool_ns'],
                replay_ns=c['replay_ns'],candidate_sha256=identity(c['coordinates']),rank=rank,
                within_budget=c.get('discovered_ms',math.inf)<=key[1],
                evaluated_candidates=len(evaluated),search_seconds=result['seconds'],
                verification_ms=result['verification_ms'],measured_tpot_ms='',
                ))
        best = math.inf; previous = -1
        for c in evaluated:
            if c['discovered_ms'] < previous:
                raise ValueError('nonmonotonic candidate discovery times')
            previous = c['discovered_ms']
            if c['pool_ns'] < best:
                best = c['pool_ns']; append('improvement', c)
        for rank,c in enumerate(result['finalists'],1):append('finalist',c,rank)
    if seen != {(m,b) for m in MODES for b in (10000,60000,600000)}:
        raise ValueError('incomplete E5-2: '+cell_id)
    return curves


def merge(same, budget, output, cells=None):
    times, curves, counts = [], [], []
    for model in MODELS:
        for batch in (1, 16):
            if cells is not None and (model,batch) not in cells:
                continue
            cell_id = f'{model}-B{batch}'
            source = same / cell_id / 'measurements.jsonl'
            rows = read(source)
            if not rows:
                raise ValueError('incomplete E5-1: '+cell_id)
            cell = rows[0]['input']; inputs(cell)
            candidates = json.loads(Path(cell['receipt']).read_text())['evaluated']
            groups = defaultdict(list); seen = set()
            for row in rows:
                key = row['candidate'], row['analysis']
                if key in seen or row['input'] != cell or row['analysis'] not in MODES:
                    raise ValueError('duplicate or mixed analysis row')
                seen.add(key)
                if (row['candidate'] not in range(len(candidates)) or
                        row['coordinates'] != candidates[row['candidate']]['coordinates'] or
                        row['pool_ns'] != candidates[row['candidate']]['pool_ns'] or
                        row['bitwise_flow_equal'] is not True):
                    raise ValueError('E5-1 changed the A0 candidate or prediction')
                if row['analysis'] == 'materialize' and row['enumerated_tasks'] != row['tasks']:
                    raise ValueError('materialization did not cover all task instances')
                fields = ('analysis_ms','release_ms','pricing_ms','evaluation_ms')
                if any(not math.isfinite(row[k]) or row[k] < 0 for k in fields):
                    raise ValueError('invalid analysis duration')
                row['total_ms'] = sum(row[k] for k in fields)
                groups[row['analysis']].append(row)
            if seen != {(i, mode) for i in range(len(candidates)) for mode in MODES}:
                raise ValueError('incomplete three-way E5-1: '+cell_id)
            for mode in MODES:
                for field in ('analysis_ms','release_ms','pricing_ms','evaluation_ms','total_ms'):
                    values = sorted(r[field] for r in groups[mode])
                    median = statistics.median(values)
                    reference = statistics.median(r[field] for r in groups['reuse'])
                    times.append(dict(model=model,batch=batch,analysis=mode,metric=field,
                        candidates=len(values),median_ms=median,p90_ms=values[math.ceil(.9*len(values))-1],
                        relative_to_reuse=median/reference if reference else '',
                        note='verified bitwise FlowProblem equality; empirical nearest-rank p90; includes cold candidate',
                        raw=str(source),raw_sha256=sha(source)))
                counts.append(dict(model=model,batch=batch,analysis=mode,candidates=len(groups[mode]),
                    derivations=sum(r['derivations'] for r in groups[mode]),
                    coupling_hits=sum(r['coupling_hits'] for r in groups[mode]),
                    enumerated_tasks=sum(r['enumerated_tasks'] for r in groups[mode]),
                    enumerated_edges=sum(r['enumerated_edges'] for r in groups[mode]),
                    note='E5-1 exact A0 E replay in discovery order; domain cardinalities reported separately',
                    raw=str(source),raw_sha256=sha(source),
                    budget_raw=str(budget / cell_id / 'searches.jsonl'),
                    budget_raw_sha256=sha(budget / cell_id / 'searches.jsonl')))
            curves.extend(budget_rows(budget, model, batch))
    output.mkdir(parents=True,exist_ok=True)
    write(output/'SM90R6_ab5_time.csv',times)
    write(output/'SM90R6_ab5_budget.csv',curves)
    write(output/'SM90R6_ab5_counts.csv',counts)
    print(json.dumps(dict(time_rows=len(times),curve_rows=len(curves),count_rows=len(counts))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--same-E',type=Path,required=True)
    p.add_argument('--budgets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget-only',action='store_true')
    a=p.parse_args()
    if a.budget_only:
        rows=[row for m in MODELS for b in (1,16) for row in budget_rows(a.budgets,m,b)]
        a.out.mkdir(parents=True,exist_ok=True)
        write(a.out/'SM90R6_ab5_budget.csv',rows)
        print(json.dumps(dict(curve_rows=len(rows),complete_searches=72)))
    else:
        merge(a.same_E,a.budgets,a.out)
