#!/usr/bin/env python3
"""Validate the eight frozen F predictions and report continuation V1(a)."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(c):
    c = dict(c)
    c.pop('schema')
    for k, v in dict(expert_fuse=False, tp=1, argmax='la').items():
        c.setdefault(k, v)
    return c


def merge(source, output, warm=None):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    expected = {(m, b) for m in ('qwen3_1_7b', 'llama3_1_8b', 'qwen3_30b_a3b', 'olmoe_1b_7b')
                for b in (1, 16)}
    seen, result = set(), []
    source_sha = sha(source)
    for row in rows:
        cell = row['input']
        key = cell['model'], cell['batch']
        if key not in expected or key in seen:
            raise ValueError('duplicate or unexpected frozen prediction')
        seen.add(key)
        for path, digest in [('coordinates', 'coordinates_sha256'), ('export', 'export_sha256'),
                             ('frozen_source', 'frozen_source_sha256'),
                             ('frozen_timing_source', 'frozen_timing_source_sha256')]:
            if sha(cell[path]) != cell[digest]:
                raise ValueError(f'changed frozen prediction input: {cell[path]}')
        if canonical(row['coordinates']) != canonical(json.loads(Path(cell['coordinates']).read_text())):
            raise ValueError('evaluation changed the frozen coordinates')
        with Path(cell['frozen_timing_source']).open() as stream:
            timing = [r for r in csv.DictReader(stream) if r['model'] == key[0] and int(r['batch']) == key[1]]
        if len(timing) != 1 or float(timing[0]['tpot_ms']) != cell['frozen_tpot_ms']:
            raise ValueError('prediction is not normalized to its frozen timing row')
        frozen = cell['frozen_tpot_ms']
        if not math.isfinite(frozen) or frozen <= 0:
            raise ValueError('invalid frozen TPOT')
        approximate = row['approximation']
        if row['coordinates']['model'] == 'moe' and not approximate:
            raise ValueError('MoE prediction must identify its expected-task approximation')
        note = ('approximate rounded expected expert task domain; local routing uses older-body RouteLa proxy; '
                if approximate else '') + 'frozen F; report only; correctness gate is independent of diagnostic ratios'
        for metric, field in [('PoolScore_ms', 'pool_ns'), ('replay_ms', 'replay_ns'), ('additive_ms', 'additive_ns')]:
            value = row[field] / 1e6
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid predicted step duration')
            result.append(dict(check='V1(a)', model=key[0], batch=key[1], tp=1, candidate='F',
                metric=metric, value=value, reference=frozen, ratio=value/frozen, status='reported',
                note=('additive diagnostic only; expected experts remain continuous; ' if field == 'additive_ns'
                      else note), raw=str(source), raw_sha256=source_sha))
        for metric in ('import_ms', 'pool_evaluation_ms', 'replay_ms'):
            value = row[metric]
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid evaluation duration')
            result.append(dict(check='V4-F-subset', model=key[0], batch=key[1], tp=1, candidate='F',
                metric=metric, value=value, reference=200 if metric == 'pool_evaluation_ms' else '',
                ratio=value/200 if metric == 'pool_evaluation_ms' else '', status='partial dataset',
                note='eight F cells only; cold imports; excludes the V2 candidate dataset; no speed gate certified',
                raw=str(source), raw_sha256=source_sha))
    if seen != expected:
        raise ValueError('incomplete eight-cell frozen matrix')
    if warm is not None:
        repeats = [json.loads(line) for line in warm.read_text().splitlines() if line.strip()]
        if len(repeats) != 32:
            raise ValueError('warm F timing requires cold + three cached repeats per cell')
        originals = {(r['input']['model'], r['input']['batch']): r for r in rows}
        timings = {}
        for row in repeats:
            cell = row['input']; key = cell['model'], cell['batch']
            if key not in originals or cell['repeat'] not in (0, 1, 2, 3):
                raise ValueError('unexpected warm timing cell')
            group = timings.setdefault(key, {})
            if cell['repeat'] in group:
                raise ValueError('duplicate warm timing repeat')
            group[cell['repeat']] = row
            original = originals[key]
            for field in ('coordinates_sha256', 'export_sha256'):
                if cell[field] != original['input'][field]:
                    raise ValueError('warm timing uses different input')
            for field in ('pool_ns', 'replay_ns', 'additive_ns', 'coordinates'):
                if row[field] != original[field]:
                    raise ValueError('caching changed the prediction')
            if row['cold_import'] != (cell['repeat'] == 0):
                raise ValueError('unexpected import/cache reuse')
        if set(timings) != expected or any(set(g) != {0,1,2,3} for g in timings.values()):
            raise ValueError('incomplete warm timing matrix')
        for key, group in timings.items():
            for i in (1,2,3):
                if any(group[i][field] != group[0][field] for field in ('space_misses', 'release_misses')):
                    raise ValueError('cached timing unexpectedly recomputed witnesses')
            value = statistics.median(group[i]['pool_evaluation_ms'] for i in (1,2,3))
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid warm evaluation duration')
            result.append(dict(check='V4-F-warm-subset', model=key[0], batch=key[1], tp=1,
                candidate='F', metric='pool_evaluation_ms_median3', value=value,
                reference=200, ratio=value/200, status='reported; U2 cache reused',
                note='three cached repeats; predictions exactly unchanged; V2 candidate dataset still pending',
                raw=str(warm), raw_sha256=sha(warm)))
    if output.exists():
        with output.open() as stream:
            existing = list(csv.DictReader(stream))
        if any(r['check'] not in ('V1(a)', 'V4-F-subset', 'V4-F-warm-subset') for r in existing):
            raise ValueError('retain other acceptance checks instead of overwriting them')
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(dict(cells=len(seen), rows=len(result), output=str(output), evidence='verified calculations')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--warm', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    merge(a.predictions, a.out, a.warm)


if __name__ == '__main__':
    main()
