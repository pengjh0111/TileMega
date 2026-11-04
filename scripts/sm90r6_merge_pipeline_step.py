#!/usr/bin/env python3
"""Validate the eight frozen F predictions and report continuation V1(a)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(c):
    c = dict(c)
    c.pop('schema')
    for k, v in dict(expert_fuse=False, tp=1, argmax='la').items():
        c.setdefault(k, v)
    return c


def merge(source, output):
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
                if approximate else '') + 'frozen F; report only; CP-P correctness gates remain incomplete'
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
    if output.exists():
        with output.open() as stream:
            existing = list(csv.DictReader(stream))
        if any(r['check'] not in ('V1(a)', 'V4-F-subset') for r in existing):
            raise ValueError('retain other acceptance checks instead of overwriting them')
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(dict(cells=len(seen), rows=len(result), output=str(output), evidence='verified calculations')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    merge(a.predictions, a.out)


if __name__ == '__main__':
    main()
