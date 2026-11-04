#!/usr/bin/env python3
"""Validate complete guarded H2 trajectories and publish their compact table."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    if read(directory / 'result.json')['returncode'] != 0:
        return None
    if read(directory / 'guard/guard_result.json')['code'] != 0:
        return None
    meta = read(directory / 'metadata.json')
    trace = read(directory / 'routing_trace.json')
    indices = read(directory / 'routing_indices.json')
    summary = read(directory / 'summary.json')
    tokens = read(directory / 'tokens.json')
    layers = {'olmoe_1b_7b': 16, 'qwen3_30b_a3b': 48}[meta['model']]
    if meta['steps'] != 1024 or meta['positions'] != list(range(63, 1087)):
        raise ValueError(f'{directory}: H2 requires all 1024 generated positions')
    if len(trace) != 1024 or len(indices) != 1024 or len(summary) != layers:
        raise ValueError(f'{directory}: incomplete trajectory or layer summary')
    if len(tokens) != meta['batch'] or any(len(row) != 1024 for row in tokens):
        raise ValueError(f'{directory}: incomplete generated tokens')
    for step, (counts, routes) in enumerate(zip(trace, indices)):
        if len(counts) != layers or len(routes) != layers:
            raise ValueError(f'{directory}: layer extent at step {step}')
        for layer, (count, batch) in enumerate(zip(counts, routes)):
            row = summary[layer]
            if len(batch) != meta['batch'] or any(
                    len(r) != row['top_k'] or len(set(r)) != row['top_k'] or
                    any(type(e) is not int or e < 0 or e >= row['experts'] for e in r)
                    for r in batch):
                raise ValueError(f'{directory}: invalid routing at step {step}, layer {layer}')
            if count != len({e for r in batch for e in r}):
                raise ValueError(f'{directory}: distinct count differs from raw indices')
    for layer, row in enumerate(summary):
        values = sorted(step[layer] for step in trace)
        if row['layer'] != layer or row['steps'] != 1024 or row['model'] != meta['model'] or \
                row['batch'] != meta['batch'] or not math.isclose(
                    float(row['distinct_mean']), statistics.fmean(values), abs_tol=5e-5) or \
                row['distinct_p10'] != values[math.ceil(.1 * len(values)) - 1] or \
                row['distinct_p90'] != values[math.ceil(.9 * len(values)) - 1]:
            raise ValueError(f'{directory}: layer summary differs from trajectory')
    return dict(directory=directory, metadata=meta, rows=summary,
                tokens=sha(directory / 'tokens.json'), routes=sha(directory / 'routing_indices.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--complete-matrix', action='store_true')
    args = parser.parse_args()
    cells = {}
    for path in sorted(args.root.rglob('summary.json')):
        directory = path.parent
        if not (directory / 'result.json').exists() or not (directory / 'metadata.json').exists():
            continue
        if read(directory / 'metadata.json')['steps'] != 1024:
            continue
        result = validate(directory)
        if result:
            meta = result['metadata']
            cells.setdefault((meta['model'], meta['batch']), []).append(result)
    if not cells:
        raise ValueError('no complete guarded H2 trajectory')
    expected = {(m, b) for m in ('olmoe_1b_7b', 'qwen3_30b_a3b') for b in (1, 2, 4, 8, 16)}
    if args.complete_matrix and set(cells) != expected:
        raise ValueError(f'missing H2 cells: {sorted(expected - set(cells))}')
    rows = []
    for key, runs in sorted(cells.items()):
        first = runs[0]
        for run in runs[1:]:
            if any(run['metadata'][field] != first['metadata'][field]
                   for field in ('checkpoint', 'decode_sha256', 'bridge_sha256', 'prompt_sha256')):
                raise ValueError(f'{key}: different experiment inputs; choose a homogeneous root')
            if (run['tokens'], run['routes']) != (first['tokens'], first['routes']):
                raise ValueError(f'{key}: token or routing trajectory differs across processes')
        for row in first['rows']:
            rows.append(dict(row, processes=len(runs), evidence='verified',
                gpu=';'.join(sorted({run['metadata']['gpu'] for run in runs})),
                decode_sha256=first['metadata']['decode_sha256'],
                routing_sha256=first['routes'], raw_directory=str(first['directory'])))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.csv.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(args.csv)
    print(json.dumps(dict(cells=len(cells), rows=len(rows), csv=str(args.csv))))


if __name__ == '__main__':
    main()
