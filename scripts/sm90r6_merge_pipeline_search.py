#!/usr/bin/env python3
"""Report V3 inspection membership without inserting the reference into E."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(paths, output):
    rows, seen = [], set()
    for path in paths:
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if len(records) != 1:
            raise ValueError('one search per V3 input is required')
        result = records[0]; cell = result['input']; tp = cell['tp']
        if tp not in (2, 4) or tp in seen or cell['batch'] != 1:
            raise ValueError('expected unique TP2/TP4 B1 searches')
        seen.add(tp)
        for field in ('seed', 'seed_source', 'inspection', 'export'):
            if sha(cell[field]) != cell[field+'_sha256']:
                raise ValueError('search input changed: '+cell[field])
        probe = result['inspection']; coordinate = probe['coordinates']
        original = json.loads(Path(cell['inspection']).read_text())
        if any(coordinate.get(k) != v for k, v in original.items()):
            raise ValueError('inspected reference coordinate changed')
        present = any(c['coordinates'] == coordinate for c in result['evaluated'])
        shortlisted = any(c['coordinates'] == coordinate for c in result['shortlist'])
        rank = 1 + sum(c['pool_ns'] < probe['pool_ns'] for c in result['evaluated'])
        if (probe['present_in_E'] != present or probe['in_shortlist'] != shortlisted or
                probe['comparison_rank'] != rank or probe['rank_is_counterfactual'] == present):
            raise ValueError('inspection membership or comparison rank mismatch')
        if result['evaluations'] != len(result['evaluated']):
            raise ValueError('incomplete candidate set')
        note = ('reference absent from E; rank is hypothetical insertion, not an actual search rank'
                if not present else 'reference present in E; rank uses PoolScore')
        for metric, value in dict(present_in_E=int(present), in_shortlist=int(shortlisted),
                comparison_rank=rank, rank_is_counterfactual=int(not present),
                evaluated_candidates=len(result['evaluated']), search_seconds=result['seconds']).items():
            rows.append(dict(check='V3-TP', model='qwen3_1_7b', batch=1, tp=tp,
                candidate='REF-R3-derived', metric=metric, value=value, reference='', ratio='',
                status='reported; offline search', note=note+'; no compilation or occupancy',
                raw=str(path), raw_sha256=sha(path)))
    if seen != {2, 4}:
        raise ValueError('both TP searches are required')
    if output.exists():
        with output.open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['check'] != 'V3-TP'] + rows
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(cells=len(seen), rows=len(rows))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--searches', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); merge(a.searches, a.out)
