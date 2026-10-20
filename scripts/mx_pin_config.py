#!/usr/bin/env python3
"""Turn a solved cell's selection into an --evaluate-configs pin file.

MX3-0 item 3: rebuilding a campaign's final plans from one HEAD needs the
selection without re-running the search. The key in plan.so.top3.tsv holds
one `TileMxTileNxTileKsStageskSplit` group per GEMM class plus `kappa=`, and
`residency` comes from the same row.
"""
import argparse, json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('cell', type=Path, help='a solved cell directory (holds plan.so.top3.tsv)')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--rank', type=int, default=1, help='1 = the selected plan')
a = ap.parse_args()

rows = [l.rstrip('\n').split('\t') for l in (a.cell / 'plan.so.top3.tsv').open()]
header, body = rows[0], rows[1:]
if a.rank > len(body):
    raise SystemExit('cell has only %d ranked plans' % len(body))
row = dict(zip(header, body[a.rank - 1]))
key = row.get('key') or body[a.rank - 1][1]

geometries = []
for part in key.split(';'):
    m = re.match(r'(\d+)x(\d+)x(\d+)s(\d+)k(\d+)$', part)
    if m:
        tm, tn, tk, s, sk = map(int, m.groups())
        geometries.append(dict(tile_m=tm, tile_n=tn, tile_k=tk, stages=s,
                               split_k=sk, impl='mma16'))
kappa = re.search(r'kappa=(\d+)', key)
residency = row.get('residency') or row.get('res') or '1'
case = dict(kappa=int(kappa.group(1)) if kappa else 1,
            residency=int(residency), geometries=geometries)
a.out.write_text(json.dumps(dict(cases=[case]), indent=2) + '\n')
print('%d geometries, kappa=%d residency=%s -> %s'
      % (len(geometries), case['kappa'], residency, a.out))
