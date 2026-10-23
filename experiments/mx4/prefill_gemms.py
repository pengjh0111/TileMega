#!/usr/bin/env python3
"""MX-4 prefill GEMM geometry (MX4 prompt §5.7): per prefill GEMM class of a
plan, its TileM and the weight scans ceil(M/TileM) with M = B * prompt
tokens, from the plan's search artifact <plan>.classes.tsv. Two plans of one
cell (the old predicted and the new measured selection) give one row each.

Usage: prefill_gemms.py --cell model:B:label:plan.so ... --prompt 64 --csv out.csv
"""
import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cell', nargs='+', required=True)
ap.add_argument('--prompt', type=int, default=64)
ap.add_argument('--csv', type=Path, required=True)
a = ap.parse_args()

NAMES = ('qkv', 'o', 'gate_up', 'down')
rows = []
for spec in a.cell:
    model, batch, label, so = spec.split(':', 3)
    m = int(batch) * a.prompt
    table = list(csv.DictReader(open(so + '.classes.tsv'), delimiter='\t'))
    last = max(int(r['gemm']) for r in table)
    classes = OrderedDict()
    for r in table:
        gemm = int(r['gemm'])
        name = 'head' if gemm == last and last % 4 == 0 else NAMES[gemm % 4]
        key = r['class']
        if key not in classes:
            classes[key] = dict(model=model, batch=int(batch), selection=label, cls=int(key), gemms=set(),
                                tile_m=int(r['tile_m']), tile_n=int(r['tile_n']), tile_k=int(r['tile_k']),
                                stages=int(r['stages']), split_k=int(r['split_k']), m=m,
                                weight_scans=math.ceil(m / int(r['tile_m'])))
        classes[key]['gemms'].add(name)
    for c in classes.values():
        c['gemms'] = '+'.join(sorted(c['gemms']))
        rows.append(c)
        print('%-12s B%-2s %-9s class %d %-14s TileM %3d scans %3d' % (model, batch, label, c['cls'], c['gemms'],
                                                                     c['tile_m'], c['weight_scans']))
with a.csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
