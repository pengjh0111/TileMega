#!/usr/bin/env python3
"""MX-4 prefill selection (MX4 prompt §5.7) from the cells' solve artifacts.

Per cell <cells>/<model>/B<b>/prefill: the shortlist (<plan>.resources.tsv),
each candidate's L1 prefill measurements (<plan>.top<rank>.measurement.r*:
pilot r-1, finalists r0 and r1), and the winner (SERVING_TOP3_WINNER in
compile.log). Writes
  --csv    one row per cell: the predicted choice (rank 1) and the measured
           winner, their measured prefill times and the selection's share
           (1 - winner / predicted);
  --gemms  one row per (cell, choice, GEMM class): TileM and weight scans
           ceil(M/TileM), M = B * prompt;
  --pins   the winner of each cell as an --evaluate-configs pin
           (<model>_B<b>_prefill.json), for rebuilding the cell.
Usage: prefill_selection.py --cells runs/mx4/tm --csv A --gemms B --pins configs/pins/mx4
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cells', type=Path, required=True)
ap.add_argument('--csv', type=Path, required=True)
ap.add_argument('--gemms', type=Path, required=True)
ap.add_argument('--pins', type=Path, required=True)
ap.add_argument('--prompt', type=int, default=64)
a = ap.parse_args()

GEOMETRY = re.compile(r'(\d+)x(\d+)x(\d+)s(\d+)k(\d+)')
NAMES = ('qkv', 'o', 'gate_up', 'down')


def parse(key):
    fields = key.split(';')
    geometries = [dict(zip(('tile_m', 'tile_n', 'tile_k', 'stages', 'split_k'), map(int, GEOMETRY.fullmatch(f).groups())))
                  for f in fields if GEOMETRY.fullmatch(f)]
    extra = dict(f.split('=') for f in fields if '=' in f)
    return geometries, int(extra['kappa']), int(extra['residency']), int(extra['Ec']), int(extra['Rq'])


def measured(prefill, rank, shortlist):
    """A candidate byte-identical to another (same generated source) is
    measured once, under the other's rank."""
    source = prefill / ('plan.so.top%d.candidate.so.cu' % rank)
    if not list(prefill.glob('plan.so.top%d.measurement.r*/measurements.json' % rank)) and source.is_file():
        for other in shortlist:
            twin = prefill / ('plan.so.top%d.candidate.so.cu' % other)
            if other != rank and twin.is_file() and twin.read_bytes() == source.read_bytes():
                return measured(prefill, other, ())
    finals, pilot = [], None
    for path in sorted(prefill.glob('plan.so.top%d.measurement.r*/measurements.json' % rank)):
        mean = json.loads(path.read_text())['modes']['L1']['mean_ms']
        if path.parent.name.endswith('r-1'):
            pilot = mean
        else:
            finals.append(mean)
    return (statistics.mean(finals), 'finalist') if finals else (pilot, 'pilot')


def class_names(prefill):
    names = {}
    # The GEMM -> class map; selected_classes.tsv is written only by a finished solve.
    path = prefill / 'plan.so.selected_classes.tsv'
    table = list(csv.DictReader(open(path if path.is_file() else prefill / 'plan.so.classes.tsv'), delimiter='\t'))
    last = max(int(r['gemm']) for r in table)
    for r in table:
        gemm = int(r['gemm'])
        names.setdefault(int(r['class']), set()).add('head' if gemm == last and last % 4 == 0 else NAMES[gemm % 4])
    return {c: '+'.join(sorted(v)) for c, v in names.items()}


rows, gemm_rows = [], []
a.pins.mkdir(parents=True, exist_ok=True)
for prefill in sorted(a.cells.glob('*/B*/prefill')):
    model, batch = prefill.parts[-3], int(prefill.parts[-2][1:])
    log = (prefill / 'compile.log').read_text(errors='replace')
    won = re.search(r'SERVING_TOP3_WINNER rank=(\d+)', log)
    if not won:
        print('skip', model, batch, '(no measured selection)')
        continue
    rc = re.findall(r'^rc=(\d+)', log, re.M)
    shortlist = {int(r['rank']): r['key'] for r in csv.DictReader(open(prefill / 'plan.so.resources.tsv'), delimiter='\t')}
    winner = int(won.group(1))
    p_ms, p_kind = measured(prefill, 1, shortlist)
    w_ms, w_kind = measured(prefill, winner, shortlist)
    names = class_names(prefill)
    m = batch * a.prompt
    for label, rank in (('predicted', 1), ('measured', winner)):
        geometries, kappa, residency, _, _ = parse(shortlist[rank])
        for c, g in enumerate(geometries):
            gemm_rows.append(dict(model=model, batch=batch, selection=label, cls=c, gemms=names.get(c, ''),
                                  residency=residency, tile_m=g['tile_m'], tile_n=g['tile_n'], tile_k=g['tile_k'],
                                  stages=g['stages'], split_k=g['split_k'], m=m,
                                  weight_scans=math.ceil(m / g['tile_m'])))
    geometries, kappa, residency, ec, rq = parse(shortlist[winner])
    pin = dict(cases=[dict(kappa=kappa, residency=residency,
                           geometries=[dict(g, impl='mma16') for g in geometries],
                           attention_kv_block=ec, attention_query_rows=rq)])
    (a.pins / ('%s_B%d_prefill.json' % (model, batch))).write_text(json.dumps(pin, indent=2) + '\n')
    rows.append(dict(model=model, batch=batch, predicted_key=shortlist[1], predicted_ms=round(p_ms, 4),
                     predicted_source=p_kind, winner_rank=winner, winner_key=shortlist[winner],
                     winner_ms=round(w_ms, 4), winner_source=w_kind,
                     selection_share=round(1 - w_ms / p_ms, 4) if winner != 1 else 0.0,
                     compile_rc=rc[-1] if rc else ''))
    print('%-12s B%-2d rank %d  predicted %.3f ms  measured %.3f ms  share %+.1f%%' % (
        model, batch, winner, p_ms, w_ms, 100 * rows[-1]['selection_share']))
for path, data in ((a.csv, rows), (a.gemms, gemm_rows)):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
