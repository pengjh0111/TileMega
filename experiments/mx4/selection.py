#!/usr/bin/env python3
"""MX4_selection_c550.csv (MX4 prompt §4.4, G-SOLVER): per solved MX-4 decode
cell, the solver's plans #1 and #2 with their predicted and measured TPOT
(the end-to-end serving measure of the compile's --selection measure), the
choice, and prediction / measurement of the chosen plan; plus the final
cell measurement when one is given.

Usage: selection.py --cells runs/mx4/tm --csv out.csv [--final DIR]
  --final: per cell <DIR>/<id>/B<b>/measure/round*/measurements.json.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cells', type=Path, required=True)
ap.add_argument('--csv', type=Path, required=True)
ap.add_argument('--final', type=Path)
a = ap.parse_args()


def classes(pin):
    p = pin['mx_decode']
    return ' '.join('%s=%d/%d/%s' % (c, p[c]['tn'], p[c]['d'], 'SK' if p[c]['part'] == 'kStreamK' else 'TR')
                    for c in ('qkv', 'o', 'gate_up', 'down', 'head'))


rows = []
for manifest in sorted(a.cells.glob('*/B*/decode/plan.so.plan.json')):
    m = json.loads(manifest.read_text())
    s = m['selection']
    if s.get('mode') not in ('measure', 'predicted'):
        continue
    model, batch = manifest.parts[-4], int(manifest.parts[-3][1:])
    p1, p2 = s['plans']
    chosen = s['chosen']
    pred = m['predicted_tpot_ns'] / 1e6
    meas = (p1, p2)[chosen - 1]['measured_tpot_ms']
    row = dict(model=model, batch=batch, tmin=m['attention_tmin'], plan1=classes(p1['pin']),
               plan2_swapped=p2['swapped'], plan1_pred_ms=round(p1['predicted_ns'] / 1e6, 4),
               plan2_pred_ms=round(p2['predicted_ns'] / 1e6, 4), plan1_meas_ms=round(p1['measured_tpot_ms'], 4),
               plan2_meas_ms=round(p2['measured_tpot_ms'], 4), chosen=chosen,
               pred_over_meas=round(pred / meas, 4) if meas else '',
               fit_r2=s['fit']['r2'], fit_max_rel=s['fit']['max_rel'], fit_rows=s['fit']['rows'])
    if a.final:
        runs = sorted((a.final / model / ('B%d' % batch) / 'measure').glob('round*/measurements.json'))
        tpots = [json.loads(r.read_text())['tpot_seconds'] * 1e3 for r in runs]
        if tpots:
            final = statistics.median(tpots)
            row.update(final_tpot_ms=round(final, 4), pred_over_final=round(pred / final, 4))
    rows.append(row)
    print(model, batch, 'chosen', chosen, 'pred/meas', row['pred_over_meas'])
fields = list(dict.fromkeys(k for r in rows for k in r))
with a.csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
