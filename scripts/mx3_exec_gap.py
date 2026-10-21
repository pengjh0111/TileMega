#!/usr/bin/env python3
"""MX3-5 item 3: each cell's predicted decode step against its measured TPOT.

The prediction is the selected plan's `predicted_ns` from the cell's
plan.so.top3.tsv (the cost model's makespan for one decode step at the
search's past); for an MX-3 cell selected by measurement, the selected rank
is the finalist with the lowest median in plan.so.top3_measured.tsv. The
measurement is the cell's end-to-end TPOT median from MX3_serving_c550.csv.
exec_gap = measured / predicted. MX-2 used predicted selection, so its
selected plan is rank 1.
"""
import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--serving', type=Path, default=REPO / 'docs/results/MX3_serving_c550.csv')
ap.add_argument('--mx3', type=Path, default=REPO / 'runs/mx3/final2')
ap.add_argument('--campaign', default='mx3-rerun')
ap.add_argument('--mx2', type=Path, default=REPO / 'runs/mx2/tm')
ap.add_argument('--out', type=Path)
a = ap.parse_args()

rows = list(csv.DictReader(a.serving.open()))


def tpot(campaign, model, batch):
    v = [float(r['tpot_ms']) for r in rows if r['campaign'] == campaign and r['system'] == 'tilemega'
         and r['model'] == model and int(r['batch']) == batch and r['tpot_ms'] not in ('', 'nan')]
    return statistics.median(v) if v else None


def table(path):
    lines = [l.rstrip('\n').split('\t') for l in path.open()]
    return [dict(zip(lines[0], l)) for l in lines[1:]]


def selected(decode, measured):
    top = table(decode / 'plan.so.top3.tsv')
    rank, chosen_ms = 1, None
    if measured and (decode / 'plan.so.top3_measured.tsv').exists():
        finals = [r for r in table(decode / 'plan.so.top3_measured.tsv') if r['mode'] != 'eliminated'
                  and r['mode'] != 'rejected']
        best = min(finals, key=lambda r: float(r['mean_ms']))
        rank, chosen_ms = int(best['rank']), float(best['mean_ms'])
    row = next(r for r in top if int(r['rank']) == rank)
    return rank, float(row['predicted_ns']) / 1e6, row, chosen_ms


out = []
for model in ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_1_8b'):
    for batch in (1, 2, 4, 8, 16):
        rec = dict(model=model, batch=batch)
        for tag, root, campaign, measured in (('mx3', a.mx3, a.campaign, True), ('mx2', a.mx2, 'mx2', False)):
            decode = root / model / ('B%d' % batch) / 'decode'
            if not (decode / 'plan.so.top3.tsv').exists():
                continue
            rank, pred, row, cand_ms = selected(decode, measured)
            meas = tpot(campaign, model, batch)
            rec.update({tag + '_rank': rank, tag + '_predicted_ms': round(pred, 3),
                        tag + '_tpot_ms': round(meas, 3) if meas else '',
                        tag + '_exec_gap': round(meas / pred, 3) if meas else ''})
            if tag == 'mx3':
                rec['mx3_candidate_ms'] = round(cand_ms, 3) if cand_ms else ''
                # MX3-6 item 2: what was selected, per GEMM class, from the key.
                rec['mx3_attention'] = row.get('attention_impl', 'simt')
                rec['mx3_residency'] = row.get('residency', '')
                rec['mx3_key'] = row['key']
                rec['mx3_candidates'] = len(table(decode / 'plan.so.top3.tsv'))
        out.append(rec)
fields = ['model', 'batch', 'mx3_rank', 'mx3_predicted_ms', 'mx3_candidate_ms', 'mx3_tpot_ms', 'mx3_exec_gap',
          'mx2_rank', 'mx2_predicted_ms', 'mx2_tpot_ms', 'mx2_exec_gap',
          'mx3_attention', 'mx3_residency', 'mx3_candidates', 'mx3_key']
for r in out:
    print('%-12s B%-3s mx3 rank %s pred %s cand %s tpot %s gap %s | mx2 rank %s pred %s tpot %s gap %s | %s res %s' % tuple(
        r.get(k, '') for k in fields[:13]))
gaps3 = [r['mx3_exec_gap'] for r in out if r.get('mx3_exec_gap')]
gaps2 = [r['mx2_exec_gap'] for r in out if r.get('mx2_exec_gap')]
if gaps3:
    print('MX-3 exec gap median %.2f (%.2f-%.2f); MX-2 median %s' % (
        statistics.median(gaps3), min(gaps3), max(gaps3),
        '%.2f (%.2f-%.2f)' % (statistics.median(gaps2), min(gaps2), max(gaps2)) if gaps2 else 'n/a'))
if a.out:
    with a.out.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, '') for k in fields})
