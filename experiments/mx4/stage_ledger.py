#!/usr/bin/env python3
"""MX-4 traced stage ledger (MX4 prompt §6 D7, §9 MX4_stage_ledger_c550.csv).

From a TILEMEGA_TRACE=STAGE=1 plan's mx_stage_trace.csv (per slot, stage,
CTA: start, work end, release in ns) and the plan manifest, per stage class
of one step (embed, qkv, attn, o, gate_up, down, head):
  measured  = CTA 0's release of the stage minus its release of the previous
              stage (the stage_bench envelope);
  work      = latest work end minus earliest start over CTAs;
  imbalance = latest minus earliest work end over CTAs;
  barrier   = release minus latest work end;
and the stage table's envelope for the same (N, K, B, Op, TN, D, Part) or
attention (D, Q, B, L, Tmin; linear in L). Medians over layers and slots.

Usage: stage_ledger.py --trace mx_stage_trace.csv --manifest plan.so.plan.json
       --model qwen3_0_6b --past 576 [--csv out.csv] [--cell name]
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--trace', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--past', type=int, required=True)
ap.add_argument('--table', type=Path, default=REPO / 'configs/targets/xcore1000_mx_decode.json')
ap.add_argument('--csv', type=Path)
ap.add_argument('--cell', default='')
# A stage_bench CSV run at L = past + 1 (STAGE_BENCH_LENGTHS): attention's T_c
# steps with L, so the table's linear interpolation is not the envelope there.
ap.add_argument('--direct', type=Path)
a = ap.parse_args()

m = json.loads(a.manifest.read_text())
layers, b = m['layers'], m['batch_lo']
table = json.loads(a.table.read_text())
CLASSES = ['qkv', 'attn', 'o', 'gate_up', 'down']


def stage_class(s):
    if s == 0:
        return 'embed', -1
    if s == 5 * layers + 1:
        return 'head', -1
    return CLASSES[(s - 1) % 5], (s - 1) // 5


rows = defaultdict(lambda: defaultdict(list))  # slot -> stage -> [(cta, start, end, release)]
for r in csv.DictReader(a.trace.open()):
    rows[int(r['slot'])][int(r['stage'])].append(
        (int(r['cta']), int(r['start_ns']), int(r['work_end_ns']), int(r['release_ns'])))

stats = defaultdict(lambda: defaultdict(list))
for slot, stages in rows.items():
    release0 = {s: next(v[3] for v in entries if v[0] == 0) for s, entries in stages.items()}
    for s, entries in stages.items():
        cls, _ = stage_class(s)
        starts = [e[1] for e in entries if e[1]]
        ends = [e[2] for e in entries if e[2]]
        if s - 1 in release0:
            stats[cls]['measured'].append(release0[s] - release0[s - 1])
        elif s == 0 and starts:
            stats[cls]['measured'].append(release0[s] - min(starts))
        if starts and ends:
            stats[cls]['work'].append(max(ends) - min(starts))
            stats[cls]['imbalance'].append(max(ends) - min(ends))
            stats[cls]['barrier'].append(release0[s] - max(ends))


def bench(cls):
    if cls == 'embed':
        rows_ = [r for r in table['embed'] if r['model'] == a.model and r['b'] == b]
        return rows_[0]['median_ns'] if rows_ else None
    if cls == 'attn':
        model = 'qwen3_0_6b' if a.model == 'qwen3_1_7b' else a.model
        pts = sorted((r['length'], r['median_ns']) for r in table['attention']
                     if r['model'] == model and r['b'] == b and r['tmin'] == m['attention_tmin'])
        length = a.past + 1
        for (l0, v0), (l1, v1) in zip(pts, pts[1:]):
            if l0 <= length <= l1:
                return v0 + (v1 - v0) * (length - l0) / (l1 - l0)
        return None
    c = m['classes'][cls]
    rows_ = [r for r in table['gemm'] if r['model'] == a.model and r['b'] == b and r['cls'] == cls and
             r['tn'] == c['tn'] and r['d'] == c['d'] and r['part'] == c['part']]
    return rows_[0]['median_ns'] if rows_ else None


def direct(cls):
    if cls != 'attn' or not a.direct:
        return None
    rows_ = [r for r in csv.DictReader(a.direct.open()) if r['kind'] == 'attention' and int(r['b']) == b and
             int(r['length']) == a.past + 1 and int(r['tmin']) == m['attention_tmin']]
    return float(rows_[0]['median_ns']) if rows_ else None


out = []
count = {'embed': 1, 'head': 1}
print('%-8s %10s %10s %6s %10s %10s %10s %5s' % ('class', 'measured', 'bench', 'ratio', 'work', 'imbal', 'barrier',
                                                 'n'))
total_m = total_b = 0
for cls in ['embed'] + CLASSES + ['head']:
    s = stats[cls]
    if not s['measured']:
        continue
    med = statistics.median(s['measured'])
    bn = bench(cls)
    per_step = 1 if cls in count else layers
    total_m += med * per_step
    total_b += (bn or 0) * per_step
    rec = dict(cell=a.cell, cls=cls, measured_ns=round(med), bench_ns=round(bn) if bn else '',
               ratio=round(med / bn, 3) if bn else '', work_ns=round(statistics.median(s['work'])),
               imbalance_ns=round(statistics.median(s['imbalance'])), barrier_ns=round(statistics.median(s['barrier'])),
               samples=len(s['measured']), per_step=per_step)
    dn = direct(cls)
    rec.update(bench_at_l_ns=round(dn) if dn else '', ratio_at_l=round(med / dn, 3) if dn else '')
    out.append(rec)
    print('%-8s %10d %10s %6s %10d %10d %10d %5d' % (cls, rec['measured_ns'], rec['bench_ns'], rec['ratio'],
                                                      rec['work_ns'], rec['imbalance_ns'], rec['barrier_ns'],
                                                      rec['samples']))
print('step: measured %.3f ms, bench %.3f ms (sum of stage medians x stages per step)' % (total_m / 1e6, total_b / 1e6))
if a.csv:
    new = not a.csv.exists()
    with a.csv.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        if new:
            w.writeheader()
        w.writerows(out)
