#!/usr/bin/env python3
"""MX-1 to MX-2 per-cell comparison (MX2-E item 3).

MX-1 measured every cell with row-major weights and let the solver pick a GEMV
variant for small-batch decode; MX-2 runs fragment-packed weights with GEMV
kept out of the domain. The protocol is otherwise identical, so the two sets
are comparable cell by cell. Baselines are not re-run: their MX-1 values stand
and the ratio column is recomputed against them.
"""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--mx1', type=Path, default=Path('/root/tilemega/docs/results/MX1_serving_c550.csv'))
ap.add_argument('--mx2', type=Path, default=Path('/root/tilemega/runs/mx2/tm'))
ap.add_argument('--impl', type=Path, default=Path('/root/tilemega/runs/mx2/selected_impl.csv'))
ap.add_argument('--out', type=Path)
a = ap.parse_args()

old, base = {}, {}
for row in csv.DictReader(a.mx1.open()):
    key = (row['model'], int(row['batch']))
    target = old if row['system'] == 'tilemega' else base
    target.setdefault((*key, row['system']) if target is base else key, []).append(row)

impl = {}
if a.impl.exists():
    for row in csv.DictReader(a.impl.open()):
        impl.setdefault((row['model'], int(row['batch'])), {})[row['phase']] = row

new = defaultdict(list)
for path in sorted(a.mx2.glob('*/B*/measure/round*/measurements.json')):
    model, batch = path.parts[-5], int(path.parts[-4].lstrip('B'))
    new[(model, batch)].append(json.loads(path.read_text()))


def med(rows, key, scale=1.0):
    values = [float(r[key]) * scale for r in rows if r.get(key) not in (None, '', 'nan')]
    return statistics.median(values) if values else float('nan')


rows = []
for key in sorted(new):
    model, batch = key
    n, o = new[key], old.get(key, [])
    vllm = med(base.get((model, batch, 'vllm'), []), 'tpot_ms')
    cell = dict(
        model=model, batch=batch,
        tpot_mx1=med(o, 'tpot_ms'), tpot_mx2=med(n, 'tpot_seconds', 1e3),
        ttft_mx1=med(o, 'ttft_ms'), ttft_mx2=med(n, 'ttft_seconds', 1e3),
        mbu_mx1=med(o, 'mbu'), tokens_per_s=med(n, 'output_tokens_per_second'),
        vllm_tpot=vllm, rounds=len(n),
        layout=(impl.get(key, {}).get('decode') or {}).get('layout', '?'),
        variants=(impl.get(key, {}).get('decode') or {}).get('variants', '?'))
    cell['tpot_delta_pct'] = ((cell['tpot_mx2'] - cell['tpot_mx1']) / cell['tpot_mx1'] * 100
                              if cell['tpot_mx1'] == cell['tpot_mx1'] else float('nan'))
    cell['vs_vllm_mx1'] = cell['tpot_mx1'] / vllm if vllm == vllm else float('nan')
    cell['vs_vllm_mx2'] = cell['tpot_mx2'] / vllm if vllm == vllm else float('nan')
    rows.append(cell)

print('%-13s %3s %8s %8s %8s | %7s %7s | %6s %6s %s' % (
    'model', 'B', 'tpot_1', 'tpot_2', 'delta', 'ttft_1', 'ttft_2', 'vLLM1', 'vLLM2', 'layout'))
for r in rows:
    print('%-13s %3d %8.2f %8.2f %+7.1f%% | %7.1f %7.1f | %5.2fx %5.2fx %s/v%s' % (
        r['model'], r['batch'], r['tpot_mx1'], r['tpot_mx2'], r['tpot_delta_pct'],
        r['ttft_mx1'], r['ttft_mx2'], r['vs_vllm_mx1'], r['vs_vllm_mx2'],
        r['layout'], r['variants']))
if rows:
    gains = [r['tpot_delta_pct'] for r in rows if r['tpot_delta_pct'] == r['tpot_delta_pct']]
    print('\n%d cells; median TPOT change %+.1f%%, best %+.1f%%, worst %+.1f%%' % (
        len(rows), statistics.median(gains), min(gains), max(gains)))
if a.out:
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('-> %s' % a.out)
