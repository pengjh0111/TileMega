#!/usr/bin/env python3
"""MX3-6 table: TM-MX2 (or MX-2 where TM-MX2 was not re-measured) -> MX-3.

TM-MX2 was rebuilt and measured on five cells and reproduced MX-2 within 0.2%
on the four MX-2 had measured, so MX-2's own medians stand in for TM-MX2 on
the remaining cells; the `ref` column says which one each row uses. Baseline
ratios use MX-1's vLLM-MetaX, SGLang and torch.compile medians (not re-run).
Prints a Markdown table and the G-PERF verdict; with --csv also the rows.
"""
import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--serving', type=Path, default=REPO / 'docs/results/MX3_serving_c550.csv')
ap.add_argument('--csv', type=Path)
ap.add_argument('--campaign', default='mx3-rerun2', help='the MX-3 result campaign')
a = ap.parse_args()

rows = list(csv.DictReader(a.serving.open()))


def med(campaign, system, model, batch, field):
    vals = [float(r[field]) for r in rows
            if r['campaign'] == campaign and r['system'] == system
            and r['model'] == model and int(r['batch']) == batch
            and r[field] not in ('', 'nan')]
    return statistics.median(vals) if vals else None


def impl(model, batch):
    for r in rows:
        if r['campaign'] == a.campaign and r['model'] == model and int(r['batch']) == batch:
            return r['impl']
    return ''


out = []
for model in ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_1_8b'):
    for batch in (1, 2, 4, 8, 16):
        new = med(a.campaign, 'tilemega', model, batch, 'tpot_ms')
        first = med('mx3', 'tilemega', model, batch, 'tpot_ms')
        ref, ref_name = med('tm-mx2', 'tilemega', model, batch, 'tpot_ms'), 'TM-MX2'
        if ref is None:
            ref, ref_name = med('mx2', 'tilemega', model, batch, 'tpot_ms'), 'MX-2'
        if ref is None:
            # Qwen3-1.7B B4: MX-2's prefill never solved, so MX-1 is the last
            # TileMega number for the cell.
            ref, ref_name = med('mx1', 'tilemega', model, batch, 'tpot_ms'), 'MX-1'
        vllm = med('mx1', 'vllm', model, batch, 'tpot_ms')
        sgl = med('mx1', 'sglang', model, batch, 'tpot_ms')
        tc = med('mx1', 'torchcompile', model, batch, 'tpot_ms')
        out.append(dict(
            model=model, batch=batch, ref=ref_name,
            ref_tpot=ref, mx3_tpot=new,
            delta_pct=(new / ref - 1) * 100 if new and ref else None,
            mx3_first_tpot=first,
            mx3_ttft=med(a.campaign, 'tilemega', model, batch, 'ttft_ms'),
            mx3_mbu=med(a.campaign, 'tilemega', model, batch, 'mbu'),
            vs_vllm=new / vllm if new and vllm else None,
            vs_sglang=new / sgl if new and sgl else None,
            vs_torchcompile=new / tc if new and tc else None,
            impl=impl(model, batch)))


def f(v, fmt):
    return fmt % v if v is not None else '—'


print('| cell | ref | ref TPOT | MX-3 TPOT | Δ | TTFT | MBU | /vLLM | /SGLang |')
print('|---|---|---|---|---|---|---|---|---|')
names = {'qwen3_0_6b': 'Qwen3-0.6B', 'qwen3_1_7b': 'Qwen3-1.7B',
         'llama3_2_1b': 'Llama-3.2-1B', 'llama3_1_8b': 'Llama-3.1-8B'}
for r in out:
    print('| %s B%d | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        names[r['model']], r['batch'], r['ref'], f(r['ref_tpot'], '%.3f'),
        f(r['mx3_tpot'], '%.3f'), f(r['delta_pct'], '%+.1f%%'),
        f(r['mx3_ttft'], '%.1f'), f(r['mx3_mbu'] * 100 if r['mx3_mbu'] else None, '%.0f%%'),
        f(r['vs_vllm'], '%.2fx'), f(r['vs_sglang'], '%.2fx')))
done = [r for r in out if r['delta_pct'] is not None]
slower = [r for r in done if r['delta_pct'] > 1.0]
if done:
    print('\n%d of 20 cells measured; median change %+.1f%%, range %+.1f%% to %+.1f%%; '
          '%d slower than the reference by more than 1%%.' % (
              len(done), statistics.median(r['delta_pct'] for r in done),
              min(r['delta_pct'] for r in done), max(r['delta_pct'] for r in done), len(slower)))
    for r in slower:
        print('  slower: %s B%d %+.1f%%' % (r['model'], r['batch'], r['delta_pct']))
if a.csv:
    with a.csv.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
