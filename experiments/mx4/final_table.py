#!/usr/bin/env python3
"""MX-4 final 20-cell table (MX4 prompt §9.1) from scripts/mx4_final_run.sh.

Per cell <cells>/<model>/B<b>: measure/round<r>/measurements.json (one fresh
process per round, warm-up 1 + timed 2; the cell value is the median over
rounds), the decode pin (decode/plan.so.mx4_pin.json: (TN, D, Part) per GEMM
class and Tmin) and the layer count (stages per step = 5L + 2). Compared with
MX-3 final (docs/results/MX3_final_c550.csv, mx3_tpot), with MX-1's vLLM-MetaX
and SGLang medians (campaign mx1 of MX3_serving_c550.csv, the baseline the
prompt names) and with the MX-4 re-run of the baselines (MX4_baselines_llm_c550.csv).
"Distinguishable" (§0.3): |Δ| beyond max(1%, the range of either side's rounds).
Writes --csv (one row per cell and round plus the median row) and prints a
Markdown table.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--cells', type=Path, default=REPO / 'runs/mx4/final')
ap.add_argument('--csv', type=Path, required=True)
a = ap.parse_args()

PEAK_BYTES_PER_S = 1.64e12  # as MX-3 (scripts/mx3_serving_csv.py)
GEOMETRY = {  # weight bytes, layers, KV heads, head dim
    'qwen3_0_6b': (1.11 * 2**30, 28, 8, 128),
    'qwen3_1_7b': (3.20 * 2**30, 28, 8, 128),
    'llama3_2_1b': (2.30 * 2**30, 16, 8, 64),
    'llama3_1_8b': (14.96 * 2**30, 32, 8, 128),
    # Outside the 20 (no MX-3 reference): measured after the Rq fix.
    'llama3_2_3b': (5.98 * 2**30, 28, 8, 128),
}
MODELS = tuple(GEOMETRY)
CLASSES = ('qkv', 'o', 'gate_up', 'down', 'head')


def mbu(model, batch, tpot_ms, max_tokens):
    weights, layers, kv_heads, head_dim = GEOMETRY[model]
    kv = 2 * batch * layers * kv_heads * head_dim * (64 + max_tokens / 2) * 2
    return (weights + kv) / (tpot_ms * 1e-3) / PEAK_BYTES_PER_S


mx3 = {(r['model'], int(r['batch'])): float(r['mx3_tpot'])
       for r in csv.DictReader(open(REPO / 'docs/results/MX3_final_c550.csv'))}
mx3_rounds = {}
for r in csv.DictReader(open(REPO / 'docs/results/MX3_serving_c550.csv')):
    if r['campaign'] == 'mx3-rerun2':
        mx3_rounds.setdefault((r['model'], int(r['batch'])), []).append(float(r['tpot_ms']))
mx1 = {}
for r in csv.DictReader(open(REPO / 'docs/results/MX3_serving_c550.csv')):
    if r['campaign'] == 'mx1' and r['system'] in ('vllm', 'sglang'):
        mx1.setdefault((r['system'], r['model'], int(r['batch'])), []).append(float(r['tpot_ms']))
mx4_base = {(r['system'], r['model'], int(r['batch'])): float(r['tpot_ms'])
            for r in csv.DictReader(open(REPO / 'docs/results/MX4_baselines_llm_c550.csv')) if r['tpot_ms']}

out, table = [], []
for model in MODELS:
    for batch in (1, 2, 4, 8, 16):
        cell = a.cells / model / ('B%d' % batch)
        runs = []
        for path in sorted(cell.glob('measure/round*/measurements.json')):
            d = json.loads(path.read_text())
            if not d.get('timed_tokens_identical', True):
                continue
            runs.append(d)
            out.append(dict(model=model, batch=batch, round=path.parent.name,
                            tpot_ms=1e3 * d['tpot_seconds'], ttft_ms=1e3 * d['ttft_seconds'],
                            e2e_s=d['e2e_seconds'], tokens_per_s=d['output_tokens_per_second'],
                            mbu=mbu(model, batch, 1e3 * d['tpot_seconds'], d['max_tokens'])))
        if not runs:
            table.append(dict(model=model, batch=batch, missing=True))
            continue
        med = lambda k: statistics.median(x[k] for x in runs)
        tpot = 1e3 * med('tpot_seconds')
        pin = json.loads((cell / 'decode/plan.so.mx4_pin.json').read_text())['mx_decode']
        layers = GEOMETRY[model][1]
        rounds = [1e3 * x['tpot_seconds'] for x in runs]
        ref = mx3.get((model, batch))
        spread = max(0.01, (max(rounds) - min(rounds)) / tpot,
                     (max(mx3_rounds.get((model, batch), [0])) - min(mx3_rounds.get((model, batch), [0])))
                     / ref) if ref else None
        delta = tpot / ref - 1 if ref else None
        row = dict(model=model, batch=batch, round='median', tpot_ms=tpot, ttft_ms=1e3 * med('ttft_seconds'),
                   e2e_s=med('e2e_seconds'), tokens_per_s=med('output_tokens_per_second'),
                   mbu=mbu(model, batch, tpot, runs[0]['max_tokens']),
                   mx3_tpot_ms=ref or '', delta_vs_mx3='' if delta is None else delta,
                   distinguishable_slower='' if delta is None else delta > spread,
                   vs_vllm_mx1=tpot / statistics.median(mx1[('vllm', model, batch)]),
                   vs_sglang_mx1=tpot / statistics.median(mx1[('sglang', model, batch)]),
                   vs_vllm_mx4=tpot / mx4_base[('vllm', model, batch)] if ('vllm', model, batch) in mx4_base else '',
                   vs_sglang_mx4=tpot / mx4_base[('sglang', model, batch)] if ('sglang', model, batch) in mx4_base else '',
                   vs_torchcompile_mx4=tpot / mx4_base[('torchcompile', model, batch)]
                   if ('torchcompile', model, batch) in mx4_base else '',
                   tmin=pin['tmin'], stages_per_step=5 * layers + 2,
                   classes=' '.join('%s=%d/%d/%s' % (c, pin[c]['tn'], pin[c]['d'], pin[c]['part'][1:])
                                    for c in CLASSES))
        out.append(row)
        table.append(row)

fields = list(max(out, key=len))
with a.csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)

print('| cell | TPOT ms | MX-3 | Δ | TTFT ms | tok/s | MBU | /vLLM | /SGLang | /vLLM⁴ | /SGLang⁴ | /t.c⁴ |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in table:
    if r.get('missing'):
        print('| %s B%d | not measured |' % (r['model'], r['batch']))
        continue
    ref = r['mx3_tpot_ms'] != ''
    print('| %s B%d | %.3f | %s | %s%s | %.1f | %.0f | %.0f%% | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
        r['model'], r['batch'], r['tpot_ms'], '%.3f' % r['mx3_tpot_ms'] if ref else '—',
        '%+.1f%%' % (100 * r['delta_vs_mx3']) if ref else '—',
        ' (slower)' if r['distinguishable_slower'] is True else '', r['ttft_ms'], r['tokens_per_s'], 100 * r['mbu'],
        r['vs_vllm_mx1'], r['vs_sglang_mx1'], r['vs_vllm_mx4'] or float('nan'),
        r['vs_sglang_mx4'] or float('nan'), r['vs_torchcompile_mx4'] or float('nan')))
done = [r for r in table if not r.get('missing') and r['mx3_tpot_ms'] != '']
if done:
    print('\n%d/20 measured; Δ vs MX-3 median %+.1f%% (%+.1f%% .. %+.1f%%); %d distinguishably slower; '
          'geomean /vLLM(MX-1) %.3f' % (
              len(done), 100 * statistics.median(r['delta_vs_mx3'] for r in done),
              100 * min(r['delta_vs_mx3'] for r in done), 100 * max(r['delta_vs_mx3'] for r in done),
              sum(r['distinguishable_slower'] for r in done),
              statistics.geometric_mean(r['vs_vllm_mx1'] for r in done)))
