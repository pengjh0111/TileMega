#!/usr/bin/env python3
"""MX-4 measured stage table (MX4 prompt §4.2) from stage_bench runs.

Reads the CSVs of the fresh stage_bench processes and the bench build log
(-resource-usage: each stage kernel's stack frame), takes per key the median
over processes of the per-process median and p90 envelopes, and writes
configs/targets/xcore1000_mx_decode.json, referenced from xcore1000.json as
`mx_decode_costs`. Rows whose kernel spilled carry `spill_bytes` > 0.

Usage: stage_table.py --runs r1.csv r2.csv r3.csv --build build.log --card 7
"""
import argparse
import csv
import datetime
import json
import re
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', type=Path, required=True)
ap.add_argument('--build', type=Path, required=True)
ap.add_argument('--card', required=True)
ap.add_argument('--out', type=Path, default=REPO / 'configs/targets/xcore1000_mx_decode.json')
a = ap.parse_args()

OPS = ['kQkvRope', 'kResidual', 'kSwiGLU', 'kArgmax']
PARTS = ['kStreamK', 'kTileRange']
spill_gemm, spill_attn = {}, {}
log = a.build.read_text(errors='replace')
for m in re.finditer(r'Function properties for\s+(\S+)\s*\n\s*(\d+) bytes stack frame', log):
    name, stack = m.group(1), int(m.group(2))
    g = re.search(r'GemmStage.*MxDecodeGemmILi(\d+)ELi(\d)ELNS\d_10MxDecodeOpE(\d)ELb([01])ELNS\d_6MxPartE([01])'
                  r'(?:ENS\d_(?:6MxRopeILi(\d+)ELi(\d+)|8MxNoRope))?', name)
    if g:
        tn, d, op, norm, part = (int(x) for x in g.groups()[:5])
        rope = (int(g.group(6)), int(g.group(7))) if g.group(6) else None
        spill_gemm[(tn, d, OPS[op], norm, PARTS[part], rope)] = stack
    t = re.search(r'AttnStage.*MxDecodeAttentionILi(\d+)ELi(\d+)ELi(\d+)', name)
    if t:
        spill_attn[tuple(int(x) for x in t.groups())] = stack

ROPE = {'qwen3_0_6b': (128, 2), 'qwen3_1_7b': (128, 2), 'llama3_2_1b': (64, 4), 'llama3_1_8b': (128, 4)}
rows = {}
for path in a.runs:
    for r in csv.DictReader(path.open()):
        if r['kind'] == 'gemm':
            key = ('gemm', r['model'], r['class'], int(r['n']), int(r['k']), int(r['b']), r['op'], int(r['norm']),
                   int(r['tn']), int(r['d']), r['part'])
        elif r['kind'] == 'attention':
            key = ('attention', r['model'], int(r['dim']), int(r['q']), int(r['hkv']), int(r['b']), int(r['length']),
                   int(r['tmin']))
        else:
            key = ('embed', r['model'], int(r['n']), int(r['b']))
        rows.setdefault(key, []).append((float(r['median_ns']), float(r['p90_ns']), int(r['iterations'])))

table = {'gemm': [], 'attention': [], 'embed': []}
for key, values in sorted(rows.items()):
    median = statistics.median(v[0] for v in values)
    p90 = statistics.median(v[1] for v in values)
    rec = dict(median_ns=round(median), p90_ns=round(p90), processes=len(values),
               spread=round((max(v[0] for v in values) - min(v[0] for v in values)) / median, 4))
    if key[0] == 'gemm':
        _, model, cls, n, k, b, op, norm, tn, d, part = key
        rope = ROPE[model] if op == 'kQkvRope' else None
        rec = dict(model=model, cls=cls, n=n, k=k, b=b, op=op, norm=norm, tn=tn, d=d, part=part,
                   spill_bytes=spill_gemm.get((tn, d, op, norm, part, rope), -1), **rec)
        table['gemm'].append(rec)
    elif key[0] == 'attention':
        _, model, dim, q, hkv, b, length, tmin = key
        rec = dict(model=model, dim=dim, q=q, hkv=hkv, b=b, length=length, tmin=tmin,
                   spill_bytes=spill_attn.get((dim, q, tmin), -1), **rec)
        table['attention'].append(rec)
    else:
        _, model, hidden, b = key
        table['embed'].append(dict(model=model, hidden=hidden, b=b, **rec))

head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short=9', 'HEAD'], capture_output=True,
                      text=True).stdout.strip()
doc = dict(
    provenance=dict(tool='experiments/mx4/stage_bench.cu', head=head, date=datetime.date.today().isoformat(),
                    card='C550 GPU %s' % a.card, processes=len(a.runs),
                    iterations=sorted({v[2] for vs in rows.values() for v in vs}),
                    method='median over processes of the per-process median (and p90) stage envelope; '
                           'envelope = consecutive post-barrier timestamps of CTA 0'),
    **table)
a.out.write_text(json.dumps(doc, indent=1) + '\n')
target = REPO / 'configs/targets/xcore1000.json'
spec = json.loads(target.read_text())
if spec.get('mx_decode_costs') != a.out.name:
    spec['mx_decode_costs'] = a.out.name
    target.write_text(json.dumps(spec, indent=2) + '\n')
print('gemm %d, attention %d, embed %d rows; spilled gemm kernels %d, attention %d' % (
    len(table['gemm']), len(table['attention']), len(table['embed']),
    sum(1 for r in table['gemm'] if r['spill_bytes'] > 0), sum(1 for r in table['attention'] if r['spill_bytes'] > 0)))
