#!/usr/bin/env python3
"""MX4_resources_c550.csv (MX4 prompt §8, §6 D3): per MX-4 decode plan its
resources (MTreg, STreg, stack, static smem, code bytes, compile time, from
the manifest), the machine-code wait check of the program kernel
(experiments/mx4/mir_check.py on the plan's own build command plus
-print-after=stack-frame-layout) and the code size of each stage body, as
the instruction count of that body's stage kernel in a stage_bench MIR dump
(the same body code in a loop with the program's barrier).

Usage: resources.py --plans DIR... --bench-mir stage_bench.mir --csv out.csv
"""
import argparse
import csv
import json
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--plans', nargs='+', type=Path, required=True)
ap.add_argument('--bench-mir', type=Path, required=True)
ap.add_argument('--csv', type=Path, required=True)
a = ap.parse_args()

OPS = {'kQkvRope': 0, 'kResidual': 1, 'kSwiGLU': 2, 'kArgmax': 3}
PARTS = {'kStreamK': 0, 'kTileRange': 1}
bench = a.bench_mir.read_text(errors='replace')
sizes = {}
heads = list(re.finditer(r'^# Machine code for function (\S+?):', bench, re.M))
for i, m in enumerate(heads):
    body = bench[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(bench)]
    count = sum(1 for line in body.split('\n') if line.startswith('  ') and not line.strip().startswith(
        ('successors', 'liveins', 'predecessors', ';')) and line.strip())
    g = re.search(r'GemmStage.*MxDecodeGemmILi(\d+)ELi(\d)ELNS\d_10MxDecodeOpE(\d)ELb([01])ELNS\d_6MxPartE([01])'
                  r'(?:ENS\d_(?:6MxRopeILi(\d+)ELi(\d+)|8MxNoRope))?', m.group(1))
    if g:
        sizes[('gemm',) + tuple(int(x) for x in g.groups()[:5]) + ((int(g.group(6)), int(g.group(7))) if g.group(6)
                                                                  else (None,))] = count
    t = re.search(r'AttnStage.*MxDecodeAttentionILi(\d+)ELi(\d+)ELi(\d+)', m.group(1))
    if t:
        sizes[('attn',) + tuple(int(x) for x in t.groups())] = count

rows = []
for plan in a.plans:
    so = plan / 'plan.so'
    m = json.loads(Path(str(so) + '.plan.json').read_text())
    command = Path(str(so) + '.build_command.txt').read_text().strip()
    words = [w for w in shlex.split(command) if w not in ('-shared', '-Xcompiler=-fPIC')]
    out = words.index('-o')
    words[out + 1] = '/dev/null'
    words = words[:out] + ['--device-obj'] + words[out:]
    mir = plan / 'program.mir'
    with mir.open('w') as f:
        subprocess.run(words + ['-mllvm', '-print-after=stack-frame-layout'], stdout=f, stderr=subprocess.STDOUT)
    check = subprocess.run(['python3', str(HERE / 'mir_check.py'), str(mir)], capture_output=True, text=True).stdout
    line = next((l for l in check.splitlines() if 'mx_decode_kernel' in l), '')
    rec = dict(plan=str(plan), model=m['model'], batch=m['batch_lo'], executor=m['executor'],
               classes=json.dumps(m['classes'], separators=(',', ':')), tmin=m['attention_tmin'],
               max_blocks=m['attention_max_blocks'], barrier_prefetch=m.get('barrier_prefetch', 1),
               watchdog=m['watchdog'], **m['resources'], smem=m['smem'],
               mir=' '.join(line.split()[:5]) if line else 'missing')
    rope = {'qwen3_0_6b': (128, 2), 'qwen3_1_7b': (128, 2), 'llama3_2_1b': (64, 4), 'llama3_1_8b': (128, 4)}
    for cls, op, norm in (('qkv', 'kQkvRope', 1), ('o', 'kResidual', 0), ('gate_up', 'kSwiGLU', 1),
                          ('down', 'kResidual', 0), ('head', 'kArgmax', 1)):
        c = m['classes'][cls]
        key = ('gemm', c['tn'], c['d'], OPS[op], norm, PARTS[c['part']],
               rope.get(m['model']) if op == 'kQkvRope' else None)
        rec[cls + '_instrs'] = sizes.get(key, '')
    rec['attn_instrs'] = sizes.get(('attn', m['head_dim'], m['q_per_kv'], m['attention_tmin']), '')
    rows.append(rec)
    print(rec['model'], rec['batch'], 'stack', rec['stack_bytes'], 'mtreg', rec['mtreg'], rec['mir'])
new = not a.csv.exists()
with a.csv.open('a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
