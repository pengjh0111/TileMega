#!/usr/bin/env python3
"""Local accesses and waits in a MACA kernel's final machine code (MX3-1 item 5).

MACA ships no machine-code disassembler (`/opt/maca/mxgpu_llvm/bin` has
llvm-dis for IR only, no llvm-objdump), but its device compiler is LLVM, so
the machine IR after the last codegen pass is the instruction stream: waits
are `ARRIVE n` (inserted by metaxgpu-insert-arrivecnts, or written by hand
through __builtin_mxc_arrive_gvmcnt), private-memory (local) accesses carry
`addrspace 5`, and the MMA is `MMA_16X16X16BF16`. Produce the dump with

  cucc <the plan's build flags> --device-obj plan.cu -o /dev/null \
       -mllvm -print-after=stack-frame-layout > plan.mir 2>&1

A loop is a back edge bb.j -> bb.i (i <= j) in layout order; its body is
bb.i..bb.j. Per function this prints instruction and local-access totals, and
per loop the MMAs, global loads, local accesses and every ARRIVE with its
operand in program order, so a wait-for-all (ARRIVE with a small operand
before the first MMA of an iteration) is visible.
"""
import argparse
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('mir', type=Path)
ap.add_argument('--csv', type=Path)
a = ap.parse_args()

FUNC = re.compile(r'^# Machine code for function (\S+?):', re.M)
BB = re.compile(r'^bb\.(\d+)')
SUCC = re.compile(r'%bb\.(\d+)')
OPC = re.compile(r'^\s+(?:[^=\n]*=\s+)?(?:(?:contract|nuw|nsw|exact|nnan|ninf|nsz|arcp|afn|reassoc|disjoint)\s+)*([A-Z][A-Z0-9_]+)')


def short(name):
    for key in ('ServingEpilogue', 'GemmCombineTaskBody', 'GemmStageTaskBody', 'RunServingAttentionTask',
                'RunServingScalarTask', 'RunStage', 'RunTask', 'ServingEpilogueDispatch'):
        if key in name:
            return key + name[-40:] if key == 'ServingEpilogue' else key
    return name[:60]


text = a.mir.read_text(errors='replace')
starts = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
rows = []
for idx, (pos, name) in enumerate(starts):
    body = text[pos:starts[idx + 1][0] if idx + 1 < len(starts) else len(text)]
    blocks, order, cur = {}, [], None
    for line in body.splitlines():
        m = BB.match(line)
        if m:
            cur = int(m.group(1))
            blocks[cur] = dict(succ=[], ins=[])
            order.append(cur)
            continue
        if cur is None:
            continue
        if line.strip().startswith('successors:'):
            blocks[cur]['succ'] = [int(s) for s in SUCC.findall(line.split(';')[0])]
            continue
        m = OPC.match(line)
        if m:
            blocks[cur]['ins'].append((m.group(1), line))
    pos_of = {b: i for i, b in enumerate(order)}

    def kind(op, line):
        if op.startswith('MMA_'):
            return 'mma'
        if op == 'ARRIVE':
            return 'arrive'
        if 'addrspace 5' in line or re.match(r'(LD|ST)P_', op):
            return 'local_load' if op.startswith('LD') else 'local_store'
        if op.startswith('LDG') or ('addrspace 1' in line and op.startswith('LD')):
            return 'global_load'
        return 'other'

    total = sum(len(b['ins']) for b in blocks.values())
    locals_ = sum(1 for b in blocks.values() for op, l in b['ins'] if kind(op, l).startswith('local'))
    print('%-40s instrs=%5d local=%3d' % (short(name), total, locals_))
    loops = []
    for b in order:
        for s in blocks[b]['succ']:
            if s in pos_of and pos_of[s] <= pos_of[b]:
                loops.append((pos_of[s], pos_of[b]))
    for lo, hi in sorted(set(loops)):
        seq = [(op, l) for bi in order[lo:hi + 1] for op, l in blocks[bi]['ins']]
        kinds = [kind(op, l) for op, l in seq]
        mma = kinds.count('mma')
        loc = sum(k.startswith('local') for k in kinds)
        if not mma and not loc:
            continue
        arrives = []
        first_mma = kinds.index('mma') if mma else None
        for i, (op, l) in enumerate(seq):
            if op == 'ARRIVE':
                n = int(re.search(r'ARRIVE\s+(\d+)', l).group(1))
                arrives.append('%d%s' % (n, '*' if first_mma is not None and i < first_mma else ''))
        row = dict(function=short(name), loop='bb.%d-bb.%d' % (order[lo], order[hi]),
                   instrs=len(seq), mma=mma, global_loads=kinds.count('global_load'),
                   local_loads=kinds.count('local_load'), local_stores=kinds.count('local_store'),
                   arrives=' '.join(arrives))
        rows.append(row)
        print('   loop %-14s instrs=%5d mma=%3d gload=%3d local=%d/%d arrive=[%s]' % (
            row['loop'], row['instrs'], mma, row['global_loads'], row['local_loads'],
            row['local_stores'], row['arrives'][:160]))
if a.csv:
    import csv
    with a.csv.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['function'])
        w.writeheader()
        w.writerows(rows)
