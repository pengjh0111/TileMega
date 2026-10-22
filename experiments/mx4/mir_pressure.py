#!/usr/bin/env python3
"""Peak MTreg pressure of one function in final MetaX MIR.

Input: the -print-after=stack-frame-layout dump (physical registers). Runs a
backward liveness over the CFG on 32-bit MTreg units and prints the blocks
whose peak live count is highest, with the instruction at the peak.
Usage: mir_pressure.py <mir> <function-substring> [top]
"""
import re
import sys

path, key = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 8
text = open(path).read()
start = None
for m in re.finditer(r'^# Machine code for function (\S+):', text, re.M):
    if key in m.group(1):
        start = m.end()
        name = m.group(1)
        break
if start is None:
    sys.exit('function not found')
body = text[start:text.index('# End machine code', start)]

REG = re.compile(r'(killed |undef |dead |implicit-def |implicit )*\$mtreg_(\d+)bits?_(\d+)')


def units(width, base):
    return range(base, base + max(1, width // 32))


blocks, order = {}, []
current = None
for line in body.split('\n'):
    m = re.match(r'^bb\.(\d+)', line)
    if m:
        current = int(m.group(1))
        blocks[current] = {'succ': [], 'ins': []}
        order.append(current)
        continue
    if current is None:
        continue
    s = line.strip()
    if s.startswith('successors:'):
        blocks[current]['succ'] = [int(x) for x in re.findall(r'%bb\.(\d+)', s.split(';')[0])]
        continue
    if not s or s.startswith(';') or s.startswith('liveins') or s.startswith('predecessors'):
        continue
    if '=' in s.split(' ')[0:3] or ' = ' in s:
        lhs, rhs = s.split(' = ', 1) if ' = ' in s else ('', s)
    else:
        lhs, rhs = '', s
    defs, uses = set(), set()
    for mm in REG.finditer(lhs):
        defs.update(units(int(mm.group(2)), int(mm.group(3))))
    for mm in REG.finditer(rhs):
        flags = mm.group(0)
        u = set(units(int(mm.group(2)), int(mm.group(3))))
        if 'implicit-def' in flags:
            defs |= u
        elif 'undef' in flags:
            continue
        else:
            uses |= u
    blocks[current]['ins'].append((s, defs, uses))
# Fall-through blocks list their successors explicitly in this dump.
live_in = {b: set() for b in blocks}
changed = True
while changed:
    changed = False
    for b in reversed(order):
        out = set()
        for s in blocks[b]['succ']:
            out |= live_in.get(s, set())
        live = set(out)
        for _, d, u in reversed(blocks[b]['ins']):
            live -= d
            live |= u
        if live != live_in[b]:
            live_in[b] = live
            changed = True
peaks = []
for b in order:
    out = set()
    for s in blocks[b]['succ']:
        out |= live_in.get(s, set())
    live = set(out)
    best, where = len(live), '(block end)'
    for ins, d, u in reversed(blocks[b]['ins']):
        live -= d
        live |= u
        if len(live | d) > best:
            best, where = len(live | d), ins
    peaks.append((best, b, where[:110], len(blocks[b]['ins'])))
peaks.sort(reverse=True)
print(name[:90])
for best, b, where, n in peaks[:top]:
    print('bb.%-4d peak %3d  (%4d instrs)  at: %s' % (b, best, n, where))

if len(sys.argv) > 4:
    b = int(sys.argv[4])
    touched = set()
    for _, d, u in blocks[b]['ins']:
        touched |= d | u
    through = sorted(live_in[b] - touched)
    print('bb.%d live-in %d, live-through (untouched) %d' % (b, len(live_in[b]), len(through)))
    # Where are the live-through units defined?
    defs = {}
    for bb in order:
        for ins, d, u in blocks[bb]['ins']:
            for r in d:
                if r in through:
                    defs.setdefault(r, []).append((bb, ins[:100]))
    seen = set()
    for r in through:
        for bb, ins in defs.get(r, [])[:1]:
            if (bb, ins) in seen:
                continue
            seen.add((bb, ins))
            print('  u%-3d bb.%-4d %s' % (r, bb, ins))
