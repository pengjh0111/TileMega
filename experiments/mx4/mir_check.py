#!/usr/bin/env python3
"""MX-4 machine-code checks of the decode bodies and programs (MX4 prompt
§3.5 rule 4, G-B): per function, the local (stack) accesses and every
ARRIVE 4160 (wait for all global and flat loads) that falls between two MMAs
of one block or inside an innermost loop that runs MMAs.

Produce the dump with
  cucc <flags> --device-obj <source> -o /dev/null \
       -mllvm -print-after=stack-frame-layout > out.mir 2>&1
Usage: mir_check.py out.mir [--csv out.csv]; exit 1 if any function fails.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('mir', type=Path)
ap.add_argument('--csv', type=Path)
a = ap.parse_args()

text = a.mir.read_text(errors='replace')
heads = list(re.finditer(r'^# Machine code for function (\S+?):', text, re.M))
rows, failed = [], 0
for i, m in enumerate(heads):
    name = m.group(1)
    body = text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
    body = body[:body.find('# End machine code')]
    blocks, order = {}, []
    current = None
    for line in body.split('\n'):
        b = re.match(r'^bb\.(\d+)', line)
        if b:
            current = int(b.group(1))
            blocks[current] = {'succ': [], 'seq': []}
            order.append(current)
            continue
        if current is None:
            continue
        s = line.strip()
        if s.startswith('successors:'):
            blocks[current]['succ'] = [int(x) for x in re.findall(r'%bb\.(\d+)', s.split(';')[0])]
        elif 'MMA_16X16X16BF16' in s:
            blocks[current]['seq'].append('M')
        elif re.search(r'\bARRIVE (\d+)', s):
            blocks[current]['seq'].append(int(re.search(r'\bARRIVE (\d+)', s).group(1)))
    local = len(re.findall(r'addrspace 5|%stack\.', body))
    in_block = 0
    for blk in blocks.values():
        seq = blk['seq']
        mmas = [j for j, x in enumerate(seq) if x == 'M']
        if len(mmas) >= 2:
            in_block += sum(1 for x in seq[mmas[0]:mmas[-1]] if x == 4160)
    # Innermost loops (a back edge whose body holds no other back edge): the
    # steady MMA loops. An outer loop also spans a segment's tail (reduction,
    # last arriver), after its last MMA, where waiting for everything is fine.
    position = {b: j for j, b in enumerate(order)}
    loops = [(position[s], position[b]) for b in order for s in blocks[b]['succ']
             if s in position and position[s] <= position[b]]
    in_loop = 0
    for lo, hi in loops:
        if any((l2, h2) != (lo, hi) and lo <= l2 and h2 <= hi for l2, h2 in loops):
            continue
        span = order[lo:hi + 1]
        if any('M' in blocks[x]['seq'] for x in span):
            in_loop += sum(blocks[x]['seq'].count(4160) for x in span)
    mma = sum(blk['seq'].count('M') for blk in blocks.values())
    ok = local == 0 and in_block == 0 and in_loop == 0
    failed += mma > 0 and not ok
    rows.append(dict(function=name, mma=mma, local=local, arrive4160_between_mma=in_block,
                     arrive4160_in_mma_loop=in_loop, ok=int(ok)))
for r in rows:
    if r['mma']:
        print('%-4s mma=%-5d local=%-4d 4160(block)=%-3d 4160(loop)=%-3d %s' % (
            'ok' if r['ok'] else 'FAIL', r['mma'], r['local'], r['arrive4160_between_mma'],
            r['arrive4160_in_mma_loop'], r['function'][:100]))
print('%d functions with MMAs, %d fail' % (sum(1 for r in rows if r['mma']), failed))
if a.csv:
    with a.csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
sys.exit(1 if failed else 0)
