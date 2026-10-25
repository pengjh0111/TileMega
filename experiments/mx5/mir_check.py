#!/usr/bin/env python3
"""Check full function identities and straight-line MMA rings in MACA MIR."""
import argparse
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mir', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); text = a.mir.read_text()
    results = []
    for match in re.finditer(r'^# Machine code for function (\S+):', text, re.M):
        name = match[1]
        end = text.find('# End machine code', match.end())
        body = text[match.end():end]
        blocks = {}; order = []; current = None
        for line in body.splitlines():
            m = re.match(r'^bb\.(\d+)', line)
            if m:
                current = int(m[1]); order.append(current); blocks[current] = dict(lines=[], successors=[])
            if current is None: continue
            blocks[current]['lines'].append(line)
            if line.strip().startswith('successors:'):
                blocks[current]['successors'] = [int(v) for v in re.findall(r'%bb\.(\d+)', line)]
        position = {v: i for i, v in enumerate(order)}
        predecessors = {v: set() for v in order}
        for block in order:
            for successor in blocks[block]['successors']:
                if successor in predecessors: predecessors[successor].add(block)
        reachable = set(); pending = [order[0]]
        while pending:
            block = pending.pop()
            if block in reachable: continue
            reachable.add(block)
            pending.extend(s for s in blocks[block]['successors'] if s in position)
        dominators = {b: ({b} if b == order[0] else set(reachable)) for b in reachable}
        changed = True
        while changed:
            changed = False
            for block in order[1:]:
                if block not in reachable: continue
                pred = predecessors[block] & reachable
                common = set.intersection(*(dominators[v] for v in pred)) if pred else set()
                updated = {block} | common
                if updated != dominators[block]: dominators[block] = updated; changed = True
        loops = set()
        for block in order:
            for successor in blocks[block]['successors']:
                if successor not in dominators.get(block, set()): continue
                nodes = {successor, block}; pending = [block] if block != successor else []
                while pending:
                    tail = pending.pop()
                    for pred in predecessors[tail]:
                        if pred not in nodes: nodes.add(pred); pending.append(pred)
                loops.add((successor, block, tuple(sorted(nodes, key=position.get))))
        ring_rows = []
        for header, latch, nodes in sorted(loops):
            sequence = '\n'.join(line for b in nodes for line in blocks[b]['lines'])
            if 'MMA_16X16X16BF16' not in sequence or 'LDG_' not in sequence: continue
            if re.search(r'\bBARRIER\b|INLINEASM &barrier', sequence): continue
            waits = [int(v) for v in re.findall(r'\bARRIVE (\d+)', sequence)]
            generic = re.findall(r'\bLDX_\w+', sequence)
            ring_rows.append(dict(begin=header, end=latch, blocks=list(nodes), waits=waits,
                                  generic_loads=len(generic), wait_all=4160 in waits))
        results.append(dict(function=name, rings=ring_rows,
                            private_accesses=body.count('addrspace 5'),
                            calls=len(re.findall(r'\bCALL\w*', body))))
    if not results: raise RuntimeError('no final machine functions found')
    failures = [dict(function=r['function'], ring=v) for r in results for v in r['rings']
                if v['generic_loads'] or v['wait_all']]
    result = dict(functions=results, failures=failures,
                  scope='natural CFG loops (dominating back edges), excluding CTA barrier paths')
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(functions=len(results), rings=sum(len(r['rings']) for r in results),
                         failures=len(failures))))
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
