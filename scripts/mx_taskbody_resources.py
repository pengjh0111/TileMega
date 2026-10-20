#!/usr/bin/env python3
"""Per-kernel register, spill and shared-memory use from the device compiler.

MX2-E item 5: MX-1 said resources were "read off the resource report" without
quoting a number. `cucc -Xcompiler -resource-usage` names the function each
report belongs to, so the megakernel entry points and the per-body probe
kernels can be reported separately. A megakernel inlines every TaskBody it
needs into one entry point, so these are per-kernel figures, not per-body:
the per-body numbers are the probe compilations under plan.so.resources,
which the janitor prunes once a cell is solved.
"""
import argparse, csv, re
from pathlib import Path

# Repository root, so a clone anywhere works (MX3-0 item 1).
REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--tm', type=Path, default=REPO / 'runs/mx2/tm')
ap.add_argument('--out', type=Path)
a = ap.parse_args()

REPORT = re.compile(
    r'Function properties for\s+(\S+)\s*\n\s*(\d+) bytes stack frame\s*\n'
    r'\s*maca info\s*:\s*Used\s+(\d+) MTregisters,\s+(\d+) STregisters,'
    r'\s+(\d+) bytes shared mem')


def kernel_name(mangled: str) -> str:
    for tag, label in (('tilemega_l1_loop_kernel', 'l1_loop'),
                       ('tilemega_l1_kernel', 'l1'), ('tilemega_l2_kernel', 'l2'),
                       ('tilemega_stage_kernel', 'stage'),
                       ('tilemega_wait_prof', 'wait_prof')):
        if tag in mangled:
            return label
    return mangled[:40]


rows = []
for log in sorted(a.tm.glob('*/B*/*/plan.so.ptxas.log')):
    cell = log.parent
    if 'rc=0' not in (cell / 'compile.log').read_text(errors='ignore'):
        continue
    seen = {}
    for mangled, stack, mt, st, smem in REPORT.findall(log.read_text(errors='ignore')):
        name = kernel_name(mangled)
        # One log can hold several passes over the same kernel; keep the worst.
        prev = seen.get(name)
        entry = (int(mt), int(st), int(stack), int(smem))
        if prev is None or entry > prev:
            seen[name] = entry
    for name, (mt, st, stack, smem) in sorted(seen.items()):
        rows.append(dict(model=cell.parts[-3], batch=cell.parts[-2], phase=cell.parts[-1],
                         kernel=name, mtregisters=mt, stregisters=st,
                         stack_frame_bytes=stack, shared_bytes=smem))

if not rows:
    raise SystemExit('no resource reports found')
print('%-13s %-4s %-8s %-8s %6s %6s %7s %7s' % (
    'model', 'B', 'phase', 'kernel', 'MTreg', 'STreg', 'stack', 'smem'))
for r in rows[:24]:
    print('%-13s %-4s %-8s %-8s %6d %6d %7d %7d' % (
        r['model'], r['batch'], r['phase'], r['kernel'],
        r['mtregisters'], r['stregisters'], r['stack_frame_bytes'], r['shared_bytes']))
mt = {r['mtregisters'] for r in rows}
st = [r['stregisters'] for r in rows]
stack = {r['stack_frame_bytes'] for r in rows}
print('\n%d kernel reports over %d cells' % (rows.__len__(), len({(r['model'], r['batch'], r['phase']) for r in rows})))
print('MTregisters: %s | STregisters %d-%d | stack frame: %s bytes' % (
    sorted(mt), min(st), max(st), sorted(stack)))
if a.out:
    with a.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('-> %s' % a.out)
