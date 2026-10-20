#!/usr/bin/env python3
"""Record, per solved MX-2 cell, the implementation the solver selected.

MX2-E item 4: the MX-1 CSV left the `note` column empty, so a reader could not
tell which executor, attention body, GEMM implementations, weight layout or
tile shapes produced a number. Everything here is read back out of the cell's
own artifacts rather than from the command line that launched it.
"""
import argparse, csv, json, re
from pathlib import Path

# Repository root, so a clone anywhere works (MX3-0 item 1).
REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--tm', type=Path, default=REPO / 'runs/mx2/tm')
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()


def build_macros(command: Path) -> dict:
    """Macros passed as -D on the device compile line.

    Feature switches such as TILEMEGA_NONPAGED_TILED never appear in the
    generated .cu; they reach the compiler as -D flags, so reading only the
    source reports every plan as row-major.
    """
    out = {}
    if not command.exists():
        return out
    for name, value in re.findall(r'-D(TILEMEGA_[A-Z0-9_]+)=(-?\d+)',
                                  command.read_text(errors='ignore')):
        out[name] = int(value)
    return out


def defines(cu: Path) -> dict:
    """The tile macros the generator wrote for this plan."""
    out = {}
    if not cu.exists():
        return out
    for line in cu.open():
        if not line.startswith('#define TILEMEGA_'):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[2].lstrip('-').isdigit():
            out[parts[1]] = int(parts[2])
        if len(out) > 400:
            break
    return out


rows = []
for compile_log in sorted(a.tm.glob('*/B*/*/compile.log')):
    cell = compile_log.parent
    text = compile_log.read_text(errors='ignore')
    if 'rc=0' not in text:
        continue
    model, batch, phase = cell.parts[-3], cell.parts[-2], cell.parts[-1]
    macros = defines(cell / 'plan.so.cu')
    macros.update(build_macros(cell / 'plan.so.build_command.txt'))
    variants = macros.get('TILEMEGA_GEMM_VARIANT_COUNT', 0)
    # A GEMV variant forces row-major weights, so the layout is a consequence
    # of the selection rather than a free choice.
    gemv = sorted(k for k, v in macros.items()
                  if re.fullmatch(r'TILEMEGA_GEMM_IMPL_\d+', k) and v)
    layout = 'tiled' if macros.get('TILEMEGA_NONPAGED_TILED') else 'row'
    tiles = []
    for index in [''] + ['_V%d' % i for i in range(1, variants)]:
        shape = tuple(macros.get('TILEMEGA_GEMM%s_TILE_%s' % (index, axis))
                      for axis in 'MNK')
        stages = macros.get('TILEMEGA_GEMM%s_STAGES' % index)
        if all(v is not None for v in shape):
            tiles.append('%dx%dx%ds%s' % (*shape, stages))
    identity = {}
    source = cell / 'plan.so.plan.json'
    if source.exists():
        try:
            identity = json.loads(source.read_text())
        except ValueError:
            identity = {}
    rows.append(dict(
        model=model, batch=int(batch.lstrip('B')), phase=phase,
        layout=layout, variants=variants,
        gemv_classes=','.join(k.rsplit('_', 1)[-1] for k in gemv) or 'none',
        tiles=' '.join(tiles),
        attention_impl=macros.get('TILEMEGA_ATTENTION_IMPL'),
        last_arriver=macros.get('TILEMEGA_NONPAGED_LA'),
        sync_v3=macros.get('TILEMEGA_SYNC_V3'),
        executor=identity.get('executor', 'L1/L2 per measurement'),
    ))

with a.out.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                            ['model', 'batch', 'phase', 'layout'])
    writer.writeheader()
    writer.writerows(rows)
print('%d solved cells -> %s' % (len(rows), a.out))
for row in rows:
    print('  %-14s B%-3d %-8s %-5s vars=%-2s gemv=%-5s %s' % (
        row['model'], row['batch'], row['phase'], row['layout'],
        row['variants'], row['gemv_classes'], row['tiles'][:46]))
