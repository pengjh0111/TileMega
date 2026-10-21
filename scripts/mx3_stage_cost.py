#!/usr/bin/env python3
"""MX3-1: untraced cost of each class of stage, from TILEMEGA_DIAG_* plans.

Every plan under <root>/<variant>/<model>/B<b> is the same pinned selection
rebuilt with one diagnostic mask (ModelHarness.cuh, ServingGemmTaskBody.h):
  drop<K>   stages of TaskKind K left out, tasks and barrier;
  empty<K>  stages of kind K keep their barrier and run no task;
  nobody<K> tasks decode their descriptors and stop before any data;
  emptyAll  every stage keeps its barrier, no task runs (the skeleton).
`set` names the header state the plans were built from. One row per plan:
the median TPOT over its rounds and the difference to its set's base.
"""
import argparse, csv, glob, json, statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
SETS = {  # set -> (base variant, members, headers)
    'pre-cache': ('base', ['dropC', 'emptyC', 'dropM', 'emptyM', 'emptyA', 'emptyG', 'emptyAll'],
                  'f4da45a9a+trace (exact-wait mainloop)'),
    'decode': ('base2', ['emptyG2', 'nobodyG', 'emptyC2', 'nobodyC', 'emptyAll2', 'nowatch'],
               '6023bd3d2+epilogue global loads (guarded mainloop)'),
    'final': ('base3', ['emptyAll3', 'emptyC3', 'dropC3'], '26632fa26 bodies (descriptor prefetch)'),
}
rows = []
for name, (base, members, headers) in SETS.items():
    for cell in sorted({p.parent.name + '/' + p.name for p in (a.root / base).glob('*/B*')}):
        def tpot(v):
            x = [json.load(open(f))['tpot_p50_seconds'] * 1e3
                 for f in glob.glob(str(a.root / v / cell / 'measure/round*/measurements.json'))]
            return statistics.median(x) if x else None
        b = tpot(base)
        for v in [base] + members:
            t = tpot(v)
            if t is None:
                continue
            rows.append(dict(set=name, headers=headers, cell=cell, plan=v, tpot_ms=round(t, 4),
                             delta_ms=round(t - b, 4) if b else ''))
with a.out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r['set'], r['cell'], r['plan'], r['tpot_ms'], r['delta_ms'])
