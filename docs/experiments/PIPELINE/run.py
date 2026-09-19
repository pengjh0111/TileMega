#!/usr/bin/env python3
"""B1-a/c/e: build, validate and time the prefetch phase on FORK6's cells.

Three arms per cell, identical in every macro but the mechanism:

  * `control`   -- FORK6's `selected` configuration verbatim;
  * `prefetch`  -- the same, plus the §5.3.1 Prefetch phase on paged shared
                   memory, issued for slot+1 ahead of slot's body;
  * `inline`    -- the same page and the same copy, issued for slot and waited
                   on at once.  It is the storage-matched control: it pays the
                   shared memory and the instructions but overlaps nothing, so
                   `prefetch` vs `inline` isolates the overlap while `inline`
                   vs `control` prices the mechanism itself.

Cells and configurations are FORK6's, read back from
`COSTMODEL/raw_kloop/*/specs.json`, so B1-e stays comparable with R6 rather
than being a second, differently scoped number.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure

FORK6=REPO/'docs/experiments/COSTMODEL/raw_kloop'
ARMS=('control','prefetch','inline')
# The page has to hold one RMSNorm scale row, or `PrefetchBytes` refuses the
# copy and the arm runs the mechanism with nothing in it: the reference models'
# hidden 512 fills the 1024-byte default exactly, the real model's 4096 needs
# 8192.  Measured, not assumed -- `E2E_PREFETCH` counts the slots that issue,
# and at 1024 the real cells report `declared=32 issued=0`.
PAGE={'gqa2':1024,'mha4':1024,'real':8192}

def specs(model,seq,phase=False):
    """FORK6's frozen selection for this cell, once per arm.

    FORK6 ran its cells with `TRACE_KLOOP=1`, which `ModelRuntime.h:354` makes
    conditional on `TRACE_PHASE`; the timed arms therefore drop it and only the
    phase build carries it back, which is also the instrumentation B1-c reads.
    """
    sel=json.loads((FORK6/f'{model}_s{seq}'/'specs.json').read_text())['selected']
    # FORK6's own sources predate the frontier field, so they cannot carry it.
    # `regen.py` re-runs FORK6's recorded projection command at HEAD; strip the
    # guarded field back out and the result is byte-identical to FORK6's file,
    # which is what keeps the cells comparable (`summary.md`, "regeneration").
    sel=dict(sel,source=str(HERE/'raw'/f'{model}_s{seq}'/'src'/
                            (Path(sel['source']).name)))
    extra=[x for x in sel.get('extra',[]) if not x.startswith('TRACE_')]
    if phase:extra=extra+['TRACE_KLOOP=1']
    # Left off at the default so the reference cells' recorded commands, and
    # hence their binaries, are the ones already measured.
    page=[] if PAGE[model]==1024 else [f'PREFETCH_PAGE_BYTES={PAGE[model]}']
    return {'control':dict(sel,extra=extra),
            'prefetch':dict(sel,extra=extra+['PREFETCH_RUNTIME=1']+page),
            'inline':dict(sel,extra=extra+['PREFETCH_RUNTIME=1','PREFETCH_INLINE=1']+page)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['build','phase_build','correctness','e2e','phase'])
    ap.add_argument('--raw',type=Path,default=HERE/'raw')
    ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real'])
    ap.add_argument('--seqs',nargs='+',type=int,default=[4,128])
    ap.add_argument('--rounds',type=int,default=50)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--arch',default='sm_89')
    # Output folder suffix. The real cells' first correctness round ran at the
    # default page, which their scale row does not fit; it is kept under the
    # unsuffixed name and the re-run at 8192 lands beside it.
    ap.add_argument('--tag',default='')
    a=ap.parse_args();a.raw.mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns());cells=[]
    for m in a.models:
        for s in a.seqs:
            cell=a.raw/f'{m}_s{s}';cell.mkdir(parents=True,exist_ok=True)
            phase=a.action in ('phase','phase_build')
            ss=specs(m,s,phase);cells.append((m,s,cell,ss))
            name='specs_phase.json' if phase else 'specs.json'
            (cell/name).write_text(json.dumps(ss,indent=2)+'\n')
    if a.action in ('build','phase_build'):
        free=shutil.disk_usage(a.raw).free//2**20
        print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        phase=a.action=='phase_build'
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[pool.submit(measure.build,c,m,arm,spec,a.arch,False,phase)
                  for m,s,c,ss in cells for arm,spec in ss.items()]
            if sum(j.result()!=0 for j in jobs):raise RuntimeError('build')
        return
    for m,s,cell,ss in cells:
        if a.action=='correctness':
            # The gate is on the mechanism; `control` is FORK6's own arm and is
            # re-proved by its own round rather than here.
            for arm in ('prefetch','inline'):
                for i in range(a.rounds):
                    measure.run(cell,m,s,arm,cell/('correctness'+a.tag)/arm,i,0,session)
        elif a.action=='e2e':
            # Rotated, so drift and launch order cannot settle on one arm.
            for i in range(a.rounds):
                for j in range(len(ARMS)):
                    arm=ARMS[(i+j)%len(ARMS)]
                    measure.run(cell,m,s,arm,cell/('e2e'+a.tag)/arm,i,j,session)
        else:
            # One dump per round, as PHASE2 keeps them: the trace is the
            # evidence, and a shared folder would leave only the last round's.
            for arm in ARMS:
                for i in range(a.rounds):
                    measure.run(cell,m,s,arm+'_phase',cell/('phase'+a.tag)/f'{arm}_r{i}',
                                i,0,session,phase=True)
        print(a.action,cell.name,flush=True)

if __name__=='__main__':main()
