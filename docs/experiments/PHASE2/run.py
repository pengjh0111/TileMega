#!/usr/bin/env python3
"""B0 SIMT-side exposed wait: build, validate and dump the FORK6 cells again.

The cells, configurations and sources are FORK6's `selected` arm verbatim, read
back from `COSTMODEL/raw_kloop/*/specs.json`. Only the instrumentation widens:
`TILEMEGA_TRACE_SIMT=1` times the barriers the SIMT bodies already execute, so
a wait share can be quoted against the whole critical path rather than against
the GEMM mainloops alone. Keeping the cells fixed is what makes FORK7
comparable to FORK6 instead of a second, differently scoped number.
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

def specs(model,seq):
    """FORK6's frozen selection for this cell, re-instrumented for B0."""
    spec=dict(json.loads((FORK6/f'{model}_s{seq}'/'specs.json').read_text())['selected'])
    spec['extra']=['TRACE_KLOOP=1','TRACE_SIMT=1']
    return {'selected':spec}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['build','correctness','dump'])
    ap.add_argument('--raw',type=Path,default=HERE/'raw')
    ap.add_argument('--models',nargs='+',default=['gqa2','mha4'])
    ap.add_argument('--seqs',nargs='+',type=int,default=[4,128])
    ap.add_argument('--rounds',type=int,default=50,
                    help='correctness processes per cell; dumps per cell')
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--arch',default='sm_89')
    a=ap.parse_args();a.raw.mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns());cells=[]
    for m in a.models:
        for s in a.seqs:
            cell=a.raw/f'{m}_s{s}';cell.mkdir(parents=True,exist_ok=True)
            ss=specs(m,s);cells.append((m,s,cell,ss))
            (cell/'specs.json').write_text(json.dumps(ss,indent=2)+'\n')
    if a.action=='build':
        free=shutil.disk_usage(a.raw).free//2**20
        print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[pool.submit(measure.build,c,m,arm,spec,a.arch,False,True)
                  for m,s,c,ss in cells for arm,spec in ss.items()]
            if sum(f.result()!=0 for f in jobs):raise RuntimeError('build')
        return
    for m,s,cell,ss in cells:
        for arm in ss:
            if a.action=='correctness':
                for i in range(a.rounds):
                    measure.run(cell,m,s,arm+'_phase',cell/'correctness'/arm,i,0,session,phase=True)
            else:
                # One fresh process per dump. FORK6 took a single dump per
                # cell; the share here lands near its threshold, so each cell
                # gets a distribution rather than a point.
                for i in range(a.rounds):
                    measure.run(cell,m,s,arm+'_phase',cell/'phase'/f'{arm}_r{i}',
                                i,0,session,phase=True)
        print(a.action,cell.name,flush=True)

if __name__=='__main__':main()
