#!/usr/bin/env python3
"""R6 K-loop collection. Old phase columns are preserved in every dump.

build: selected R5 configurations plus reference K16/K32/K64 geometries.
correctness: 50 fresh processes in each of the six cells.
dump: one fresh traced process per geometry and cell; never overwrite logs.
Outputs raw_kloop/<model>_s<seq>/{build,bin,correctness,phase,specs.json}.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure

def specs(model,seq,root):
    old=root/f'{model}_s{seq}'
    choice=json.loads((old/'choice.json').read_text())['choice']
    result={'selected':dict(choice)}
    if model!='real':
        for row in measure.table(old/'top3.tsv'):
            if (row['m'],row['n'],row['split'])==('32','16','1'):
                result['k'+row['k']]=dict(row,placement_macro='0')
    for spec in result.values():spec['extra']=['TRACE_KLOOP=1']
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['build','correctness','dump'])
    ap.add_argument('--raw',type=Path,default=HERE/'raw_kloop')
    ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real'])
    ap.add_argument('--seqs',nargs='+',type=int,default=[4,128])
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--arch',default='sm_89')
    ap.add_argument('--calibration-sources',type=Path)
    ap.add_argument('--champion-root',type=Path,default=REPO/'docs/experiments/JOINT/raw')
    a=ap.parse_args();a.raw.mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns());cells=[]
    for m in a.models:
        for s in a.seqs:
            cell=a.raw/f'{m}_s{s}';cell.mkdir(parents=True,exist_ok=True)
            spec=specs(m,s,a.champion_root)
            if a.calibration_sources:spec.update(json.loads(a.calibration_sources.read_text())[m])
            cells.append((m,s,cell,spec))
            (cell/'specs.json').write_text(json.dumps(spec,indent=2)+'\n')
    if a.action=='build':
        free=shutil.disk_usage(a.raw).free//2**20
        print(f'DISK NEED_MIB=16384 FREE_MIB={free}',flush=True)
        if free<16384:raise RuntimeError('disk budget')
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[pool.submit(measure.build,c,m,arm,spec,a.arch,False,True)
                  for m,s,c,ss in cells for arm,spec in ss.items()]
            failures=sum(f.result()!=0 for f in jobs)
        if failures:raise RuntimeError(f'{failures} builds failed')
        return
    for m,s,c,ss in cells:
        if a.action=='correctness':
            for i in range(50):
                measure.run(c,m,s,'selected_phase',c/'correctness',i,0,session)
        else:
            for j,arm in enumerate(ss):
                measure.run(c,m,s,arm+'_phase',c/'phase'/arm,0,j,session,phase=True)
        print(a.action,m,s,flush=True)

if __name__=='__main__':main()
