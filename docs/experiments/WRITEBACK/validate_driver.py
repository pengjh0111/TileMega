#!/usr/bin/env python3
"""Validate compiler-produced source; retain every fresh process and command."""
import argparse,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'docs/experiments/JOINT'))
import measure

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','correctness']);p.add_argument('--model',default='gqa2');p.add_argument('--seq',type=int,default=4);p.add_argument('--past',type=int,default=3);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--processes',type=int,default=50)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='build':
        free=shutil.disk_usage(a.out).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        source=a.source.read_text()
        import re
        kap=re.search(r'^#define TILEMEGA_EVENT_KAPPA (\d+)$',source,re.M)
        res=re.search(r'^#define TILEMEGA_RESIDENCY_CAP (\d+)$',source,re.M)
        spec=dict(source=str(a.source.resolve()),kappa=kap.group(1),residency=res.group(1),placement_macro='0')
        (a.out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        if measure.build(a.out,a.model,'solved',spec,'sm_89'):raise RuntimeError('compile failed')
    else:
        session=str(time.time_ns())
        for i in range(a.processes):
            measure.run(a.out,a.model,a.seq,'solved',a.out/'correctness',i,0,session,past=a.past)
        print(f'CORRECTNESS model={a.model} seq={a.seq} pass={a.processes}/{a.processes}',flush=True)
if __name__=='__main__':main()
