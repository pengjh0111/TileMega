#!/usr/bin/env python3
"""Measure unchanged frozen candidate Plans with each W=1 protocol candidate."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
RAW=HERE/'target_positions'
sys.path.insert(0,str(REPO/'docs/experiments/FENCE'))
import run as fence
CONFIGS={'baseline':[], 'c1':['-DTILEMEGA_RELEASE_AFTER_BARRIER=1'],
         'c2':['-DTILEMEGA_RELEASE_AFTER_BARRIER=1','-DTILEMEGA_ASYNC_PUBLISH=1']}


def main():
    global RAW
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('build','measure','summarize'))
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--raw',type=Path,default=RAW)
    args=p.parse_args()
    RAW=args.raw.resolve()
    if any(k.startswith('TILEMEGA_') for k in os.environ):
        raise ValueError('inherited TILEMEGA override')
    rows=list(csv.DictReader((HERE/'targets.tsv').open(),delimiter='\t'))
    selected=[r for r in rows if r['candidate'] not in ('legacy_grid_stride','rotate')]
    RAW.mkdir(parents=True,exist_ok=True)
    if args.phase=='build':
        free=shutil.disk_usage(RAW).free//2**20
        print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192: raise ValueError('insufficient disk')
        work=[]
        for config,flags in CONFIGS.items():
            raw=RAW/config
            for folder in ('bin','log'): (raw/folder).mkdir(parents=True,exist_ok=True)
            for r in selected:
                key=f'{r["model"]}_s{r["seq"]}_{r["candidate"]}'
                placement=4 if r['candidate']=='balanced' else 0
                if not (raw/'log'/f'{key}_p{placement}_full.build.json').exists():
                    work.append((key,raw,flags,REPO/r['source'],placement))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda x:fence.build((x[0],x[4],'full'),x[1],x[2],source_override=x[3]),work))
        return
    if args.phase=='measure':
        session=str(time.time_ns())
        for model in ('gqa2','mha4'):
            for seq in (4,128):
                arms=[(c,r) for c in CONFIGS for r in selected if r['model']==model and int(r['seq'])==seq]
                for i in range(25):
                    for order in range(len(arms)):
                        config,r=arms[(i+order)%len(arms)]
                        key=f'{model}_s{seq}_{r["candidate"]}'
                        placement=4 if r['candidate']=='balanced' else 0
                        binary=RAW/config/'bin'/f'{key}_p{placement}_full'
                        folder=RAW/config/'paired'/key
                        folder.mkdir(parents=True,exist_ok=True)
                        log=folder/f'r{i}.log'
                        if log.exists(): raise ValueError(f'refusing overwrite {log}')
                        cmd=[str(binary),str(REPO/f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3')]
                        result=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
                        output=result.stdout+result.stderr
                        log.write_text(output)
                        log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,exit_code=result.returncode,
                            session=session,round=i,order=order,time_ns=time.time_ns(),binary_sha256=fence.sha(binary)))+'\n')
                        if result.returncode or re.findall(r'^RESULT status=(\S+)',output,re.M)!=['PASS']:
                            raise ValueError(f'{log}: correctness failed')
                    print(f'TARGET_POSITION {model} s{seq} round={i+1}/25',flush=True)
        return
    import statistics
    output=[]
    for r in rows:
        candidates={}
        for config in CONFIGS:
            if r['candidate'] in ('legacy_grid_stride','rotate'):
                placement=0 if r['candidate']=='legacy_grid_stride' else 5
                folder=HERE/'ablation'/config/'paired'/f'{r["model"]}_s{r["seq"]}_p{placement}'/'full'
            else:
                folder=RAW/config/'paired'/f'{r["model"]}_s{r["seq"]}_{r["candidate"]}'
            values=[fence.timing(folder/f'r{i}.log')['l2_ms'] for i in range(25)]
            candidates[config]=(statistics.median(values),folder)
        best=min(candidates,key=lambda k:candidates[k][0])
        value,folder=candidates[best]
        output.append(dict(candidate=r['candidate'],model=r['model'],seq=r['seq'],configuration=best,
            floor_ms=r['floor_ms'],target_ms=r['target_ms'],l2_ms=value,
            delta_target_ms=value-float(r['target_ms']),evidence=str(folder.relative_to(REPO))))
    with (RAW/'positions.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(output)


if __name__=='__main__':main()
