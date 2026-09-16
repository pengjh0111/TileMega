#!/usr/bin/env python3
"""Rebuild only unsafe window probe arms after closing the look-ahead hole."""
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'docs/experiments/FENCE'))
import run as fence


def main():
    out=HERE/'window_probe_fix'
    free=shutil.disk_usage(out).free//2**20
    print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
    if free<8192:raise ValueError('insufficient disk')
    configs=(('window2',2,False),('local2',2,True),('window4',4,False),('local4',4,True))
    work=[];old=[]
    for name,w,local in configs:
        flags=['-DTILEMEGA_RELEASE_AFTER_BARRIER=1','-DTILEMEGA_ASYNC_PUBLISH=1']
        if local:flags+=['-DTILEMEGA_LOCAL_DEP_SMEM=1']
        for m in ('gqa2','mha4'):
            for p in (0,5):
                for arm in ('nowait','neither'):
                    raw=HERE/name;key=f'{m}_p{p}_{arm}'
                    meta=json.loads((raw/'log'/f'{key}.build.json').read_text())
                    old.append(dict(configuration=name,key=key,build=meta))
                    work.append(((m,p,arm),raw,flags,w,None))
    for name,local in (('window2',False),('local2',True)):
        flags=['-DTILEMEGA_RELEASE_AFTER_BARRIER=1','-DTILEMEGA_ASYNC_PUBLISH=1']
        if local:flags+=['-DTILEMEGA_LOCAL_DEP_SMEM=1']
        raw=HERE/'realwidth'/name
        for arm in ('nowait','neither'):
            key=f'real_p0_{arm}'
            old.append(dict(configuration='realwidth/'+name,key=key,
                build=json.loads((raw/'log'/f'{key}.build.json').read_text())))
            work.append((('real',0,arm),raw,flags,2,REPO/'docs/experiments/REALMODEL/raw/work/r2sim_s4/model.cu'))
    (out/'before_builds.json').write_text(json.dumps(old,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda x:fence.build(x[0],x[1],x[2],window=x[3],source_override=x[4]),work))
    (out/'build_status.txt').write_text('PASS\n')


if __name__=='__main__':main()
