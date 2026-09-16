#!/usr/bin/env python3
"""Build phase/off arms, validate, and measure fresh rotated process pairs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def source(model):
    return REPO/(f'docs/experiments/PLAN_CONTRACT/legacy_identity/plan/{model}.cu' if model!='real'
                else 'docs/experiments/REALMODEL/raw/work/r2sim_s4/model.cu')

def fixture(model,seq):
    return REPO/(f'docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3' if model!='real'
                else f'docs/experiments/REALMODEL/raw/work/r2sim_s{seq}/export/fixture')

def build(raw,model,p,arm,arch):
    name=f'{model}_p{p}_{arm}'
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2',f'-arch={arch}','-lineinfo',
         '-DTILEMEGA_EVENT_KAPPA=1',f'-DTILEMEGA_PLACEMENT={p}',
         f'-DTILEMEGA_TRACE_PHASE={int(arm=="phase")}',
         *['-I'+str(REPO/x) for x in ('include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test')],
         str(source(model)),str(REPO/'build-portable/libtilemega.a'),'-L/usr/local/cuda/lib64','-lcudart','-o',str(raw/'bin'/name)]
    with (raw/'build'/f'{name}.log').open('w') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    meta=dict(command=cmd,exit_code=result.returncode,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              source=str(source(model)),source_sha256=sha(source(model)),time_ns=time.time_ns())
    if not result.returncode: meta['binary_sha256']=sha(raw/'bin'/name)
    (raw/'build'/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
    result.check_returncode()
    print('BUILT',name,flush=True)

def run(raw,model,seq,p,arm,folder,session,round_,order,dump=False):
    binary=raw/'bin'/f'{model}_p{p}_{arm}'
    folder.mkdir(parents=True,exist_ok=True)
    log=folder/f'r{round_}.log'
    if log.exists(): raise ValueError(f'refusing overwrite {log}')
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    if 'correctness' in folder.parts:
        env.update(TILEMEGA_WARMUP='0', TILEMEGA_REPEAT='1')
    if arm=='phase': env.update(TILEMEGA_TRACE_PHASE='1',TILEMEGA_MODEL_NAME=model)
    if dump: env['TILEMEGA_TRACE_PHASE_OUT']=str(folder/'dump')
    cmd=[str(binary),str(fixture(model,seq))]
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
    log.write_text(result.stdout+result.stderr)
    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('TILEMEGA_')},
        exit_code=result.returncode,session=session,round=round_,order=order,time_ns=time.time_ns(),binary_sha256=sha(binary)))+'\n')
    if result.returncode or 'RESULT status=PASS' not in log.read_text(): raise RuntimeError(f'correctness: {log}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','pilot','measure','correctness','dump'])
    ap.add_argument('--raw',type=Path,default=HERE/'raw');ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--arch',default='sm_89');ap.add_argument('--models',nargs='+',default=None)
    a=ap.parse_args();raw=a.raw.resolve()
    if a.models is None: a.models=['gqa2','mha4'] if a.action=='correctness' else ['gqa2','mha4','real']
    for d in ('bin','build'): (raw/d).mkdir(parents=True,exist_ok=True)
    session=str(time.time_ns())
    if a.action=='build':
        free=shutil.disk_usage(raw).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192: raise RuntimeError('disk budget')
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures=[pool.submit(build,raw,m,p,arm,a.arch) for m in a.models for p in (0,5) for arm in ('off','phase')]
            for f in futures: f.result()
        return
    for m in a.models:
        for seq in (4,128):
            if a.action=='correctness':
                for i in range(50): run(raw,m,seq,5,'phase',raw/'correctness'/f'{m}_s{seq}',session,i,0)
            elif a.action=='dump':
                for p in (0,5): run(raw,m,seq,p,'phase',raw/'trace'/f'{m}_s{seq}_p{p}',session,0,0,True)
            else:
                for p in (0,5):
                    for i in range(3 if a.action=='pilot' else 25):
                        for j in range(2):
                            arm=('off','phase')[(i+j)%2]
                            run(raw,m,seq,p,arm,raw/a.action/f'{m}_s{seq}_p{p}'/arm,session,i,j)
            print(a.action,m,seq,flush=True)

if __name__=='__main__': main()
