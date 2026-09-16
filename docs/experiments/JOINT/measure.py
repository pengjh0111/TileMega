#!/usr/bin/env python3
"""Compile top-three, freeze a pilot choice, then collect fresh rotated pairs."""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from search import REPO,HERE,table,source

PROTOCOL=['BARRIER_V2=1','EVENT_SOLO=1','EVENT_RED_PUBLISH=1','WAIT_POLICY=1',
 'WAIT_SPIN_ITERS=64','WAIT_BACKOFF_NS=64','WAIT_BACKOFF_GROW=1','WAIT_BACKOFF_CAP_NS=64','SLOT_WINDOW=1']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(m,s,past=3):
    if os.getenv('R5_INPUT_ROOT'):return Path(os.environ['R5_INPUT_ROOT'])/'fixture'/f'{m}_s{s}_p{past}'
    return REPO/(f'docs/experiments/SEQSCAN/raw/fixture/{m}_s{s}_p{past}' if m!='real' else f'docs/experiments/REALMODEL/raw/work/r2sim_s{s}/export/fixture')
def timing(p):
    line=re.findall(r'^E2E_TIME (.*)$',p.read_text(),re.M);assert len(line)==1
    return {k:float(v) for k,v in re.findall(r'(\w+)=([\d.eE+-]+)',line[0])}
def specs(cell,m):
    rows=table(cell/'top3.tsv')
    result={'control':dict(source=str(source(m)),kappa='1',residency='0',placement_macro='5')}
    for i,r in enumerate(rows):result[f'top{i+1}']=dict(r,placement_macro='0')
    return result
def build(cell,m,arm,spec,arch,trace=False,phase=False):
    d=cell/'build';b=cell/'bin';d.mkdir(exist_ok=True);b.mkdir(exist_ok=True)
    name=arm+('_trace' if trace else '_phase' if phase else '');binary=b/name
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2',f'-arch={arch}','-lineinfo','-Xptxas=-v',
         *['-DTILEMEGA_'+x for x in PROTOCOL+spec.get('extra',[])],f"-DTILEMEGA_EVENT_KAPPA={spec['kappa']}",
         f"-DTILEMEGA_RESIDENCY_CAP={spec['residency']}",f"-DTILEMEGA_PLACEMENT={spec['placement_macro']}",
         f'-DTILEMEGA_TRACE_V2={int(trace)}',f'-DTILEMEGA_TRACE_PHASE={int(phase)}',
         *['-I'+str(REPO/x) for x in ('include','third_party/cutlass/include','third_party/cutlass/tools/util/include','third_party/cutlass/test')],
         str(Path(spec['source']).resolve()),str(REPO/'build-portable/libtilemega.a'),'-L/usr/local/cuda/lib64','-lcudart','-o',str(binary.resolve())]
    begin=time.time_ns()
    with (d/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    meta=dict(command=cmd,exit_code=r.returncode,source_sha256=sha(Path(spec['source'])),started_ns=begin,elapsed_ns=time.time_ns()-begin,
              head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    if r.returncode==0:meta['binary_sha256']=sha(binary)
    (d/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print('BUILD',cell.name,name,r.returncode,flush=True)
    return r.returncode
def run(cell,m,seq,arm,folder,round_,order,session,dump=False,past=3,phase=False):
    folder.mkdir(parents=True,exist_ok=True);log=folder/f'r{round_}.log'
    if log.exists():raise RuntimeError('refusing overwrite '+str(log))
    binary=cell/'bin'/arm
    env={k:v for k,v in os.environ.items() if not k.startswith('TILEMEGA_')}
    if 'correctness' in folder.parts or 'seqscan' in folder.parts:env.update(TILEMEGA_WARMUP='0',TILEMEGA_REPEAT='1')
    if dump:env.update(TILEMEGA_TRACE_V2='1',TILEMEGA_TRACE_V2_OUT=str((folder/'dump').resolve()),TILEMEGA_MODEL_NAME=m,TILEMEGA_PLACEMENT_BASE_DUMP='1',TILEMEGA_GLOBALTIMER_NS=os.getenv('PHASE_TICK_NS','1024'))
    if phase:env.update(TILEMEGA_TRACE_PHASE='1',TILEMEGA_TRACE_PHASE_OUT=str((folder/'dump').resolve()),TILEMEGA_MODEL_NAME=m,TILEMEGA_GLOBALTIMER_NS=os.getenv('PHASE_TICK_NS','1024'))
    cmd=[str(binary.resolve()),str(fixture(m,seq,past))];start=time.time_ns()
    try:
        with open('/tmp/tilemega-r5-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);start=time.time_ns()
            r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300);status=r.returncode;text=r.stdout+r.stderr
    except subprocess.TimeoutExpired as e:status=124;text=str(e)
    log.write_text(text)
    log.with_suffix('.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('TILEMEGA_')},
        exit_code=status,session=session,round=round_,order=order,started_ns=start,elapsed_ns=time.time_ns()-start,binary_sha256=sha(binary)))+'\n')
    if status or 'RESULT status=PASS' not in text:raise RuntimeError('correctness: '+str(log))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','pilot','freeze','measure','correctness','trace_build','trace','phase_build','phase'])
    ap.add_argument('--raw',type=Path,default=HERE/'raw');ap.add_argument('--models',nargs='+',default=['gqa2','mha4','real']);ap.add_argument('--seqs',nargs='+',type=int,default=[4,128]);ap.add_argument('--arch',default='sm_89');ap.add_argument('--jobs',type=int,default=2)
    a=ap.parse_args();session=str(time.time_ns())
    if a.action in ('build','trace_build','phase_build'):
        free=shutil.disk_usage(a.raw).free//2**20;print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
        if free<8192:raise RuntimeError('disk budget')
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            jobs=[]
            for m in a.models:
                for seq in a.seqs:
                    cell=a.raw/f'{m}_s{seq}';s=specs(cell,m)
                    if a.action in ('trace_build','phase_build'):
                        choice=json.loads((cell/'choice.json').read_text())['arm'];s={k:s[k] for k in ((choice,'control') if a.action=='trace_build' else (choice,))}
                    jobs.extend(pool.submit(build,cell,m,k,v,a.arch,a.action=='trace_build',a.action=='phase_build') for k,v in s.items())
            failures=sum(j.result()!=0 for j in jobs)
        if failures:raise RuntimeError(f'{failures} builds failed; retain and exclude them explicitly before replacement')
        return
    for m in a.models:
        for seq in a.seqs:
            cell=a.raw/f'{m}_s{seq}';s=specs(cell,m)
            if a.action=='freeze':
                means={arm:statistics.median(timing(cell/'pilot'/arm/f'r{i}.log')['l2_ms'] for i in range(5)) for arm in ('top1','top2','top3')}
                choice=min(means,key=means.get);p=cell/'choice.json'
                if p.exists():raise RuntimeError('choice is frozen')
                p.write_text(json.dumps(dict(arm=choice,pilot_medians=means,choice=s[choice],frozen_ns=time.time_ns()),indent=2)+'\n');continue
            if a.action in ('pilot','measure'):
                arms=['control','top1','top2','top3']
                if a.action=='measure' and not (cell/'choice.json').exists():raise RuntimeError('freeze choice before confirmatory measurement')
                for i in range(5 if a.action=='pilot' else 25):
                    for j in range(len(arms)):
                        arm=arms[(i+j)%len(arms)];run(cell,m,seq,arm,cell/a.action/arm,i,j,session)
            elif a.action=='correctness':
                choice=json.loads((cell/'choice.json').read_text())['arm']
                for i in range(50):run(cell,m,seq,choice,cell/'correctness',i,0,session)
            elif a.action=='phase':
                choice=json.loads((cell/'choice.json').read_text())['arm']
                run(cell,m,seq,choice+'_phase',cell/'phase'/choice,0,0,session,phase=True)
            elif a.action=='trace':
                choice=json.loads((cell/'choice.json').read_text())['arm']
                for j,arm in enumerate(('control',choice)):run(cell,m,seq,arm+'_trace',cell/'trace'/arm,0,j,session,True)
            print(a.action,m,seq,flush=True)
if __name__=='__main__':main()
