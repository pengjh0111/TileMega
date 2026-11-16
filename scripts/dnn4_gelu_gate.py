#!/usr/bin/env python3
"""Measure DT-GELU precision and the132-CTA resident epilogue on reserved GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path);p.add_argument('--execute',choices=('accuracy','epilogue'))
    p.add_argument('--source',type=Path,default=ROOT/'scripts/dnn4_probe_gelu.cu')
    p.add_argument('--cpu-set',default='136-143')
    p.add_argument('--cal-cpu-set',default='32-39')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    if a.execute:
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!=('0' if a.execute=='accuracy' else '1'):raise ValueError('wrong GPU reservation')
        b=json.loads(a.build.read_text());rows=[]
        if sha(b['binary'])!=b['sha256']:raise ValueError('binary changed')
        sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
        for i in range(5):
            start=gpu(os.environ['TILEMEGA_DEVICE_INDEX'])
            r=subprocess.run([b['binary'],a.execute],capture_output=True,text=True,timeout=120)
            end=gpu(os.environ['TILEMEGA_DEVICE_INDEX']);(a.out/f'p{i}.stdout').write_text(r.stdout);(a.out/f'p{i}.stderr').write_text(r.stderr)
            if r.returncode:raise RuntimeError('GELU probe failed: '+r.stderr+' '+r.stdout)
            rows.append(dict(measurement=json.loads(r.stdout),start_sm_clock_mhz=start['sm_clock_mhz'],end_sm_clock_mhz=end['sm_clock_mhz']))
        if any(sha(f)!=value for f,value in b['inputs'].items()):raise ValueError('frozen inputs changed')
        metrics={k:statistics.median(r['measurement'][k] for r in rows) for k in rows[0]['measurement']}
        save(a.out/'result.json',dict(status='passed',evidence='verified',mode=a.execute,processes=rows,median=metrics,passes='5/5',build=str(a.build),sass_sha256=b['sass_sha256']))
        return
    src=a.out/'source';src.mkdir();shutil.copytree(ROOT/'include',src/'include');shutil.copyfile(a.source,src/'probe.cu')
    frozen={str(f):sha(f) for f in src.rglob('*') if f.is_file()};binary=a.out/'probe';log=a.out/'ptxas.log'
    command=['taskset','-c',a.cpu_set,'nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(src/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),str(src/'probe.cu'),'-o',str(binary)]
    with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=600)
    text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
    if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('compile/spill gate failed: '+str(log))
    sass=a.out/'probe.sass'
    with sass.open('w') as f:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=f,check=True)
    build=a.out/'build.json';save(build,dict(binary=str(binary),sha256=sha(binary),sass_sha256=sha(sass),inputs=frozen,command=command))
    jobs=[];queue=a.out/'queue';queue.mkdir()
    for mode,device in [('accuracy',0),('epilogue',1)]:
        env=dict(CUDA_VISIBLE_DEVICES=str(device),TILEMEGA_DEVICE_INDEX=str(device),TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{device}.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
        jobs.append(dict(name='dnn4-gelu-'+mode,gpu=True,timing=mode=='epilogue',timeout_s=1800,needs_free_mib=8192,cwd=str(ROOT),env=env,command=['taskset','-c','120-127' if device==0 else a.cal_cpu_set,str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute',mode,'--build',str(build),'--out','{attempt_dir}/probe']))
    save(queue/'queue_gelu.json',jobs)
    with (a.out/'dispatch.log').open('w') as f:subprocess.run([str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(queue),'--out',str(a.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6'],env=dict(os.environ,PYTHONPATH=str(ROOT/'python')),stdout=f,stderr=subprocess.STDOUT,check=True)
    state=json.loads((a.out/'scheduler/state.json').read_text());print(json.dumps(state))
    if any(r['status']!='done' for r in state.values()):raise RuntimeError('GELU cohort incomplete')
if __name__=='__main__':main()
