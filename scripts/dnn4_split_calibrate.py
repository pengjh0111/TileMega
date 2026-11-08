#!/usr/bin/env python3
"""Build and guard the two132-CTA split-K acceptance probes and swap S32."""
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
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    if args.execute:
        if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or os.environ.get('TILEMEGA_DEVICE_INDEX')!='1':
            raise ValueError('split calibration requires reserved GPU1')
        sys.path.insert(0,str(ROOT/'scripts'));from dnn1_device import gpu
        build=json.loads(args.build.read_text());rows=[]
        for item in build['probes']:
            if sha(item['binary'])!=item['sha256']:raise ValueError('probe binary changed')
            samples=[]
            for i in range(5):
                before=gpu('1')
                r=subprocess.run([item['binary']],capture_output=True,text=True,timeout=120)
                after=gpu('1')
                stem=args.out/(item['name']+'_'+str(i));stem.with_suffix('.stdout').write_text(r.stdout)
                stem.with_suffix('.stderr').write_text(r.stderr)
                if r.returncode:raise RuntimeError('split probe failed: '+r.stderr)
                samples.append(dict(measurement=json.loads(r.stdout),start_sm_clock_mhz=before['sm_clock_mhz'],end_sm_clock_mhz=after['sm_clock_mhz']))
            rows.append(dict(probe=item,processes=samples,median_ns=statistics.median(s['measurement']['ns'] for s in samples),passes='5/5'))
            save(args.out/'result.json',dict(status='partial',rows=rows,evidence='verified'))
        if any(sha(path)!=value for path,value in build['inputs'].items()):raise ValueError('frozen input changed')
        save(args.out/'result.json',dict(status='passed',rows=rows,evidence='verified',gpu=1,threads=384,qualification='provisional_container_pid_namespace'))
        return
    source=args.out/'source';source.mkdir();shutil.copytree(ROOT/'include',source/'include')
    shutil.copyfile(ROOT/'scripts/dnn4_probe_split.cu',source/'probe.cu')
    frozen={str(f):sha(f) for f in source.rglob('*') if f.is_file()};probes=[]
    for name,tn,s,swap in [('normal64x32s6',32,6,0),('normal64x128s16',128,16,0),('swap16x128s32',128,32,1)]:
        binary=args.out/name;log=args.out/(name+'.ptxas.log')
        command=['taskset','-c','136-143','nvcc','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a',
                 '--expt-relaxed-constexpr','-Xptxas','-v','-I'+str(source/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),
                 f'-DDNN4_SPLIT_TN={tn}',f'-DDNN4_SPLIT_S={s}',f'-DDNN4_SPLIT_SWAP={swap}',str(source/'probe.cu'),'-o',str(binary)]
        with log.open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=300)
        text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
        if r.returncode or not spills or any(spills) or re.search(r'C750[47]',text):raise RuntimeError('compile/spill gate: '+str(log))
        sass=args.out/(name+'.sass')
        with sass.open('w') as stream:subprocess.run(['cuobjdump','--dump-sass',str(binary)],stdout=stream,check=True)
        probes.append(dict(name=name,binary=str(binary),sha256=sha(binary),sass_sha256=sha(sass),command=command))
    save(args.out/'build.json',dict(inputs=frozen,probes=probes))
    queue=args.out/'queue';queue.mkdir()
    env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',PYTHONPATH=str(ROOT/'python'))
    save(queue/'queue_split.json',[dict(name='dnn4-split-calibration',gpu=True,timing=True,needs_free_mib=8192,timeout_s=1800,cwd=str(ROOT),env=env,
        command=['taskset','-c','104-111',str(ROOT/'scripts/dnn4_guard_python.py'),str(Path(__file__).resolve()),'--execute','--build',str(args.out/'build.json'),'--out','{attempt_dir}/cal'])])
    command=[str(ROOT/'scripts/dnn4_guard_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(queue),'--out',str(args.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6']
    with (args.out/'dispatch.log').open('w') as stream:subprocess.run(command,env=dict(os.environ,**env),stdout=stream,stderr=subprocess.STDOUT,check=True)
    state=json.loads((args.out/'scheduler/state.json').read_text())['dnn4-split-calibration']
    print(json.dumps(state))
    if state['status']!='done':raise RuntimeError('calibration did not complete')

if __name__=='__main__':main()
