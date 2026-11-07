#!/usr/bin/env python3
"""Collect five-process resident F3 main-loop constants; not a complete §9.2 table."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def inputs():
    paths=[Path(__file__), ROOT/'scripts/dnn3_constants.cu',
           *sorted((ROOT/'include').rglob('*.h')),
           *sorted((ROOT/'include').rglob('*.cuh'))]
    return {str(p):sha(p) for p in paths}


def execute(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('constant collection requires the scheduler and guard')
    args.out.mkdir(parents=True, exist_ok=False)
    build=json.loads(args.build.read_text())
    if build['inputs']!=inputs():
        raise ValueError('constant probe inputs differ from their build')
    rows=[]
    for probe in build['probes']:
        binary=Path(probe['binary'])
        if sha(binary)!=probe['binary_sha256']:
            raise ValueError('constant binary changed')
        processes=[]
        for index in range(5):
            stem=f"{probe['mode']}_{probe['tile_m']}x{probe['tile_n']}_{index+1}"
            done=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            (args.out/(stem+'.stdout')).write_text(done.stdout)
            (args.out/(stem+'.stderr')).write_text(done.stderr)
            if done.returncode:
                raise RuntimeError('constant process failed: '+stem)
            row=json.loads(done.stdout)
            if (row['metric']!='f3_k64_ns' or row['mode']!=probe['mode'] or
                    row['tile_m']!=probe['tile_m'] or row['tile_n']!=probe['tile_n'] or
                    not math.isfinite(row['ns']) or row['ns']<=0):
                raise ValueError('constant output differs from the probe inventory')
            processes.append(row)
        rows.append(dict(metric='f3_k64_ns',mode=probe['mode'],tile_m=probe['tile_m'],
                         tile_n=probe['tile_n'],ns=statistics.median(r['ns'] for r in processes),
                         processes=processes,passes='5/5',binary_sha256=probe['binary_sha256']))
        save(args.out/'result.json',dict(evidence='verified',status='partial',rows=rows))
    if inputs()!=build['inputs']:
        raise ValueError('constant source changed during measurement')
    save(args.out/'result.json',dict(evidence='verified',status='passed',rows=rows,
         scope='resident normal/gather main loops only; other §9.2 families pending',
         gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']),build=str(args.build),
         build_sha256=sha(args.build),qualification='provisional_container_pid_namespace'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tile',default='128x64',help='normal TMxTN; gather128x64 is also measured')
    parser.add_argument('--all-normal',action='store_true',help='all11 normal tiles plus gather128x64')
    parser.add_argument('--build',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.execute:
        execute(args)
        return
    tm,tn=map(int,args.tile.split('x'))
    if tm not in (64,128,256) or tn not in (32,64,128,256) or tm==tn==256:
        raise ValueError('unsupported F3 normal tile')
    args.out.mkdir(parents=True,exist_ok=False)
    frozen=inputs();probes=[]
    normal=[(m,n) for m in (64,128,256) for n in (32,64,128,256)
            if (m,n)!=(256,256)] if args.all_normal else [(tm,tn)]
    for mode,m,n in [('normal',m,n) for m,n in normal]+[('gather',128,64)]:
        name=f'{mode}_{m}x{n}';binary=args.out/name;log=args.out/(name+'.ptxas.log')
        cmd=['taskset','-c','136-143','nvcc','-rdc=true','-std=c++17','-O3',
             '-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
             '-Xptxas','-v','-Xptxas','--device-function-maxrregcount=224',
             '-DF3_PROBE_TM='+str(m),'-DF3_PROBE_TN='+str(n),
             '-DF3_PROBE_GATHER='+str(int(mode=='gather')),
             '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),
             str(ROOT/'scripts/dnn3_constants.cu'),'-o',str(binary)]
        with log.open('w') as stream:
            done=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=1200,
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        text=log.read_text()
        spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
        registers=list(map(int,re.findall(r'Used (\d+) registers',text)))
        if done.returncode or not spills or any(spills) or not registers or max(registers)>224:
            raise RuntimeError('constant build rejected; retain '+str(log))
        probes.append(dict(mode=mode,tile_m=m,tile_n=n,binary=str(binary),
             binary_sha256=sha(binary),command=cmd,log=str(log),log_sha256=sha(log),
             registers=max(registers),spill_bytes=0))
    if inputs()!=frozen:
        raise ValueError('constant inputs changed during builds')
    build=args.out/'build.json';save(build,dict(inputs=frozen,probes=probes))
    environment=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',
        TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT/'python'))
    queue=args.out/'queue';queue.mkdir()
    cmd=['taskset','-c','128-135',sys.executable,str(Path(__file__).resolve()),
         '--execute','--build',str(build),'--out','{attempt_dir}/constants']
    save(queue/'queue_constants.json',[dict(name='dnn3-resident-constants',gpu=True,timing=True,
         timeout_s=3600,needs_free_mib=4096,command=cmd,cwd=str(ROOT),env=environment)])
    cmd=[str(ROOT/'scripts/dnn1_idle_floor_python.py'),'-m','tilemega.bench.scheduler',
         '--queue-dir',str(queue),'--out',str(args.out/'scheduler'),
         '--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6']
    with (args.out/'dispatch.log').open('w') as stream:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,**environment),stdout=stream,
                       stderr=subprocess.STDOUT,check=True)
    state=json.loads((args.out/'scheduler/state.json').read_text())['dnn3-resident-constants']
    if state['status']!='done':
        raise RuntimeError('constant cohort failed; retain scheduler receipt')


if __name__=='__main__':
    main()
