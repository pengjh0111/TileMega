#!/usr/bin/env python3
"""Guard five-process resident normal-mode epilogue calibration (§9.2)."""
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
CLASSES=('plain','bias_act','bias_res_act','deferred_ln',
         'deferred_ln_gelu','residual_ln_rowstats')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def inputs():
    paths=[Path(__file__),ROOT/'scripts/dnn3_epilogue_constants.cu',
           *sorted((ROOT/'include').rglob('*.h')),
           *sorted((ROOT/'include').rglob('*.cuh'))]
    return {str(p):sha(p) for p in paths}


def execute(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('constant collection requires the scheduler and guard')
    args.out.mkdir(parents=True,exist_ok=False)
    build=json.loads(args.build.read_text())
    if inputs()!=build['inputs']:
        raise ValueError('epilogue inputs changed after compilation')
    rows=[]
    for probe in build['probes']:
        if sha(probe['binary'])!=probe['binary_sha256']:
            raise ValueError('epilogue binary changed')
        processes=[]
        for index in range(5):
            stem=f"{probe['class']}_{probe['tile_m']}x{probe['tile_n']}_{index+1}"
            done=subprocess.run([probe['binary']],capture_output=True,text=True,timeout=120)
            (args.out/(stem+'.stdout')).write_text(done.stdout)
            (args.out/(stem+'.stderr')).write_text(done.stderr)
            if done.returncode:
                raise RuntimeError('epilogue process failed: '+stem)
            row=json.loads(done.stdout)
            if (row['metric']!='f3_epilogue_ns' or
                    any(row[k]!=probe[k] for k in ('class','tile_m','tile_n')) or
                    not math.isfinite(row['ns']) or row['ns']<=0):
                raise ValueError('epilogue output differs from inventory')
            processes.append(row)
        rows.append(dict(metric='f3_epilogue_ns',**{k:probe[k] for k in
                         ('class','tile_m','tile_n','binary_sha256')},
                         ns=statistics.median(row['ns'] for row in processes),
                         processes=processes,passes='5/5'))
        save(args.out/'result.json',dict(evidence='verified',status='partial',rows=rows))
    if inputs()!=build['inputs']:
        raise ValueError('epilogue inputs changed during collection')
    save(args.out/'result.json',dict(evidence='verified',status='passed',rows=rows,
         scope='resident normal-mode epilogue; one region per launch; host overhead not clocked',
         rejected_builds=build['rejected_builds'],build=str(args.build),
         build_sha256=sha(args.build),gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']),
         qualification='provisional_container_pid_namespace'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tile',default='128x64')
    parser.add_argument('--all-normal',action='store_true')
    parser.add_argument('--tiles',help='explicit normal tiles to repair, comma separated')
    parser.add_argument('--classes',default=','.join(CLASSES))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--build',type=Path)
    args=parser.parse_args();args.out=args.out.resolve()
    if args.execute:
        execute(args);return
    tm,tn=map(int,args.tile.split('x'))
    tiles=[(m,n) for m in (64,128,256) for n in (32,64,128,256)
           if (m,n)!=(256,256)] if args.all_normal else [(tm,tn)]
    if args.tiles:
        tiles=[tuple(map(int,tile.split('x'))) for tile in args.tiles.split(',')]
    classes=args.classes.split(',')
    if (any(c not in CLASSES for c in classes) or len(set(classes))!=len(classes) or
            any(m not in (64,128,256) or n not in (32,64,128,256) or
                m==n==256 for m,n in tiles)):
        raise ValueError('unsupported epilogue inventory')
    args.out.mkdir(parents=True,exist_ok=False)
    frozen=inputs();probes=[];rejected=[]
    for m,n in tiles:
        for c in classes:
            name=f'{c}_{m}x{n}';binary=args.out/name;log=args.out/(name+'.ptxas.log')
            command=['taskset','-c','136-143','nvcc','-rdc=true','-std=c++17','-O3',
                '-gencode','arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
                '-Xptxas','-v','-Xptxas','--device-function-maxrregcount=224',
                '-DF3_PROBE_TM='+str(m),'-DF3_PROBE_TN='+str(n),
                '-DF3_PROBE_CLASS='+str(CLASSES.index(c)),'-I'+str(ROOT/'include'),
                '-I'+str(ROOT/'third_party/cutlass/include'),
                str(ROOT/'scripts/dnn3_epilogue_constants.cu'),'-o',str(binary)]
            with log.open('w') as stream:
                done=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                     timeout=1200,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            text=log.read_text()
            spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
            registers=list(map(int,re.findall(r'Used (\d+) registers',text)))
            row=dict(**{'class':c},tile_m=m,tile_n=n,command=command,
                     log=str(log),log_sha256=sha(log),compile_exit=done.returncode,
                     spill_max=max(spills,default=-1),register_max=max(registers,default=-1))
            # A DT-SPILL exclusion remains visible; it never produces a cost.
            if not done.returncode and spills and not any(spills) and registers and max(registers)<=224:
                row.update(binary=str(binary),binary_sha256=sha(binary));probes.append(row)
            elif done.returncode or not spills or not registers:
                save(args.out/'build_failure.json',row)
                raise RuntimeError('epilogue compilation failed; retain '+str(log))
            else:
                rejected.append(row)
    if inputs()!=frozen or not probes:
        raise ValueError('epilogue sources changed or no legal probes remain')
    build=args.out/'build.json'
    save(build,dict(inputs=frozen,probes=probes,rejected_builds=rejected))
    environment=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',
        TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT/'python'))
    queue=args.out/'queue';queue.mkdir()
    command=['taskset','-c','128-135',sys.executable,str(Path(__file__).resolve()),
             '--execute','--build',str(build),'--out','{attempt_dir}/constants']
    save(queue/'queue_constants.json',[dict(name='dnn3-resident-epilogues',gpu=True,
         timing=True,timeout_s=3600,needs_free_mib=4096,command=command,
         cwd=str(ROOT),env=environment)])
    command=[str(ROOT/'scripts/dnn1_idle_floor_python.py'),'-m','tilemega.bench.scheduler',
        '--queue-dir',str(queue),'--out',str(args.out/'scheduler'),
        '--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),'--deadline-hours','6']
    with (args.out/'dispatch.log').open('w') as stream:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**environment),
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    state=json.loads((args.out/'scheduler/state.json').read_text())['dnn3-resident-epilogues']
    if state['status']!='done':
        raise RuntimeError('epilogue cohort failed; retain scheduler receipt')


if __name__=='__main__':
    main()
