#!/usr/bin/env python3
"""Calibrate production F3 swap, split-K, attention and small-body families."""
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
NORMAL_SPLITS=[2,3,4,6,7,8,9,12,14,16]
SWAP_SPLITS=sorted(set(NORMAL_SPLITS+[28,32]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(source):
    return {str(p):sha(p) for p in [Path(__file__).resolve(),Path(source),
            *sorted((ROOT/'include').rglob('*.h')),*sorted((ROOT/'include').rglob('*.cuh'))]}


def execute(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or not os.environ.get('TILEMEGA_RUN_NONCE'):
        raise ValueError('calibration requires scheduler/guard')
    args.out.mkdir(parents=True,exist_ok=False)
    build=json.loads(args.build.read_text())
    if inputs(build['source'])!=build['inputs']:
        raise ValueError('calibration source changed after build')
    rows=[]
    for probe in build['probes']:
        if sha(probe['binary'])!=probe['binary_sha256']:
            raise ValueError('calibration binary changed')
        for slices in probe['slices']:
            processes=[]
            for index in range(5):
                stem=f"{probe['name']}_s{slices}_{index+1}"
                result=subprocess.run([probe['binary'],str(slices)],capture_output=True,text=True,timeout=120)
                (args.out/(stem+'.stdout')).write_text(result.stdout)
                (args.out/(stem+'.stderr')).write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError('calibration process failed: '+stem)
                payload=json.loads(result.stdout)
                if not payload.get('rows') or any(not math.isfinite(r['ns']) or r['ns']<=0 for r in payload['rows']):
                    raise ValueError('invalid calibration result: '+stem)
                processes.append(payload)
            selectors=[{k:v for k,v in r.items() if k not in ('ns','samples_ns')}
                       for r in processes[0]['rows']]
            for process in processes:
                if selectors!=[{k:v for k,v in r.items() if k not in ('ns','samples_ns')}
                               for r in process['rows']]:
                    raise ValueError('fresh processes returned different keys')
            for index,selector in enumerate(selectors):
                rows.append(dict(**selector,**{k:v for k,v in processes[0].items()
                    if k not in ('rows','cycles_per_ns')},ns=statistics.median(p['rows'][index]['ns'] for p in processes),
                    processes=processes,passes='5/5',binary_sha256=probe['binary_sha256']))
            save(args.out/'result.json',dict(evidence='verified',status='partial',rows=rows))
    if inputs(build['source'])!=build['inputs']:
        raise ValueError('calibration source changed during collection')
    save(args.out/'result.json',dict(evidence='verified',status='passed',rows=rows,
        rejected=build['rejected'],gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']),
        build=str(args.build),build_sha256=sha(args.build),method=build['method'],
        qualification='provisional_container_pid_namespace'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('split','swap','small','attention'),required=True)
    p.add_argument('--tile',default='128x64')
    p.add_argument('--all-tiles',action='store_true')
    p.add_argument('--consumer-only',action='store_true',help='plain split calibration with only its two consumer warpgroups')
    p.add_argument('--tiles',help='explicit normal tiles to repair, comma separated')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.out=args.out.resolve()
    if args.execute:
        execute(args);return
    if args.consumer_only and args.family!='split':
        raise ValueError('consumer-only fixture is only for plain split-K')
    source=ROOT/'scripts'/('dnn3_split_swap_constants.cu' if args.family in ('split','swap') else 'dnn3_task_constants.cu')
    args.out.mkdir(parents=True,exist_ok=False);frozen=inputs(source);inventory=[]
    if args.family=='swap':
        inventory=[(f'swap_{m}',1,m,128,[1]) for m in (8,16)]
    elif args.family=='split':
        tm,tn=map(int,args.tile.split('x'))
        tiles=[(m,n) for m in (64,128,256) for n in (32,64,128,256)
               if (m,n)!=(256,256)] if args.all_tiles else [(tm,tn)]
        if args.tiles:
            tiles=[tuple(map(int,tile.split('x'))) for tile in args.tiles.split(',')]
            if any(m not in (64,128,256) or n not in (32,64,128,256) or
                   m==n==256 for m,n in tiles) or len(set(tiles))!=len(tiles):
                raise ValueError('unsupported or duplicate normal repair tiles')
        inventory=[(f'split_{m}x{n}',0,m,n,NORMAL_SPLITS) for m,n in tiles]
        if not args.consumer_only:
            inventory.extend((f'split_swap_{m}',2,m,128,SWAP_SPLITS) for m in (8,16))
    else:
        inventory=[(kind,code,0,0,[1]) for code,kind in enumerate(
            ['layout','pool2','pool3','global_pool','layernorm','embedding','attention'])
            if (kind=='attention')==(args.family=='attention')]
    probes=[];rejected=[]
    for name,family,tm,tn,slices in inventory:
        binary=args.out/name;log=args.out/(name+'.ptxas.log')
        cmd=['nvcc','-rdc=true','-std=c++17','-O3','-gencode','arch=compute_90a,code=sm_90a',
             '--expt-relaxed-constexpr','-Xptxas','-v','-Xptxas','--device-function-maxrregcount=224',
             '-DF3_PROBE_FAMILY='+str(family),'-DF3_PROBE_TM='+str(tm),'-DF3_PROBE_TN='+str(tn),
             '-DF3_PROBE_CONSUMER_ONLY='+str(int(args.consumer_only)),
             '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party/cutlass/include'),str(source),'-o',str(binary)]
        wrapper=['taskset','-c','136-143','python3',str(ROOT/'scripts/run_sm90r1_build.py'),
                 '--receipt',str(binary)+'.limits.json','--max-gib','48','--timeout','1800',
                 '--lock','/tmp/tilemega_dnn3_constant_build.lock','--',*cmd]
        with log.open('w') as f:
            done=subprocess.run(wrapper,stdout=f,stderr=subprocess.STDOUT,timeout=2100,
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        text=log.read_text();spills=list(map(int,re.findall(r'(\d+) bytes spill (?:stores|loads)',text)))
        regs=list(map(int,re.findall(r'Used (\d+) registers',text)))
        if done.returncode:
            raise RuntimeError('calibration compile failed: '+str(log))
        if not spills or any(spills) or not regs or max(regs)>224:
            rejected.append(dict(name=name,spill_max=max(spills,default=-1),register_max=max(regs,default=-1),
                                 log=str(log),log_sha256=sha(log),gpu_executed=False))
            save(args.out/'rejected.json',rejected);continue
        probes.append(dict(name=name,binary=str(binary),binary_sha256=sha(binary),slices=slices,
                           command=cmd,log=str(log),log_sha256=sha(log),registers=max(regs)))
    if inputs(source)!=frozen:
        raise ValueError('calibration inputs changed during builds')
    if not probes:
        raise RuntimeError('no resource-admissible calibration binaries')
    build=args.out/'build.json';save(build,dict(inputs=frozen,probes=probes,rejected=rejected,
        source=str(source),consumer_only=args.consumer_only,method=('two consumer warpgroups; no producer work in plain split region; ' if args.consumer_only else '')+'clock64 production regions; split/swap one sample per launch; 20ms globaltimer calibration; 8 warm samples discarded'))
    env=dict(CUDA_VISIBLE_DEVICES='1',TILEMEGA_DEVICE_INDEX='1',TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu1.lock',
        TILEMEGA_NUMA_NODE='0',TILEMEGA_GPU_PID_MODE='namespace_limited',TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT/'python'))
    queue=args.out/'queue';queue.mkdir()
    name='dnn3-'+args.family+'-constants'
    command=['taskset','-c','128-135',sys.executable,str(Path(__file__).resolve()),'--family',args.family,
             '--execute','--build',str(build),'--out','{attempt_dir}/constants']
    save(queue/'queue_constants.json',[dict(name=name,gpu=True,timing=True,timeout_s=10800,
         needs_free_mib=4096,command=command,cwd=str(ROOT),env=env)])
    command=[str(ROOT/'scripts/dnn1_idle_floor_python.py'),'-m','tilemega.bench.scheduler','--queue-dir',str(queue),
             '--out',str(args.out/'scheduler'),'--policy',str(ROOT/'configs/bench/sm90r2_guard_policy.json'),
             '--deadline-hours','6']
    with (args.out/'dispatch.log').open('w') as f:
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,**env),stdout=f,stderr=subprocess.STDOUT,check=True)
    state=json.loads((args.out/'scheduler/state.json').read_text())[name]
    if state['status']!='done':
        raise RuntimeError('calibration cohort failed; retain raw receipt')


if __name__=='__main__':
    main()
