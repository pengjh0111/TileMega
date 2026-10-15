#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit public-CUDA opt-in workaround for an unchanged pre-fix artifact."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import run_r16_h7a_recertify as recert
base=recert.base


def verified(path):
    c=recert.verified(path)
    if base.sha(c['optin_driver'])!=c['optin_driver_sha256']:raise ValueError('frozen opt-in driver changed')
    return c


def child(path,module,arguments):
    c=verified(path);base.require_lease(c)
    if module not in ('tilemega.dnn.check_generated','tilemega.dnn.benchmark'):raise ValueError('workaround module is not a registered full-model runner')
    library=Path(c['directory'])/'plan.so'
    if '--library'not in arguments or Path(arguments[arguments.index('--library')+1]).resolve()!=library.resolve():raise ValueError('workaround must apply to the measured artifact')
    import torch
    lib=C.CDLL(str(library),mode=C.RTLD_LOCAL);cuda=C.CDLL('libcudart.so.12')
    cuda.cudaSetDevice.argtypes=[C.c_int];cuda.cudaSetDevice.restype=C.c_int
    cuda.cudaFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int];cuda.cudaFuncSetAttribute.restype=C.c_int
    if cuda.cudaSetDevice(0):raise ValueError('CUDA visible device could not be selected')
    shared=base.read(c['old_resources'])['dynamic_shared'];applied=[]
    for symbol in c['kernel_symbols']:
        address=C.cast(getattr(lib,symbol),C.c_void_p)
        status=cuda.cudaFuncSetAttribute(address,8,shared)
        if status:raise ValueError('public cudaFuncSetAttribute rejected opt-in: '+str(status))
        applied.append(symbol)
    metadata=dict(enabled=True,temporary_workaround=True,production_fix_commit=c['production_fix_commit'],artifact_head_unchanged=c['source_head'],artifact_id=recert.recovery.native_identity(library)['artifact_id'],library_sha256=base.sha(library),driver_sha256=c['optin_driver_sha256'],dynamic_shared_bytes=shared,api='cudaFuncSetAttribute(MaxDynamicSharedMemorySize)',kernels=applied,scope='per-process CUDA function configuration only; no kernel/source/weights/threshold changes')
    receipt=Path(os.environ['TILEMEGA_OPTIN_RECEIPT']);base.write(receipt,dict(metadata,pid=os.getpid(),module=module,started=time.time(),completed=False))
    sys.argv=[module,*arguments];runpy.run_module(module,run_name='__main__')
    destination=Path(arguments[arguments.index('--out')+1]);destination=destination if module.endswith('check_generated')else destination/'measurements.json'
    value=base.read(destination);value['runtime_shared_memory_optin_workaround']=metadata;value['configuration']={**value.get('configuration',{}),'runtime_shared_memory_optin_workaround':metadata};base.write(destination,value)
    base.write(receipt,dict(metadata,pid=os.getpid(),module=module,ended=time.time(),completed=True))


def execute(path,operation,fresh=None,rnd=None,attempt=None):
    c=verified(path)
    if operation=='recertify':return recert.recertify(path,attempt)
    if operation=='accept':return recert.accept(path)
    if operation=='aggregate':return base.aggregate(path)
    original=base.child
    def wrapped(command,cwd,env,log):
        if len(command)<3 or command[1]!='-m' or command[2]not in ('tilemega.dnn.check_generated','tilemega.dnn.benchmark'):raise ValueError('unexpected measured child command')
        command=[command[0],c['optin_driver'],'child','--candidate',str(path),'--module',command[2],'--',*command[3:]]
        return original(command,cwd,{**env,'TILEMEGA_OPTIN_RECEIPT':str(Path(log).parent/'runtime-optin.json')},log)
    base.child=wrapped
    if operation=='check':return base.check(path,fresh,attempt)
    return base.measure(path,rnd,attempt)


def prepare(args):
    old=recert.verified(args.candidate);origin=Path(old['directory']);accepted=base.read(origin/'cpu-build-result.json')
    if not accepted.get('passed'):raise ValueError('requires a separately recertified immutable CPU artifact')
    out=args.out.absolute()
    if out.exists():raise FileExistsError('new workaround run-id required')
    out.mkdir(parents=True);directory=out/'artifact';directory.mkdir()
    for name in old['artifact_files']:shutil.copy2(origin/name,directory/name)
    for name,module in [('run_r16_h7a_optin.py',__file__),('run_r16_h7a_recertify.py',recert.__file__),('run_r16_h7a_recovery.py',recert.recovery.__file__),('run_r16_dnn_candidates.py',base.__file__),('r16_measurement_validation.py',Path(base.__file__).with_name('r16_measurement_validation.py'))]:shutil.copy2(module,out/name)
    c={**old,'candidate_id':out.name+':'+old['arm']+':B1','directory':str(directory),'gpu':args.gpu,'production_fix_commit':'186c2c48c','previous_recertification':str(args.candidate.absolute()),'runtime_workaround':'explicit public CUDA shared-memory opt-in; original artifact unchanged; final production rebuild still required'}
    for key,name in [('optin_driver','run_r16_h7a_optin.py'),('recertification_driver','run_r16_h7a_recertify.py'),('recovery_driver','run_r16_h7a_recovery.py'),('baseline_driver','run_r16_dnn_candidates.py'),('validation_helper','r16_measurement_validation.py')]:c[key]=str(out/name);c[key+'_sha256']=base.sha(out/name)
    stem='h7a-'+out.name;c['build_job']=stem+'-recertify';path=out/'candidate.json';base.write(path,c)
    root=Path(c['run_root']);node=0 if args.gpu<4 else 1;env=dict(PYTHONPATH=c['source']+'/python',CUDA_VISIBLE_DEVICES=str(args.gpu),TILEMEGA_DEVICE_INDEX=str(args.gpu),TILEMEGA_NUMA_NODE=str(node),TILEMEGA_GPU_PID_MODE='namespace_limited',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');prefix=[c['python'],c['optin_driver']];common=dict(cwd=str(root),env=env,priority=2,needs_free_mib=4096)
    jobs=[dict(common,name=c['build_job'],gpu=True,timing=False,timeout_s=300,command=['numactl',f'--cpunodebind={node}']+prefix+['recertify','--candidate',str(path),'--attempt','{attempt_dir}']),dict(common,name=stem+'-accept',gpu=False,timing=False,timeout_s=300,after=[c['build_job']],command=prefix+['accept','--candidate',str(path)])];checks=[]
    for fresh in range(5):
        name=stem+f'-check-f{fresh}';checks.append(name);jobs.append(dict(common,name=name,gpu=True,timing=False,timeout_s=900,after=[stem+'-accept'],command=['numactl',f'--cpunodebind={node}']+prefix+['check','--candidate',str(path),'--fresh',str(fresh),'--attempt','{attempt_dir}']))
    jobs.append(dict(common,name=stem+'-gate',gpu=False,timing=False,timeout_s=120,after=checks,command=prefix+['aggregate','--candidate',str(path)]))
    for rnd in range(3):jobs.append(dict(common,name=stem+f'-measure-r{rnd}',gpu=True,timing=True,timeout_s=1800,after=[stem+'-gate'],command=['numactl',f'--cpunodebind={node}']+prefix+['measure','--candidate',str(path),'--round-index',str(rnd),'--attempt','{attempt_dir}']))
    base.write(out/'queue-preview.json',jobs);print(json.dumps(dict(queue=str(out/'queue-preview.json'),jobs=len(jobs))))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True);a=sub.add_parser('prepare');a.add_argument('--candidate',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--gpu',type=int,default=0)
    for op in ['recertify','accept','check','aggregate','measure','child']:
        a=sub.add_parser(op);a.add_argument('--candidate',type=Path,required=True)
        if op in ['recertify','check','measure']:a.add_argument('--attempt',type=Path,required=True)
        if op=='check':a.add_argument('--fresh',type=int,required=True)
        if op=='measure':a.add_argument('--round-index',type=int,required=True)
        if op=='child':a.add_argument('--module',required=True);a.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.operation=='prepare':prepare(args)
    elif args.operation=='child':child(args.candidate,args.module,args.arguments[1:]if args.arguments[:1]==['--']else args.arguments)
    else:execute(args.candidate,args.operation,getattr(args,'fresh',None),getattr(args,'round_index',None),getattr(args,'attempt',None))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError)as error:print('H7a opt-in workaround rejected: '+str(error),file=sys.stderr);raise SystemExit(3)
