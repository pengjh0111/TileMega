#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Recertify immutable CPU artifacts under a new lease, preserving rejected guards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import run_r16_h7a_recovery as recovery
base=recovery.base

HOST_PROBE=r'''
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
static void check(cudaError_t code) { if(code!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(code));std::exit(2); } }
int main(int argc,char**argv) {
  if(argc<5)return 2;void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!lib){std::fprintf(stderr,"dlopen: %s\n",dlerror());return 2;}
  int threads=std::atoi(argv[2]),shared=std::atoi(argv[3]);cudaDeviceProp dev{};check(cudaGetDeviceProperties(&dev,0));
  std::printf("{\"num_sms\":%d,\"major\":%d,\"minor\":%d,\"kernels\":[",dev.multiProcessorCount,dev.major,dev.minor);
  for(int i=4;i<argc;++i) {
    void*function=dlsym(lib,argv[i]);if(!function)return 3;cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr,function));check(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int resident=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,function,threads,shared));
    std::printf("%s{\"name\":\"%s\",\"registers\":%d,\"static_shared\":%zu,\"max_threads\":%d,\"resident\":%d,\"dynamic_shared\":%d,\"threads\":%d}",i==4?"":",",argv[i],attr.numRegs,attr.sharedSizeBytes,attr.maxThreadsPerBlock,resident,shared,threads);
  }
  std::printf("]}\n");dlclose(lib);return 0;
}
'''


def resource_gate(old,query,actual,target,requested):
    if query!=old:raise ValueError('official query resource metadata changed on recertification GPU')
    if min(old['resident'],old['l1'],old['l2'])<requested:raise ValueError('official query cannot support declared placement')
    if actual['num_sms']!=target['resources']['num_sms'] or (actual['major'],actual['minor'])!=(target['sm_major'],target['sm_minor']):raise ValueError('actual artifact device differs from declared target')
    kinds=('tilemega_l1_kernel','tilemega_l2_kernel','tilemega_l1_loop_kernel','tilemega_stage_kernel')
    kernels=actual['kernels']
    if len(kernels)!=len(kinds) or any(sum(kind in row['name'] for row in kernels)!=1 for kind in kinds):raise ValueError('actual artifact kernel set is incomplete')
    for row in kernels:
        if row['resident']<requested or row['threads']!=old['threads'] or row['max_threads']<old['threads'] or row['dynamic_shared']!=old['dynamic_shared']:raise ValueError('actual kernel cannot support declared placement')
        if row['registers']<=0:raise ValueError('actual register metadata is invalid')
    return min(row['resident']for row in kernels)


def verified(path):
    c=recovery.verified(path)
    for key in ('recertification_driver','host_probe','official_query','old_resources'):
        if base.sha(c[key])!=c[key+'_sha256']:raise ValueError('recertification dependency changed: '+key)
    for name,expected in c['artifact_files'].items():
        if base.sha(Path(c['directory'])/name)!=expected:raise ValueError('immutable CPU output changed: '+name)
    for name,expected in c['previous_failure_files'].items():
        if base.sha(name)!=expected:raise ValueError('original rejected lease receipt changed')
    return c


def recertify(path,attempt):
    c=verified(path);base.require_lease(c);d=Path(c['directory']);out=Path(attempt).absolute();out.mkdir(parents=True,exist_ok=True)
    old=base.read(c['old_resources']);query_proc=subprocess.run([c['official_query']],capture_output=True,text=True)
    (out/'official-query.stdout').write_text(query_proc.stdout);(out/'official-query.stderr').write_text(query_proc.stderr)
    if query_proc.returncode:raise ValueError('official independent resource query failed')
    query=json.loads(query_proc.stdout)
    command=[c['host_probe'],str(d/'plan.so'),str(old['threads']),str(old['dynamic_shared']),*c['kernel_symbols']]
    proc=subprocess.run(command,capture_output=True,text=True)
    (out/'actual-artifact.stdout').write_text(proc.stdout);(out/'actual-artifact.stderr').write_text(proc.stderr)
    if proc.returncode:raise ValueError('actual final artifact CUDA resource query failed')
    actual=json.loads(proc.stdout);identity=recovery.native_identity(d/'plan.so');target=base.read(c['target'])
    resident=resource_gate(old,query,actual,target,c['requested_residency'])
    value=dict(exit_code=0,kind='independent_guarded_recertification',time=time.time(),pid=os.getpid(),gpu=c['gpu'],source_head=c['source_head'],cli_sha256=c['cli_sha256'],artifact_id=identity['artifact_id'],binary_sha256=identity['binary_sha256'],official_query=query,actual_final_artifact_resources=actual,actual_min_residency=resident,command=command,previous_rejected_guard=c['previous_failure_files'],original_compile_was_guard_accepted=False,no_compilation_performed=True,public_cuda_api='cudaFuncGetAttributes/cudaFuncSetAttribute/cudaOccupancyMaxActiveBlocksPerMultiprocessor on actual dlopen hoststubs; same API as TargetSpec::ActiveBlocksPerSM')
    base.write(d/'guarded-build-result.json',value)


def accept(path):
    c=verified(path);recovery.accept(path)
    d=Path(c['directory']);value=base.read(d/'cpu-build-result.json');value.update(recertification=base.read(d/'guarded-build-result.json'),original_compile_was_guard_accepted=False)
    base.write(d/'cpu-build-result.json',value)


def prepare(args):
    old=recovery.verified(args.candidate);r=Path(old['run_root']);out=args.out.absolute()
    if out.exists():raise FileExistsError('new immutable recertification run-id required')
    original=Path(old['directory']);old_result=base.read(original/'guarded-build-result.json');failure=args.failed_attempt.absolute()/'guard_result.json'
    if old_result['exit_code']!=0 or base.read(failure).get('code')==0:raise ValueError('requires successful CPU output from an explicitly rejected lease')
    identity=recovery.native_identity(original/'plan.so');out.mkdir(parents=True);d=out/'artifact';d.mkdir()
    artifact_files={}
    for p in original.glob('plan.so*'):
        if p.is_file():
            dest=d/p.name;shutil.copy2(p,dest);artifact_files[p.name]=base.sha(dest)
    base.write(out/'original-compiler-result.json',old_result);shutil.copy2(failure,out/'original-guard-result.json')
    for src,name in [(Path(__file__),'run_r16_h7a_recertify.py'),(Path(recovery.__file__),'run_r16_h7a_recovery.py'),(Path(base.__file__),'run_r16_dnn_candidates.py'),(Path(recovery.__file__).with_name('r16_measurement_validation.py'),'r16_measurement_validation.py')]:shutil.copy2(src,out/name)
    query=args.query.absolute();qcopy=out/'official-resource-query';shutil.copy2(query,qcopy);shutil.copy2(query.parent/'resources.json',out/'old-resources.json');shutil.copy2(query.parent/'query.cu',out/'official-query.cu');shutil.copy2(query.parent/'candidate.cu',out/'official-query-candidate.cu')
    source=out/'actual-artifact-probe.cpp';source.write_text(HOST_PROBE);probe=out/'actual-artifact-probe'
    command=['g++','-std=c++17','-O2',str(source),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl','-o',str(probe)]
    subprocess.run(command,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});base.write(out/'host-probe-build.json',dict(command=command,source_sha256=base.sha(source),binary_sha256=base.sha(probe),gpu_used=False))
    symbols=subprocess.check_output(['nm','-D',str(d/'plan.so')],text=True);kinds=['tilemega_l1_kernel','tilemega_l2_kernel','tilemega_l1_loop_kernel','tilemega_stage_kernel'];selected=[]
    for kind in kinds:
        matches=[line.split()[-1]for line in symbols.splitlines()if kind in line and '__device_stub__'not in line]
        if len(matches)!=1:raise ValueError('actual kernel hoststub symbol is not unique: '+kind)
        selected.append(matches[0])
    c={**old,'candidate_id':out.name+':'+old['arm']+':B1','directory':str(d),'gpu':args.gpu,'original_candidate':str(args.candidate.absolute()),'original_failed_attempt':str(args.failed_attempt.absolute()),'artifact_files':artifact_files,'previous_failure_files':{str(out/'original-guard-result.json'):base.sha(out/'original-guard-result.json'),str(out/'original-compiler-result.json'):base.sha(out/'original-compiler-result.json')},'official_query':str(qcopy),'official_query_sha256':base.sha(qcopy),'old_resources':str(out/'old-resources.json'),'old_resources_sha256':base.sha(out/'old-resources.json'),'host_probe':str(probe),'host_probe_sha256':base.sha(probe),'kernel_symbols':selected,'requested_residency':identity['plan'].get('residency',1) or 1}
    for key,name in [('recertification_driver','run_r16_h7a_recertify.py'),('recovery_driver','run_r16_h7a_recovery.py'),('baseline_driver','run_r16_dnn_candidates.py'),('validation_helper','r16_measurement_validation.py')]:c[key]=str(out/name);c[key+'_sha256']=base.sha(out/name)
    stem='h7a-'+out.name;c['build_job']=stem+'-recertify';path=out/'candidate.json';base.write(path,c)
    node=0 if args.gpu<4 else 1;env=dict(PYTHONPATH=c['source']+'/python',CUDA_VISIBLE_DEVICES=str(args.gpu),TILEMEGA_DEVICE_INDEX=str(args.gpu),TILEMEGA_NUMA_NODE=str(node),TILEMEGA_GPU_PID_MODE='namespace_limited',OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');prefix=[c['python'],c['recertification_driver']];common=dict(cwd=str(r),env=env,priority=2,needs_free_mib=4096)
    jobs=[dict(common,name=c['build_job'],gpu=True,timing=False,timeout_s=300,command=['numactl',f'--cpunodebind={node}']+prefix+['recertify','--candidate',str(path),'--attempt','{attempt_dir}']),dict(common,name=stem+'-accept',gpu=False,timing=False,timeout_s=300,after=[c['build_job']],command=prefix+['accept','--candidate',str(path)])];checks=[]
    for fresh in range(5):
        name=stem+f'-check-f{fresh}';checks.append(name);jobs.append(dict(common,name=name,gpu=True,timing=False,timeout_s=900,after=[stem+'-accept'],command=['numactl',f'--cpunodebind={node}']+prefix+['check','--candidate',str(path),'--fresh',str(fresh),'--attempt','{attempt_dir}']))
    jobs.append(dict(common,name=stem+'-gate',gpu=False,timing=False,timeout_s=120,after=checks,command=prefix+['aggregate','--candidate',str(path)]))
    for rnd in range(3):jobs.append(dict(common,name=stem+f'-measure-r{rnd}',gpu=True,timing=True,timeout_s=1800,after=[stem+'-gate'],command=['numactl',f'--cpunodebind={node}']+prefix+['measure','--candidate',str(path),'--round-index',str(rnd),'--attempt','{attempt_dir}']))
    base.write(out/'queue-preview.json',jobs);print(json.dumps(dict(queue=str(out/'queue-preview.json'),jobs=len(jobs))))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True);a=sub.add_parser('prepare');a.add_argument('--candidate',type=Path,required=True);a.add_argument('--failed-attempt',type=Path,required=True);a.add_argument('--query',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--gpu',type=int,default=0)
    for operation in ['recertify','accept','check','aggregate','measure']:
        a=sub.add_parser(operation);a.add_argument('--candidate',type=Path,required=True)
        if operation in ['recertify','check','measure']:a.add_argument('--attempt',type=Path,required=True)
        if operation=='check':a.add_argument('--fresh',type=int,required=True)
        if operation=='measure':a.add_argument('--round-index',type=int,required=True)
    args=p.parse_args()
    if args.op=='prepare':prepare(args)
    elif args.op=='recertify':recertify(args.candidate,args.attempt)
    elif args.op=='accept':accept(args.candidate)
    else:
        verified(args.candidate)
        if args.op=='check':base.check(args.candidate,args.fresh,args.attempt)
        elif args.op=='aggregate':base.aggregate(args.candidate)
        else:base.measure(args.candidate,args.round_index,args.attempt)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError)as error:print('H7a recertification rejected: '+str(error),file=sys.stderr);raise SystemExit(3)
