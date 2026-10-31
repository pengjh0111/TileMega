#!/usr/bin/env python3
"""Measure frozen MX-6 stage keys in one guarded process, never select a plan."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np


class Gemm(C.Structure):
    _fields_=[(n,C.c_void_p)for n in ('a','w','residual','out','partial','tickets','a_tokens','residual_tokens')]+[('a_stride',C.c_int),('residual_stride',C.c_int),('epsilon',C.c_float)]


class Regular(C.Structure):
    _fields_=[('gemm',Gemm)]+[(n,C.c_void_p)for n in ('head_partial','head_tickets','tokens','counter','stamps')]+[('resources',C.POINTER(C.c_int)),('iterations',C.c_int),('copies',C.c_int)]


class Grouped(C.Structure):
    _fields_=[(n,C.c_void_p)for n in ('x','w','logits','h','partials','residual','out','tickets','counter','starts','ends')]+[('resources',C.POINTER(C.c_int)),('iterations',C.c_int),('copies',C.c_int),('epsilon',C.c_float)]


class AttentionOps(C.Structure):
    _fields_=[(n,C.c_void_p)for n in ('qkv','q_norm','k_norm','cosine','sine','k_cache','v_cache','ctx','partial','tickets')]+[('capacity',C.c_int),('past',C.c_int),('epsilon',C.c_float)]


class Attention(C.Structure):
    _fields_=[('attention',AttentionOps)]+[(n,C.c_void_p)for n in ('counter','stamps')]+[('resources',C.POINTER(C.c_int)),('iterations',C.c_int),('copies',C.c_int)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(ref):
    p=Path(ref['path'])
    if sha(p)!=ref['sha256']:
        raise ValueError('changed cost evidence: '+str(p))
    return p


def library(row):
    p=checked(row['binary']);build=json.loads(checked(row['build']).read_text())
    compiled_key=dict(row['key'])
    if compiled_key['stage']=='attention':compiled_key.pop('length')
    if build['status']!='resource_fit' or build['binary']!=row['binary'] or build['key']!=compiled_key:
        raise ValueError('key/build/source mismatch')
    return C.CDLL(str(p)),build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--round',type=int,choices=(1,2,3,4,5),required=True)
    args=parser.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':parser.error('requires mx_gpu_job.sh')
    if args.out.exists():parser.error('preserve previous raw samples')
    from tilemega.mx5_plan_occupancy import guard_evidence
    guard=guard_evidence(os.environ)
    import torch as t
    t.set_num_threads(1)
    t.manual_seed(63102)
    stream=t.cuda.current_stream().cuda_stream
    keys=json.loads(args.keys.read_text());cache={}
    args.out.parent.mkdir(parents=True,exist_ok=True)

    def tensor(shape,dtype=t.bfloat16,zero=False):
        if zero:return t.zeros(shape,dtype=dtype,device='cuda')
        return t.empty(shape,dtype=dtype,device='cuda').uniform_(-.03,.03)

    def emit(output,row):
        output.write(json.dumps(row)+'\n');output.flush()

    with args.out.open('x')as output:
        for row in keys:
            key=row['key'];kind=key['stage'];b=key['b'];epsilon=key['epsilon'];lib,build=library(row)
            grouped=kind in ('gate_up','down');attention=kind=='attention'
            Type=Grouped if grouped else Attention if attention else Regular
            name='mx6_grouped_bench'if grouped else'mx6_attention_bench'if attention else'mx6_stage'
            size=getattr(lib,name+'_args_size');size.restype=C.c_uint
            if size()!=C.sizeof(Type):raise ValueError('benchmark ABI mismatch')
            function=getattr(lib,name);function.argtypes=[C.POINTER(Type),C.c_void_p];function.restype=C.c_int
            resource=(C.c_int*4)();probe=Type();probe.resources=resource;probe.copies=1
            if attention:probe.attention.capacity=1088;probe.attention.past=key['length']-1
            status=function(C.byref(probe),stream)
            common=dict(key=key,round=args.round,PID=os.getpid(),physical_card=int(os.environ['TILEMEGA_DEVICE_INDEX']),
                executor='mx_static_stage_bench',implementation='mx6_'+kind+'_v'+str(key.get('attn_version',3)),contract='bf16',source_digest=build['source_digest'],
                binary=row['binary'],build=row['build'],driver_sha256=sha(__file__),key_file_sha256=sha(args.keys),
                resources=dict(MTreg=resource[0],stack=resource[1],static_smem=resource[2],resident=resource[3]),
                timing_eligible=False,production_acceptance=False,evidence='verified',input_seed=63102,guard=guard)
            if status:
                emit(output,dict(common,status='resource_rejected',query_status=status));continue
            assert resource[1]==resource[2]==0 and resource[3]==1 and resource[0]<=255
            assert resource[0]==build['resources']['registers']
            if row.get('zero_prefetch'):
                zero=row['zero_prefetch'];zero_lib=C.CDLL(str(checked(zero['binary'])))
                zero_fn=getattr(zero_lib,name);zero_fn.argtypes=[C.POINTER(Type),C.c_void_p];zero_fn.restype=C.c_int
                zero_resources=(C.c_int*4)();probe.resources=zero_resources
                assert zero_fn(C.byref(probe),stream)==0
                assert list(zero_resources)==[key['r_next'],0,0,1]
                common['zero_prefetch_query']=dict(binary=zero['binary'],MTreg=zero_resources[0],stack=0,static_smem=0,resident=1)
            if attention:
                dim,q,hkv=key['dim'],key['q'],key['hkv'];heads=b*hkv;blocks=(key['length']+31)//32
                addressed_per_copy=heads*blocks*32*dim*4
                copies=max(1,math.ceil(256*1024**2/addressed_per_copy));iterations=max(128,2*copies+2)
                tag=(key['model'],b,'attention',key.get('attn_version',3),copies)
                if tag not in cache:
                    # All capacity bytes are initialized, but qualification counts only read blocks.
                    cache.clear();t.cuda.empty_cache()
                    capacity=1088
                    cache[tag]=dict(qkv=tensor((heads,q+2,dim)),qn=tensor((heads*q*dim,)),kn=tensor((heads*dim,)),
                        cosine=tensor((capacity,dim)),sine=tensor((capacity,dim)),k=tensor((copies,heads,capacity*dim)),
                        v=tensor((copies,heads,capacity*dim)),ctx=tensor((heads,q,dim)),
                        partial=tensor((heads*(34 if key.get('attn_version',3)==2 else 4)*q*(dim+2),),t.float32),tickets=tensor((heads,),t.int32,True))
                a=cache[tag];counter=tensor((8,),t.int64,True);stamps=tensor((iterations,),t.int64,True)
                ops=AttentionOps(*[a[n].data_ptr()for n in ('qkv','qn','kn','cosine','sine','k','v','ctx','partial','tickets')],1088,key['length']-1,epsilon)
                native=Attention(ops,counter.data_ptr(),stamps.data_ptr(),resource,iterations,copies);result_tensor=a['ctx']
                cold=dict(copies=copies,addressed_rotating_KV_bytes=addressed_per_copy*copies,
                    allocated_KV_bytes=a['k'].numel()*4,rotation_kind='distinct block-fragment KV copies',read_blocks_per_copy=blocks)
            elif grouped:
                n,k,e=key['n'],key['k'],key['experts'];meta_path=Path(key['replay']['path']);checked(key['replay']);meta=json.loads(meta_path.read_text())
                replay_path=Path(meta['bundle_path'])
                if sha(replay_path)!=meta['bundle_sha256']:raise ValueError('changed routing replay')
                replay=t.load(replay_path,map_location='cpu',weights_only=False);active=[set(v.reshape(-1).tolist())for v in replay['selected']]
                copies=max(1,math.ceil(256*1024**2/(min(map(len,active))*n*k*2)));iterations=128
                tag=(key['model'],b,kind,copies)
                if tag not in cache:
                    cache.clear();t.cuda.empty_cache();hidden,intermediate=key['hidden'],key['intermediate']
                    cache[tag]=dict(x=tensor((b,hidden)),w=tensor((copies,e,n,k)),logits=replay['logits'].cuda(),
                        h=tensor((e,b,intermediate)),partials=tensor((b*8,hidden)),residual=tensor((b,hidden)),
                        out=tensor((b,hidden)),tickets=tensor((hidden//16,),t.int32,True))
                a=cache[tag];counter=tensor((8,),t.int64,True);starts=tensor((iterations,),t.int64,True);ends=tensor((iterations,),t.int64,True)
                native=Grouped(*[a[nm].data_ptr()for nm in ('x','w','logits','h','partials','residual','out','tickets')],counter.data_ptr(),starts.data_ptr(),ends.data_ptr(),resource,iterations,copies,epsilon)
                result_tensor=a['h']if kind=='gate_up'else a['out']
                touched=[set()for _ in range(copies)]
                for i in range(iterations):touched[i%copies].update(active[i%64])
                addressed=sum(map(len,touched))*n*k*2
                assert addressed>=256*1024**2
                cold=dict(copies=copies,addressed_rotating_weight_bytes=addressed,allocated_weight_bytes=a['w'].numel()*2,
                    replay=key['replay'],instances=64,route_repetitions=2,active_mean=meta['replay_active_mean'],
                    RouteTable_charged=kind=='gate_up',down_initial_prefetch='after untimed RouteTable prepare barrier',
                    outgoing_prefetch='none: replay metadata changes on every iteration')
            else:
                n,k=key['n'],key['k'];copies=max(1,math.ceil(256*1024**2/(n*k*2)));iterations=max(128,2*copies+2)
                tag=(key['model'],b,kind,copies)
                if tag not in cache:
                    cache.clear();t.cuda.empty_cache()
                    # Random bytes are already in the candidate's packed page order.
                    cache[tag]=dict(x=tensor((b,k)),w=tensor((copies,n,k)),residual=tensor((b,n)),out=tensor((b,n)),
                        partial=tensor((max(3*n,104*256)*b,),t.float32),tickets=tensor((n//16,),t.int32,True),
                        head_partial=tensor((104,b,2),t.int32,True),head_tickets=tensor((1,),t.int32,True),tokens=tensor((b,1088),t.int32,True))
                a=cache[tag];counter=tensor((8,),t.int64,True);stamps=tensor((iterations,),t.int64,True)
                ops=Gemm(*[a[nm].data_ptr()for nm in ('x','w','residual','out','partial','tickets')],None,None,k,n,epsilon)
                native=Regular(ops,a['head_partial'].data_ptr(),a['head_tickets'].data_ptr(),a['tokens'].data_ptr(),counter.data_ptr(),stamps.data_ptr(),resource,iterations,copies)
                result_tensor=a['tokens']if kind=='head'else a['out']
                cold=dict(copies=copies,addressed_rotating_weight_bytes=copies*n*k*2,allocated_weight_bytes=a['w'].numel()*2,
                    packed_data='seeded BF16 random directly in candidate tile-page order')
            t.cuda.synchronize()
            status=function(C.byref(native),stream)
            if status:raise RuntimeError('stage launch failed '+str(status))
            t.cuda.synchronize()
            if grouped:
                times=(ends-starts).cpu().numpy()[64:128].astype(np.float64)
                assert len(times)==64
            else:
                times=np.diff(stamps.cpu().numpy()).astype(np.float64)[max(2,copies):]
            assert len(times)>0 and np.all(times>0)
            first=result_tensor.clone()
            counter.zero_()
            for nm in ('tickets','head_tickets'):
                if nm in a:a[nm].zero_()
            status=function(C.byref(native),stream);assert status==0;t.cuda.synchronize()
            if grouped and kind=='gate_up':
                # Inactive h rows are untouched in both repetitions.
                assert t.equal(first.view(t.int16),result_tensor.view(t.int16))
            else:
                assert t.equal(first.view(t.int16),result_tensor.view(t.int16))
            cost=float(times.mean())if grouped else float(np.median(times))
            emit(output,dict(common,status='pass',query_status=0,iterations=iterations,samples=len(times),mean_ns=float(times.mean()),
                median_ns=float(np.median(times)),cost_ns=cost,p90_ns=float(np.quantile(times,.9)),spread_ns=float(np.ptp(times)),
                repeat_bitwise=True,cost_statistic='mean of64 actual route instances'if grouped else'median warm-loop interval',**cold))
            print('MX6_COST_SAMPLE',kind,b,key.get('tn'),key.get('wn'),key.get('d'),cost,flush=True)


if __name__=='__main__':main()
