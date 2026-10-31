#!/usr/bin/env python3
"""Fresh-process RouteTable prologue diagnostic, before and after P1b."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics

from tilemega.build.identity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libraries',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--round',type=int,required=True);a=p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':p.error('requires mx_gpu_job.sh')
    import torch
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for b,e in((1,64),(1,128),(16,64),(16,128)):
        g=torch.Generator().manual_seed(6600+b+e)
        logits=torch.randn(b,e,generator=g).bfloat16().cuda();sink=torch.empty(104,dtype=torch.int32,device='cuda')
        for mode in('serial','both','weights','ordinal'):
            so=a.libraries/(mode+'.so');lib=C.CDLL(str(so));fn=lib.mx6_route_bench
            fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p,C.POINTER(C.c_int),C.c_int];fn.restype=C.c_int
            stream=torch.cuda.current_stream().cuda_stream;r=(C.c_int*4)()
            assert fn(logits.data_ptr(),sink.data_ptr(),b,e,2000,stream,r,1)==0
            assert fn(logits.data_ptr(),sink.data_ptr(),b,e,100,stream,None,0)==0
            torch.cuda.synchronize();samples=[]
            for _ in range(7):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();assert fn(logits.data_ptr(),sink.data_ptr(),b,e,2000,stream,None,0)==0
                end.record();end.synchronize();samples.append(start.elapsed_time(end)*1e6/2000)
            rows.append(dict(round=a.round,PID=os.getpid(),batch=b,experts=e,variant=mode,
                median_ns=statistics.median(samples),p90_ns=float(torch.quantile(torch.tensor(samples,dtype=torch.float64),.9)),
                spread_ns=max(samples)-min(samples),samples_ns=samples,MTreg=r[0],stack=r[1],static_smem=r[2],resident=r[3],
                binary_sha256=sha(so),contract='bf16',threads=512,ctas=104,executor='repeated_prologue',
                metadata_cache='hot identical logits shared by all CTAs',sink_store_per_iteration=True,timing_eligible=False))
            print('ROUTE_PROLOGUE',a.round,b,e,mode,rows[-1]['median_ns'],flush=True)
    (a.out/'result.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
