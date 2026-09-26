#!/usr/bin/env python3
"""Compare page-fed attention with FP32 torch and explicit BF16 boundaries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch


def check(path):
    data=path.read_bytes();d,q,norm,past,extent,cap=np.frombuffer(data[:24],dtype=np.int32).tolist()
    at=24
    def read(n):
        nonlocal at
        bits=np.frombuffer(data[at:at+n*2],dtype=np.uint16).copy();at+=n*2
        return torch.from_numpy(bits).view(torch.bfloat16).float()
    qkv=read((q+2)*d).reshape(q+2,d);key=read(cap*d).reshape(cap,d);value=read(cap*d).reshape(cap,d)
    cosine,sine,qw,kw=read(d),read(d),read(d),read(d)
    actual=read(q*d).reshape(q,d)
    bf=lambda x:x.to(torch.bfloat16).float()
    def rotate(x,w):
        if norm:
            x=bf(bf(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))*w)
        half=torch.cat((-x[...,d//2:],x[...,:d//2]),dim=-1)
        return bf(bf(x*cosine)+bf(half*sine))
    queries=rotate(qkv[:q],qw);newkey=rotate(qkv[q],kw)
    def ordered(x):
        bits=x.to(torch.bfloat16).view(torch.int16).to(torch.int32)&65535
        return torch.where((bits&32768)!=0,65535-bits,32768+bits)
    cache_ulp=int((ordered(key[past])-ordered(newkey)).abs().max())
    value_equal=bool(torch.equal(value[past],qkv[q+1]))
    key[past]=newkey
    scores=queries@key[:past+1].T/(d**0.5)
    probability=torch.exp(scores-scores.max(-1,keepdim=True).values)
    reference=bf(probability)@value[:past+1]/probability.sum(-1,keepdim=True)
    error=float((reference-actual).abs().max());limit=float(reference.abs().max())/128+1/256
    return dict(case=path.stem,max_abs_error=error,tolerance=limit,cache_max_ulp=cache_ulp,
                cache_value_equal=value_equal,passed=error<=limit and cache_ulp<=1 and value_equal)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    dumps=a.out/'dumps'
    with open(os.environ.get('TILEMEGA_GPU_LOCK','/root/r10_work/serving_gpu.lock'),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (a.out/'cuda.log').open('w') as log:
            subprocess.run([str(a.binary)],env={**os.environ,'TILEMEGA_TEST_DUMP':str(dumps)},
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    rows=[check(p) for p in sorted(dumps.glob('*.bin'))]
    result=dict(cases=len(rows),passed=sum(r['passed'] for r in rows),rows=rows,
                reference='torch FP32 matmul; BF16 norm/RoPE and unnormalized P; <=1 BF16 ULP cache K')
    (a.out/'torch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return 0 if len(rows)==60 and all(r['passed'] for r in rows) else 1

if __name__=='__main__':raise SystemExit(main())
