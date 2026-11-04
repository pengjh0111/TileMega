#!/usr/bin/env python3
"""Prepare an independent CPU BF16 reference; never initialize the NPU."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(out):
    from tilemega.bench.as2_models import RUNS,controlled
    import numpy as np
    import torch
    out=controlled(out,[RUNS])
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    shapes=[(1,128,1024,1024,1024),(16,128,1024,1024,1024),
            (3,97,144,160,176),(16,17,32,48,64)]
    generator=torch.Generator(device='cpu').manual_seed(20261016)
    files=[];cases=[]
    for i,(rows,columns,ls,rs,os) in enumerate(shapes):
        left=torch.randn((4,rows,columns),generator=generator).to(torch.bfloat16)
        right=torch.randn((4,rows,columns),generator=generator).to(torch.bfloat16)
        # Cancellation, BF16 halfway rounding and signed zero use the same
        # independent FP32-add/output-cast definition as ordinary inputs.
        left[0,0,:8]=torch.tensor([0.,-0.,1.,1.,-1.,256.,-256.,80.],dtype=torch.bfloat16)
        right[0,0,:8]=torch.tensor([-0.,-0.,1./256,1./128,1.,1.,-1.,-80.],dtype=torch.bfloat16)
        expected=(left.float()+right.float()).to(torch.bfloat16)
        for name,data,stride in [('left',left,ls),('right',right,rs),('expected',expected,os)]:
            payload=np.full(4*rows*stride+(256 if name=='expected' else 0),0x7fc1,dtype=np.uint16)
            payload[:4*rows*stride].reshape(4,rows,stride)[:,:,:columns]=data.view(torch.uint16).numpy()
            raw=payload.tobytes();p=out/f'{i}_{name}.bin';p.write_bytes(raw)
            files.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        cases.append(dict(rows=rows,columns=columns,left_stride=ls,right_stride=rs,output_stride=os))
    result=dict(schema='tilemega.as2.residual_probe_inputs.v1',seed=20261016,
        reference='independent CPU torch: BF16 inputs -> FP32 add -> BF16 output',
        torch_version=torch.__version__,workers=4,epochs=4,cases=cases,files=files,
        status='prepared_not_run',model_admitted=False,performance_admitted=False)
    (out/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();result=prepare(args.out)
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),files=len(result['files']))))
