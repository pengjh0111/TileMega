import ctypes as C
import json
import os
from pathlib import Path

import pytest

pytestmark=pytest.mark.skipif(os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or not os.environ.get('MX5_OLMOE_DISPATCH_SO'),reason='guarded OLMoE dispatch fixture required')


class Operands(C.Structure):
    _fields_=[(name,C.c_void_p)for name in ('logits','n_active','active','row_count','rows','weights','u_gate_up','u_down')]


def test_full_softmax_mass_deterministic_dispatch(tmp_path):
    import torch
    torch.set_num_threads(2);torch.manual_seed(2110)
    lib=C.CDLL(os.environ['MX5_OLMOE_DISPATCH_SO']);f=lib.mx5_olmoe_dispatch;f.argtypes=[Operands,C.c_int,C.c_void_p];f.restype=C.c_int
    reports=[]
    for b in (1,16):
        for mode in ('uniform','random','tied'):
            logits=torch.randn(b,64).bfloat16()
            if mode=='uniform':logits.zero_()
            if mode=='tied':logits=(torch.arange(64)%11).expand(b,-1).bfloat16()
            index=logits.float().argsort(dim=-1,descending=True,stable=True)[:,:8]
            expected=logits.float().softmax(-1).gather(1,index).bfloat16()
            gpu=logits.cuda();active=torch.full((64,),-1,dtype=torch.int32,device='cuda');counts=torch.zeros_like(active);rows=torch.full((64,b,2),-1,dtype=torch.int32,device='cuda');weights=torch.empty(b,8,dtype=torch.bfloat16,device='cuda');n=torch.zeros(1,dtype=torch.int32,device='cuda');u=torch.zeros(2,dtype=torch.int32,device='cuda')
            ops=Operands(gpu.data_ptr(),n.data_ptr(),active.data_ptr(),counts.data_ptr(),rows.data_ptr(),weights.data_ptr(),u.data_ptr(),u[1:].data_ptr());observed=[]
            for _ in range(2):
                assert f(ops,b,C.c_void_p(torch.cuda.current_stream().cuda_stream))==0
                torch.cuda.synchronize();observed.append([t.cpu().clone()for t in (active,counts,rows,weights,n,u)])
            assert all(torch.equal(x,y)for x,y in zip(*observed))
            assert torch.equal(observed[0][3],expected),(b,mode,(observed[0][3]-expected).abs().max())
            chosen=sorted(set(index.flatten().tolist()));assert n.item()==len(chosen)
            assert active[:len(chosen)].cpu().tolist()==chosen
            for ordinal,expert in enumerate(chosen):
                positions=[(token,rank)for token in range(b)for rank in range(8)if index[token,rank]==expert]
                assert counts[ordinal].item()==len(positions)
                assert rows[ordinal,:len(positions)].cpu().tolist()==[list(p)for p in positions]
            assert u.cpu().tolist()==[len(chosen)*32,len(chosen)*32]
            if mode=='uniform':assert torch.equal(weights.cpu().sum(-1),torch.full((b,),.125,dtype=torch.bfloat16))
            reports.append(dict(batch=b,mode=mode,active=len(chosen),bitwise_weights=True,deterministic=True))
    (tmp_path/'dispatch_results.json').write_text(json.dumps(reports,indent=2)+'\n')
