"""Compare native TileRange M4 logits at identical arithmetic geometries."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or
    not os.environ.get('MX6_ROUTER_LOGITS_SO'),reason='guarded MX6 router-logits fixture required')


class Dispatch(C.Structure):
    _fields_=[(n,C.c_void_p) for n in ('logits','n_active','active','row_count','rows','weights','u_gate_up','u_down')]


class Router(C.Structure):
    _fields_=[(n,C.c_void_p) for n in ('a','w','residual','out','partial','tickets','a_tokens','residual_tokens')]+[
        ('a_stride',C.c_int),('residual_stride',C.c_int),('epsilon',C.c_float),
        ('router_ticket',C.c_void_p),('dispatch',Dispatch)]


@pytest.mark.parametrize('batch',[1,16])
@pytest.mark.parametrize('experts',[64,128])
@pytest.mark.parametrize('depth',[1,2,3,4])
def test_tile_range_logits_bits(batch,experts,depth):
    import torch
    from tilemega.serving.weights import _packed_gpu
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(6100+batch+experts)
    cpu_x=torch.randn(batch,2048,generator=g).bfloat16()
    cpu_w=(torch.randn(experts,2048,generator=g)/2048**.5).bfloat16()
    x=cpu_x.cuda()
    w=_packed_gpu(dict(kind='tile_pages',layout='xcore_mma',tile_n=16,tile_k=32,
        source=dict(kind='alias',source='matrix')),lambda _:cpu_w).cuda()
    lib=C.CDLL(os.environ['MX6_ROUTER_LOGITS_SO'])
    assert lib.mx6_router_operands_size()==C.sizeof(Router)
    fn=lib.mx6_router_logits
    fn.argtypes=[Router]+[C.c_int]*4+[C.c_void_p,C.POINTER(C.c_int),C.c_int]
    fn.restype=C.c_int
    stream=torch.cuda.current_stream().cuda_stream
    outputs=[];queries=[]
    for version in (2,3,3):
        logits=torch.full((batch,experts),-93,dtype=torch.bfloat16,device='cuda')
        ints=[torch.zeros(n,dtype=torch.int32,device='cuda') for n in
              (1,experts,experts,experts*batch*2,1,1,experts//16,16)]
        na,active,count,rows,ug,ud,tickets,router_ticket=ints
        weights=torch.empty(batch,8,dtype=torch.bfloat16,device='cuda')
        dispatch=Dispatch(*(z.data_ptr() for z in (logits,na,active,count,rows,weights,ug,ud)))
        ops=Router(x.data_ptr(),w.data_ptr(),None,logits.data_ptr(),None,tickets.data_ptr(),None,None,
                   2048,0,1e-6,router_ticket.data_ptr(),dispatch)
        resources=(C.c_int*4)()
        assert fn(ops,batch,experts,depth,version,stream,resources,1)==0
        assert resources[1]==resources[2]==0 and resources[3]==1
        assert fn(ops,batch,experts,depth,version,stream,None,0)==0
        torch.cuda.synchronize()
        outputs.append(logits.cpu().view(torch.uint8))
        queries.append(dict(version=version,MTreg=resources[0],local=resources[1],
                            static=resources[2],resident=resources[3]))
        assert torch.all(tickets==0) and torch.all(router_ticket==0)
    assert torch.equal(outputs[0],outputs[1])
    assert torch.equal(outputs[1],outputs[2])
    if os.environ.get('MX6_ROUTER_LOGITS_OUT'):
        with Path(os.environ['MX6_ROUTER_LOGITS_OUT']).open('a') as f:
            f.write(json.dumps(dict(batch=batch,experts=experts,depth=depth,queries=queries,
                v2_logits_bitwise=True,repeated_v3_bitwise=True,PID=os.getpid(),
                binary_sha256=hashlib.sha256(Path(os.environ['MX6_ROUTER_LOGITS_SO']).read_bytes()).hexdigest()))+'\n')
