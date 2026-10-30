"""V2/V3 table equivalence for 360 cases per contract; run with a guarded fixture."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1' or not os.environ.get('MX6_ROUTE_TABLE_SO'),
    reason='guarded MX6 route-table fixture required')


class Operands(C.Structure):
    _fields_ = [(name,C.c_void_p) for name in
               ('logits','n_active','active','row_count','rows','weights','u_gate_up','u_down')]


@pytest.mark.parametrize('batch',[1,2,4,8,16])
@pytest.mark.parametrize('experts',[64,128])
@pytest.mark.parametrize('normalize',[0,1])
@pytest.mark.parametrize('fp32',[0,1])
def test_v2_table_bits_and_repeated_build(batch,experts,normalize,fp32):
    import torch
    torch.set_num_threads(1)
    cases = 360
    generator = torch.Generator().manual_seed(6000+batch+experts)
    cpu = torch.randn(cases,batch,experts,generator=generator).bfloat16()
    cpu[0].zero_()
    cpu[1] = torch.tensor([0.,-0.]).repeat(batch,experts//2).bfloat16()
    for ulp in (0,1,2):
        row = torch.full((batch,experts),-1.,dtype=torch.bfloat16)
        row[:,:7] = 2.
        row[:,7] = 1.
        row[:,8] = torch.tensor([0x3f80-ulp],dtype=torch.int16).view(torch.bfloat16)[0]
        cpu[2+ulp] = row
    cpu[5] = torch.arange(experts).remainder(11).expand(batch,-1).bfloat16()
    # The last 104 CTAs build the same input independently, as in M5.
    cpu[-104:] = cpu[6:7]
    logits = cpu.cuda()
    lib = C.CDLL(os.environ['MX6_ROUTE_TABLE_SO'])
    fn = lib.mx6_route_table
    fn.argtypes = [Operands]+[C.c_int]*6+[C.c_void_p,C.POINTER(C.c_int),C.c_int]
    fn.restype = C.c_int
    stream = torch.cuda.current_stream().cuda_stream
    outputs,queries = [],[]
    for version in (2,3,3):
        tensors = [logits, torch.zeros(cases,dtype=torch.int32,device='cuda'),
            torch.full((cases,experts),-1,dtype=torch.int32,device='cuda'),
            torch.full((cases,experts),-1,dtype=torch.int32,device='cuda'),
            torch.full((cases,experts,batch,2),-1,dtype=torch.int32,device='cuda'),
            torch.zeros(cases,batch,8,dtype=torch.float32 if fp32 else torch.bfloat16,device='cuda'),
            torch.zeros(cases,dtype=torch.int32,device='cuda'),
            torch.zeros(cases,dtype=torch.int32,device='cuda')]
        operands = Operands(*(x.data_ptr() for x in tensors))
        args = (operands,batch,experts,normalize,fp32,version,cases)
        resources = (C.c_int*4)()
        assert fn(*args,stream,resources,1) == 0
        assert resources[1] == resources[2] == 0 and resources[3] == 1
        assert fn(*args,stream,None,0) == 0
        torch.cuda.synchronize()
        outputs.append([x.cpu().view(torch.uint8) for x in tensors[1:]])
        queries.append(dict(version=version,MTreg=resources[0],local=resources[1],
                            static=resources[2],resident=resources[3]))
    assert all(torch.equal(x,y) for x,y in zip(outputs[0],outputs[1]))
    assert all(torch.equal(x,y) for x,y in zip(outputs[1],outputs[2]))
    record = dict(batch=batch,experts=experts,normalize=normalize,fp32=fp32,cases=cases,
        independent_inputs=cases-103,replicated_ctas=104,queries=queries,v2_bitwise=True,
        repeated_v3_bitwise=True,PID=os.getpid(),
        binary_sha256=hashlib.sha256(Path(os.environ['MX6_ROUTE_TABLE_SO']).read_bytes()).hexdigest())
    if os.environ.get('MX6_ROUTE_TABLE_OUT'):
        with Path(os.environ['MX6_ROUTE_TABLE_OUT']).open('a') as f:
            f.write(json.dumps(record)+'\n')
