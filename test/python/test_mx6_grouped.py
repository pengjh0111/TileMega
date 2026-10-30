"""Same-geometry V2/V3 expert intermediates and residuals, including repeats."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

pytestmark=pytest.mark.skipif(os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1' or
    not os.environ.get('MX6_GROUPED_SO'),reason='requires guard and MX6_GROUPED_SO')


class Args(C.Structure):
    _fields_=[(n,C.c_void_p)for n in('x','w','out','residual','combined','tickets','active',
        'count','rows','calls','table')]+[('units',C.c_uint)]


@pytest.mark.parametrize('batch',[1,16])
@pytest.mark.parametrize('experts',[64,128])
@pytest.mark.parametrize('pattern',['random','skew','spread'])
def test_expert_intermediates_bits(batch,experts,pattern):
    import torch as t
    from tilemega.serving.weights import _packed_gpu
    t.set_num_threads(1)
    g=t.Generator().manual_seed(6200+batch+experts)
    if pattern=='skew':route=t.randperm(experts,generator=g)[:8].repeat(batch,1)
    elif pattern=='spread':
        perm=t.randperm(experts,generator=g)
        route=perm[t.arange(batch*8).reshape(batch,8)%experts]
    else:route=t.stack([t.randperm(experts,generator=g)[:8]for _ in range(batch)])
    active=sorted(set(route.flatten().tolist()));counts=[];mapping=t.zeros(experts,batch,2,dtype=t.int32)
    weight=t.rand(batch,8,generator=g).bfloat16()
    weight=(weight.float()/weight.float().sum(-1,keepdim=True)).bfloat16()
    host_table=bytearray(2576)
    for a,e in enumerate(active):
        pairs=[(b,r)for b in range(batch)for r in range(8)if int(route[b,r])==e]
        counts.append(len(pairs));mapping[a,:len(pairs)]=t.tensor(pairs,dtype=t.int32)
        host_table[16+a]=e;host_table[144+a]=len(pairs)
        for j,(b,r)in enumerate(pairs):host_table[272+16*a+j]=(b<<3)|r
    host_table[2320:2320+batch*16]=bytes(weight.view(t.uint8).flatten().tolist())
    table=t.tensor(list(host_table),dtype=t.uint8,device='cuda')
    active_gpu=t.tensor(active,dtype=t.int32,device='cuda')
    count_gpu=t.tensor(counts,dtype=t.int32,device='cuda');rows_gpu=mapping.cuda()
    source=t.randn(batch,512,generator=g).bfloat16().cuda()
    residual=t.randn(batch,512,generator=g).bfloat16().cuda()
    cpu_w={gu:(t.randn(experts,512,512 if gu else 256,generator=g)*.04).bfloat16()
        for gu in(True,False)}
    library=Path(os.environ['MX6_GROUPED_SO']);lib=C.CDLL(str(library))
    assert lib.mx6_grouped_args_size()==C.sizeof(Args)
    fn=lib.mx6_grouped;fn.argtypes=[Args]+[C.c_int]*6+[C.c_void_p,C.POINTER(C.c_int),C.c_int];fn.restype=C.c_int
    stream=t.cuda.current_stream().cuda_stream;records=[]
    coords=[(32,1,1),(64,1,2),(32,2,2)]
    if batch==16 and len(active)*16>=104*7+1:coords.append((32,8,2))
    for tn,wn,depth in coords:
        assert wn<=(len(active)*(512//tn)+103)//104
        packed={gu:t.stack([_packed_gpu(dict(kind='tile_pages',layout='xcore_mma',
            tile_n=tn,tile_k=32,source=dict(kind='alias',source='matrix')),
            lambda _,e=e:cpu_w[gu][e])for e in range(experts)]).cuda()for gu in(True,False)}
        outputs=[];queries=[]
        for version in(2,3,3):
            h=t.full((experts,batch,256),-93,dtype=t.bfloat16,device='cuda')
            partial=t.full((batch,8,512),-93,dtype=t.bfloat16,device='cuda')
            merged=t.full_like(residual,-93);tickets=t.zeros(512//tn,dtype=t.int32,device='cuda')
            calls=t.zeros(104,8,dtype=t.int32,device='cuda')
            struct.pack_into('4I',host_table,0,len(active),len(active)*512//tn,len(active)*512//tn,0)
            table.copy_(t.tensor(list(host_table),dtype=t.uint8,device='cuda'))
            for gu in(True,False):
                calls.zero_();inp=source if gu else h;out=h if gu else partial
                args=Args(*(v.data_ptr()for v in(inp,packed[gu],out,residual,merged,tickets,
                    active_gpu,count_gpu,rows_gpu,calls,table)),len(active)*512//tn)
                resource=(C.c_int*4)()
                assert fn(args,batch,tn,wn,depth,version,int(gu),stream,resource,1)==0
                assert resource[1]==resource[2]==0 and resource[3]==1
                queries.append(dict(version=version,gate_up=gu,MTreg=resource[0],
                    local=resource[1],static=resource[2],resident=resource[3]))
                assert fn(args,batch,tn,wn,depth,version,int(gu),stream,None,0)==0
                t.cuda.synchronize();assert bool((calls==1).all())
                assert bool((tickets==0).all())
            outputs.append([x.cpu().view(t.uint8)for x in(h,partial,merged)])
        assert all(t.equal(a,b)for a,b in zip(outputs[0],outputs[1]))
        assert all(t.equal(a,b)for a,b in zip(outputs[1],outputs[2]))
        records.append(dict(batch=batch,experts=experts,pattern=pattern,tn=tn,wn=wn,depth=depth,
            n_active=len(active),max_count=max(counts),V2_V3_h_partials_x_bitwise=True,
            repeated_V3_bitwise=True,queries=queries,PID=os.getpid(),
            binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest()))
    if os.environ.get('MX6_GROUPED_OUT'):
        with Path(os.environ['MX6_GROUPED_OUT']).open('a')as f:
            for row in records:f.write(json.dumps(row)+'\n')
