"""The explicit grouped-mm repair implements ragged BF16 matmul in fullgraph."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark=pytest.mark.skipif(os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1',reason='C550 GPU guard required')


def test_ragged_grouped_mm_and_compiled_graph(tmp_path):
    import torch
    from tilemega.serving.metax_grouped_mm import grouped_mm
    torch.manual_seed(2033)
    rows=[]
    for e,k,n,s,skew in [(4,16,32,7,False),(64,128,96,128,False),
                         (128,2048,1536,8,False),(64,1024,2048,128,False),
                         (4,64,64,64,True)]:
        a=torch.randn(s,k,device='cuda',dtype=torch.bfloat16)
        weight=torch.randn(e,n,k,device='cuda',dtype=torch.bfloat16).transpose(1,2)
        ids=torch.zeros(s,dtype=torch.long) if skew else torch.randint(e,(s,))
        counts=ids.bincount(minlength=e)
        offsets=counts.cumsum(0).to(torch.int32).cuda().contiguous()
        expected=torch.empty(s,n,device='cuda',dtype=torch.bfloat16)
        start=0
        for expert,end in enumerate(counts.cumsum(0).tolist()):
            if end>start:expected[start:end]=(a[start:end].float()@weight[expert].float()).bfloat16()
            start=end
        actual=grouped_mm(a,weight,offsets)
        torch.testing.assert_close(actual.float(),expected.float(),atol=.04,rtol=.02)
        compiled=torch.compile(lambda x,w,o:grouped_mm(x,w,o)+0,fullgraph=True,mode='reduce-overhead')
        for _ in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            result=compiled(a,weight,offsets)
            torch.testing.assert_close(result.float(),expected.float(),atol=.04,rtol=.02)
        torch.cuda.synchronize()
        rows.append(dict(experts=e,k=k,n=n,rows=s,skew=skew,max_abs=float((actual.float()-expected.float()).abs().max()),fullgraph=True))
        del a,weight,expected,actual,result,compiled
        torch._dynamo.reset();torch.cuda.empty_cache()
    (tmp_path/'grouped_mm_results.json').write_text(json.dumps(rows,indent=2)+'\n')
