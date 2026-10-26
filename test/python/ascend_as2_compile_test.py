"""CPU compilation-audit contracts; these do not prove NPU compilation."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config
from transformers.cache_utils import StaticCache
from tilemega.bench.as2_compile import CompileAudit,NpuStaticCache


class CompileContracts(unittest.TestCase):
    def test_graph_runtime_rejects_eager_fallback_and_timed_capture(self):
        class Graph:
            def __init__(self):self._graphs_meta={};self.fallback_to_eager=False;self.config={}
            def compile(self,key):self._graphs_meta[key]=True;return key
        with tempfile.TemporaryDirectory(dir='/home/d50064638/junhuip/tilemega-as2/tmp') as temp:
            audit=CompileAudit(None,Path(temp)/'compile',{'name':'CPU_TEST_ONLY'})
            audit.guard_graph_runtime(Graph);graph=Graph();graph.compile('prefill')
            audit.in_timed_request=True;graph.compile('prefill')
            with self.assertRaises(RuntimeError):graph.compile('decode')
            audit.in_timed_request=False;graph.fallback_to_eager=True
            with self.assertRaises(RuntimeError):graph.compile('prefill')

    def test_scatter_cache_matches_hf_boundaries_and_request_reset(self):
        torch.set_num_threads(1)
        config=Qwen3Config(hidden_size=1024,num_attention_heads=16,num_key_value_heads=8,
            head_dim=128,num_hidden_layers=1)
        for batch in (1,16):
            kw=dict(config=config,max_batch_size=batch,max_cache_len=1088,device='cpu',dtype=torch.bfloat16)
            expected=StaticCache(**kw);actual=NpuStaticCache(**kw)
            def scatter(data,indices,updates,axis):return data.index_copy_(axis,indices,updates)
            with patch.object(torch.ops.npu,'scatter_update_',side_effect=scatter,create=True):
                for epoch in range(2):
                    for position in (0,63,64,65,127,128,129,1023,1024,1087):
                        values=torch.full((batch,8,1,128),float(position+epoch),dtype=torch.bfloat16)
                        args=(values,-values,0,{'cache_position':torch.tensor([position])})
                        x=actual.update(*args);y=expected.update(*args)
                        self.assertTrue(all(torch.equal(a,b) for a,b in zip(x,y)))
                    actual.reset();expected.reset()
                    self.assertEqual(int(torch.count_nonzero(actual.key_cache[0])),0)

    def test_rejects_timed_recompile_and_preserves_backend_failure(self):
        with tempfile.TemporaryDirectory(dir='/home/d50064638/junhuip/tilemega-as2/tmp') as temp:
            graph=types.SimpleNamespace(code='def forward(x): return x + 1\n')
            audit=CompileAudit(lambda gm,args: object(),Path(temp)/'compile',{'name':'CPU_TEST_ONLY'},max_graphs=2)
            with self.assertRaises(RuntimeError):audit.check_timed(0)
            audit(graph,[]);audit.check_timed(1);audit(graph,[])
            with self.assertRaises(RuntimeError):audit.check_timed(1)
            with self.assertRaises(RuntimeError):audit(graph,[])
            def fail(gm,args):raise ValueError('compiler failed')
            broken=CompileAudit(fail,Path(temp)/'failed',{'name':'CPU_TEST_ONLY'})
            with self.assertRaises(ValueError):broken(graph,[])
            self.assertEqual(broken.graphs[0]['status'],'failed')
            self.assertIn('compiler failed',(Path(temp)/'failed/compile.json').read_text())


if __name__=='__main__':unittest.main()
