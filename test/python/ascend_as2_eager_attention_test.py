"""The independent eager adapter preserves GQA heads and causal mask meaning."""
import types
import unittest
from unittest.mock import patch
import torch
from tilemega.bench.as2_eager_attention import fia_v2_forward


class EagerAttentionContract(unittest.TestCase):
    def test_gqa_three_uses_unexpanded_kv_and_blocked_mask(self):
        query=torch.zeros((1,24,64,128),dtype=torch.bfloat16)
        kv=torch.zeros((1,8,1088,128),dtype=torch.bfloat16)
        blocked=torch.arange(1088)[None,:]>torch.arange(64)[:,None]
        mask=torch.zeros((1,1,64,1088),dtype=torch.bfloat16).masked_fill(blocked,torch.finfo(torch.bfloat16).min)
        def operator(q,k,v,**kwargs):
            self.assertEqual(k.shape[1],8)
            self.assertEqual(kwargs['num_query_heads'],24)
            self.assertEqual(kwargs['num_key_value_heads'],8)
            self.assertTrue(torch.equal(kwargs['atten_mask'][0,0],blocked))
            self.assertEqual(kwargs['input_layout'],'BNSD')
            return q,None
        with patch('tilemega.bench.as2_eager_attention.torch_npu.npu_fused_infer_attention_score_v2',operator):
            out,_=fia_v2_forward(types.SimpleNamespace(training=False),query,kv,kv,mask,scaling=.125)
        self.assertEqual(out.shape,(1,64,24,128))

    def test_unknown_mask_or_dropout_never_falls_back(self):
        q=torch.zeros((1,16,1,128),dtype=torch.bfloat16)
        kv=torch.zeros((1,8,1088,128),dtype=torch.bfloat16)
        module=types.SimpleNamespace(training=False)
        with self.assertRaises(ValueError):fia_v2_forward(module,q,kv,kv,None)
        with self.assertRaises(ValueError):fia_v2_forward(module,q,kv,kv,None,dropout=.1)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
