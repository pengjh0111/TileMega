import copy
from pathlib import Path

import pytest
import torch

from tilemega.serving.export import Decoder, export
from tilemega.serving.state import rotary_tables


def configuration():
    from transformers import OlmoeConfig
    c=OlmoeConfig(hidden_size=128,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=4,vocab_size=256,
        num_experts=16,num_experts_per_tok=8,norm_topk_prob=False)
    c.architectures=['OlmoeForCausalLM'];c._attn_implementation='eager'
    c._experts_implementation='grouped_mm'
    return c


def test_olmoe_matches_hf_with_full_row_norm_and_unnormalized_routing():
    from transformers import OlmoeForCausalLM
    torch.set_num_threads(2);torch.manual_seed(2031)
    c=configuration();hf=OlmoeForCausalLM(c).bfloat16().eval()
    ours=Decoder(c.to_dict()).bfloat16().eval()
    with torch.no_grad():
        for name,p in hf.named_parameters():
            p.copy_(1+.1*torch.randn_like(p) if 'norm.weight' in name else .03*torch.randn_like(p))
    ours.load_state_dict(hf.state_dict(),strict=True)
    ids=torch.tensor([[1,5,7,2],[1,4,8,3]])
    with torch.inference_mode():
        cos,sin=rotary_tables(c,4,torch.device('cpu'))
        empty=torch.empty(2,4,0,32,dtype=torch.bfloat16)
        actual=ours.execute(ids,(empty,empty),cos,sin)[0]
        expected=hf(ids,use_cache=False).logits
        _,weights,_=hf.model.layers[0].mlp.gate(torch.randn(7,128).bfloat16())
    assert ours.model.layers[0].self_attn.row_qk_norm
    assert ours.model.layers[0].self_attn.q_norm.weight.shape==(128,)
    assert not ours.model.layers[0].mlp.normalize
    assert torch.all(weights.float().sum(-1)<.95)
    torch.testing.assert_close(actual.float(),expected.float(),atol=.02,rtol=.02)


def test_olmoe_export_preserves_row_weights_and_moe_boundary(tmp_path):
    c=configuration().to_dict()
    export(c,'decode',128,tmp_path)
    p=torch.export.load(tmp_path/'exported_program.pt2')
    assert p.state_dict['model.layers.0.self_attn.q_norm.weight'].shape==(128,)
    assert p.state_dict['model.layers.0.self_attn.k_norm.weight'].shape==(128,)
    assert p.state_dict['model.layers.0.mlp.experts.gate_up_proj'].shape==(16,128,128)
    assert all(x.device.type=='meta'for x in p.state_dict.values())
    assert sum(str(n.target)=='tilemega.moe_experts.default'for n in p.graph.nodes)==1


@pytest.mark.parametrize('change',[{'clip_qkv':1.0},{'attention_bias':True}])
def test_olmoe_rejects_attention_semantics_not_implemented(change):
    c=configuration().to_dict();c.update(change)
    with pytest.raises(ValueError,match='unclipped, bias-free'):
        Decoder(c)


def test_structure_signature_reports_row_normalization():
    from tilemega.serving.structure_check import inspect
    from tilemega.serving.workload import Workload
    result=inspect(configuration().to_dict(),(Workload(64,16,batches=(1,)),))
    assert result['attention_signature']['qk_norm'] is True
    assert result['attention_signature']['qk_norm_scope']=='row'
