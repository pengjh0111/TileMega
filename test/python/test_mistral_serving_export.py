from pathlib import Path

import pytest
import torch


def test_mistral_decoder_matches_hf():
    from transformers import MistralConfig, MistralForCausalLM
    from tilemega.serving.export import Decoder

    config = MistralConfig(hidden_size=128, intermediate_size=256,
                           num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, vocab_size=256,
                           sliding_window=None, rope_theta=1000000.)
    config._attn_implementation = 'eager'
    torch.manual_seed(20260925)
    hf = MistralForCausalLM(config).to(torch.bfloat16).eval()
    d = config.to_dict()
    d['architectures'] = ['MistralForCausalLM']
    native = Decoder(d).to(torch.bfloat16).eval()
    native.load_state_dict(hf.state_dict(), strict=True)
    tokens = torch.tensor([[1, 5, 7, 2], [1, 4, 8, 3]])
    positions = torch.arange(4).unsqueeze(0)
    with torch.inference_mode():
        cosine, sine = hf.model.rotary_emb(torch.zeros(1, dtype=torch.bfloat16), positions)
        empty = torch.empty(2, 2, 0, 32, dtype=torch.bfloat16)
        result = native.execute(tokens, (empty, empty), cosine[0], sine[0])[0]
        reference = hf(tokens, use_cache=False).logits
    torch.testing.assert_close(result.float(), reference.float(), atol=.02, rtol=.02)


def test_export_rejects_olmoe_instead_of_emitting_dense_graph(tmp_path):
    from tilemega.serving.export import export
    with pytest.raises(ValueError, match='unsupported.*OlmoeForCausalLM'):
        export({'architectures': ['OlmoeForCausalLM']}, 'decode', 1088, tmp_path)


def test_export_rejects_unimplemented_mistral_sliding_window(tmp_path):
    from tilemega.serving.export import export
    with pytest.raises(ValueError, match='sliding_window=null'):
        export({'architectures': ['MistralForCausalLM'], 'sliding_window': 4096},
               'decode', 1088, tmp_path)
