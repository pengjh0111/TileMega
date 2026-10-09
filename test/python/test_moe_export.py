"""Original-FQN MoE exports and the opaque experts' HF numerical contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tilemega.moe.export import Region, SparseMLP, export_region, moe_experts
from tilemega.serving.export import Decoder, export


def config():
    return dict(architectures=['Qwen3MoeForCausalLM'], model_type='qwen3_moe',
        hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=2, num_experts=4, num_experts_per_tok=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        rms_norm_eps=1e-6, rope_theta=1e6, norm_topk_prob=True,
        tie_word_embeddings=False, hidden_act='silu')


class MoeExport(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(940)

    def test_reference_matches_hf_grouped_mm(self):
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        c = Qwen3MoeConfig(**config())
        c._experts_implementation = 'grouped_mm'
        hf = Qwen3MoeSparseMoeBlock(c).eval().bfloat16()
        ours = SparseMLP(config()).eval().bfloat16()
        with torch.no_grad():
            for parameter in hf.parameters():
                parameter.copy_(torch.randn_like(parameter)*.125)
            ours.gate.weight.copy_(hf.gate.weight)
            ours.experts.load_state_dict(hf.experts.state_dict(), strict=True)
            for tokens in (1, 3, 17, 64):
                x = torch.randn(1, tokens, 32).bfloat16()
                expected, actual = hf(x), ours(x)
                self.assertTrue(torch.equal(expected, actual), (tokens, (expected-actual).abs().max()))
        with self.assertRaisesRegex(ValueError, 'index outside'):
            moe_experts(x[0], torch.full((64, 2), 4), torch.ones(64, 2).bfloat16(),
                        ours.experts.gate_up_proj, ours.experts.down_proj)

    def test_region_preserves_fqns_and_dynamic_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_region(config(), 1, temporary)
            program = torch.export.load(Path(temporary)/'exported_program.pt2')
            names = {item.target for item in program.graph_signature.input_specs
                     if item.kind.name == 'PARAMETER'}
            self.assertEqual(names, {'model.layers.1.post_attention_layernorm.weight',
                'model.layers.1.mlp.gate.weight', 'model.layers.1.mlp.experts.gate_up_proj',
                'model.layers.1.mlp.experts.down_proj'})
            targets = [str(node.target) for node in program.graph.nodes]
            self.assertEqual(targets.count('tilemega.moe_experts.default'), 1)
            self.assertEqual(manifest['tokens_range'], [1, 4096])
            self.assertTrue(all(value.device.type == 'meta' for value in program.state_dict.values()))
            constraint = next(iter(program.range_constraints.values()))
            self.assertEqual((int(constraint.lower), int(constraint.upper)), (1, 4096))
            for tokens in (1, 4096):
                output = program.module()(torch.empty(tokens, 32, device='meta', dtype=torch.bfloat16))
                self.assertEqual(tuple(output.shape), (tokens, 32))

    def test_serving_phases_preserve_untied_head_and_qk_norm(self):
        with tempfile.TemporaryDirectory() as temporary:
            for phase in ('prefill', 'decode'):
                directory = Path(temporary)/phase
                export(config(), phase, 128, directory)
                program = torch.export.load(directory/'exported_program.pt2')
                names = {item.target for item in program.graph_signature.input_specs
                         if item.kind.name == 'PARAMETER'}
                self.assertIn('lm_head.weight', names)
                self.assertIn('model.embed_tokens.weight', names)
                self.assertIn('model.layers.0.self_attn.q_norm.weight', names)
                self.assertIn('model.layers.1.mlp.experts.down_proj', names)
                targets = [str(node.target) for node in program.graph.nodes]
                self.assertEqual(targets.count('tilemega.moe_experts.default'), 2)
                self.assertTrue(all(value.device.type == 'meta' for value in program.state_dict.values()))
        model = Decoder(config())
        self.assertIsNot(model.lm_head.weight, model.model.embed_tokens.weight)
        dense = copy.deepcopy(config()); dense['architectures'] = ['Qwen3ForCausalLM']
        model = Decoder(dense)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_rotary_matches_hf(self):
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
        from tilemega.serving.state import rotary_tables
        c = Qwen3MoeConfig(**config())
        actual = rotary_tables(c, 67, torch.device('cpu'))
        module = Qwen3MoeRotaryEmbedding(c).to('cpu')
        reference = module(torch.empty(1, 1, 1, 8).bfloat16(), torch.arange(67)[None])
        for value, expected in zip(actual, reference):
            self.assertTrue(torch.equal(value, expected[0]))


if __name__ == '__main__':
    unittest.main()
