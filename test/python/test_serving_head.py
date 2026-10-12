"""Untied vocabulary heads retain their FQN and checkpoint tensor."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from tilemega.serving.export import export
from tilemega.serving.weights import Checkpoint


class ServingHead(unittest.TestCase):
    def test_dense_export_uses_configured_head_in_both_phases(self):
        config = dict(hidden_size=32, intermediate_size=64,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      num_hidden_layers=1, vocab_size=64, rms_norm_eps=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            for architecture in ('LlamaForCausalLM', 'Qwen3ForCausalLM'):
                for tied in (False, True):
                    for phase in ('prefill', 'decode'):
                        with self.subTest(architecture=architecture, tied=tied, phase=phase):
                            out = Path(temporary)/f'{architecture}_{tied}_{phase}'
                            export(dict(config, architectures=[architecture],
                                        tie_word_embeddings=tied), phase, 128, out)
                            program = torch.export.load(out/'exported_program.pt2')
                            names = {item.arg.name: item.target for item in
                                     program.graph_signature.input_specs
                                     if item.kind.name == 'PARAMETER'}
                            head = [node for node in program.graph.nodes
                                    if node.target == torch.ops.aten.linear.default and
                                    'lm_head' in str(node.meta.get('nn_module_stack'))]
                            self.assertEqual(len(head), 1)
                            self.assertEqual(names[head[0].args[1].name],
                                             'model.embed_tokens.weight' if tied
                                             else 'lm_head.weight')

    def test_prefill_export_uses_request_sequence_length(self):
        config=dict(architectures=['LlamaForCausalLM'],hidden_size=32,
                    intermediate_size=64,num_attention_heads=4,num_key_value_heads=2,
                    head_dim=8,num_hidden_layers=1,vocab_size=64,rms_norm_eps=1e-6,
                    tie_word_embeddings=True)
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            export(config,'prefill',2176,out,prompt_len=128)
            program=torch.export.load(out/'exported_program.pt2')
            user_inputs=[node for node in program.graph.nodes if node.op=='placeholder'
                         and node.name=='input_ids']
            self.assertEqual(len(user_inputs),1)
            self.assertEqual(user_inputs[0].meta['val'].shape[1],128)

    def test_checkpoint_does_not_substitute_untied_head(self):
        embedding = torch.full((64, 32), 1.0, dtype=torch.bfloat16)
        head = torch.full_like(embedding, 2.0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'config.json').write_text(json.dumps({'tie_word_embeddings': False}))
            save_file({'model.embed_tokens.weight': embedding, 'lm_head.weight': head},
                      directory/'model.safetensors')
            checkpoint = Checkpoint(directory)
            self.assertTrue(torch.equal(checkpoint.tensor('lm_head.weight'), head))
            self.assertFalse(torch.equal(checkpoint.tensor('lm_head.weight'), embedding))
            save_file({'model.embed_tokens.weight': embedding}, directory/'model.safetensors')
            with self.assertRaisesRegex(KeyError, 'lm_head.weight'):
                Checkpoint(directory).tensor('lm_head.weight')
            (directory/'config.json').write_text(json.dumps({'tie_word_embeddings': True}))
            self.assertTrue(torch.equal(Checkpoint(directory).tensor('lm_head.weight'), embedding))
            (directory/'config.json').unlink()
            with self.assertRaisesRegex(KeyError, 'lm_head.weight'):
                Checkpoint(directory).tensor('lm_head.weight')


if __name__ == '__main__':
    unittest.main()
