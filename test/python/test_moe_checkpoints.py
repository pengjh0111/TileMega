"""Original expert-FQN resharding, validation and unchanged HF loading."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from tilemega.moe.checkpoints import expected_tensors, index_check, rewrite


class MoeCheckpoints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(841);torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name);self.source = self.root/'source';self.source.mkdir()
        self.config = Qwen3MoeConfig(hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2, vocab_size=64, tie_word_embeddings=False)
        self.config.save_pretrained(self.source)
        self.values = {name:torch.randn(shape).bfloat16()
                       for name,shape in expected_tensors(self.config.to_dict()).items()}
        mapping = {};total = 0
        for index in range(3):
            tensors = {name:value for n,(name,value) in enumerate(self.values.items()) if n%3 == index}
            shard = f'shard-{index}.safetensors';save_file(tensors, str(self.source/shard))
            mapping.update({name:shard for name in tensors});total += sum(t.numel()*2 for t in tensors.values())
        (self.source/'model.safetensors.index.json').write_text(json.dumps(dict(
            metadata=dict(total_size=total), weight_map=mapping)))

    def test_exact_original_index(self):
        result = index_check(self.source, expected_bytes=sum(t.numel()*2 for t in self.values.values()))
        self.assertEqual(set(result['tensors']), set(self.values))
        full = dict(self.config.to_dict(), hidden_size=2048, moe_intermediate_size=768,
            num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
            num_experts=128, num_experts_per_tok=8, vocab_size=151936)
        full.pop('num_local_experts', None)
        expected = expected_tensors(full)
        self.assertEqual(len(expected), 18867)
        self.assertEqual(sum(2*torch.Size(shape).numel() for shape in expected.values()), 61064245248)

    def check_rewrite(self, name, **options):
        output = self.root/name
        result = rewrite(self.source, output, shard_bytes=8192, **options)
        self.assertGreater(len(result['shards']), 1)
        config = json.loads((output/'config.json').read_text())
        checked = index_check(output)
        self.assertEqual(checked['total_bytes'], result['total_bytes'])
        for key,item in checked['tensors'].items():
            with safe_open(output/item['shard'], framework='pt', device='cpu') as stream:
                actual = stream.get_tensor(key)
            reference = self.values[key]
            if key.endswith('mlp.gate.weight'):
                reference = reference[:config['num_experts']]
            self.assertTrue(torch.equal(actual,reference), key)
        # The conversion mapping is HF's own, including original expert names.
        model = Qwen3MoeForCausalLM.from_pretrained(output, dtype=torch.bfloat16,
            attn_implementation='sdpa', experts_implementation='grouped_mm').eval()
        self.assertEqual(len(model.model.layers), config['num_hidden_layers'])
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertNotEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())
        self.assertTrue(torch.equal(model.lm_head.weight, self.values['lm_head.weight']))
        for layer_index,layer in enumerate(model.model.layers):
            prefix = f'model.layers.{layer_index}.'
            self.assertTrue(torch.equal(layer.mlp.gate.weight,
                self.values[prefix+'mlp.gate.weight'][:config['num_experts']]))
            for expert in range(config['num_experts']):
                stem = prefix+f'mlp.experts.{expert}.'
                self.assertTrue(torch.equal(layer.mlp.experts.gate_up_proj[expert],
                    torch.cat([self.values[stem+'gate_proj.weight'],self.values[stem+'up_proj.weight']])) )
                self.assertTrue(torch.equal(layer.mlp.experts.down_proj[expert],self.values[stem+'down_proj.weight']))
        return output

    def test_truncate(self):
        output = self.check_rewrite('truncated', layers=1)
        config = json.loads((output/'config.json').read_text())
        self.assertEqual((config['num_hidden_layers'],config['num_experts']), (1,4))
        with self.assertRaises(ValueError):
            rewrite(self.source, output, layers=1)

    def test_expert_slice(self):
        output = self.check_rewrite('sliced', experts=2)
        config = json.loads((output/'config.json').read_text())
        self.assertEqual((config['num_hidden_layers'],config['num_experts'],config['num_experts_per_tok']), (2,2,2))

    def test_invalid_index_and_geometry(self):
        for options in [dict(layers=0),dict(layers=3),dict(experts=1),dict(experts=5),dict(),dict(layers=1,experts=2)]:
            with self.assertRaises(ValueError):
                rewrite(self.source,self.root/'invalid',**options)
        path = self.source/'model.safetensors.index.json';original = path.read_text()
        value = json.loads(original);value['metadata']['total_size'] += 2;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'total_size'):
            index_check(self.source)
        path.write_text(original);value = json.loads(original);value['weight_map'].pop(next(iter(value['weight_map'])))
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'header and checkpoint index'):
            index_check(self.source)


if __name__ == '__main__':
    unittest.main()
