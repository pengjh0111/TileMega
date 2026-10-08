"""CPU checks for streaming HF weights, token sources and routing statistics."""
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from tilemega.moe.routing_profile import (
    IndexedCheckpoint, load_hf_layer, make_queue, routing_statistics, token_sequences,
)


class RoutingProfileTest(unittest.TestCase):
    def test_histograms_match_independent_windows(self):
        rows = [[0, 1], [0, 2], [2, 3], [2, 0], [1, 3], [0, 3], [1, 2], [3, 1]]
        result = routing_statistics([rows], 4, 2, (1, 2, 4, 8))
        for t in (1, 2, 4, 8):
            windows = [rows[i:i + t] for i in range(0, len(rows), t)]
            counts = [[sum(e in row for row in window) for e in range(4)] for window in windows]
            expected = [sum(n > 0 for n in window) for window in counts]
            self.assertEqual(result[str(t)]['expected_distinct_experts'], sum(expected) / len(expected))
            for e, histogram in enumerate(result[str(t)]['tokens_per_expert_histograms']):
                observed = [window[e] for window in counts]
                self.assertEqual(histogram, {str(n): observed.count(n) for n in set(observed)})
                self.assertEqual(sum(histogram.values()), len(windows))
        zeros = routing_statistics([[[0, 1]] * 4], 4, 2, (4,))['4']
        self.assertEqual(zeros['tokens_per_expert_histograms'][2], {'0': 1})

    def test_routing_rejects_invalid_observations(self):
        for values, experts, k, tokens in (
            ([], 4, 2, (1,)), ([[[0, 0]]], 4, 2, (1,)),
            ([[[0, 4]]], 4, 2, (1,)), ([[[-1, 1]]], 4, 2, (1,)),
            ([[[0, 1]]], 1, 2, (1,)), ([[[0, 1]]], 4, 2, (2,)),
            ([[[0, 1]]], 4, 2, (1, 1)), ([[[0, 1, 2]]], 4, 2, (1,)),
            ([[[0.5, 1.0]]], 4, 2, (1,)),
        ):
            with self.subTest(values=values, experts=experts, k=k, tokens=tokens), self.assertRaises(ValueError):
                routing_statistics(values, experts, k, tokens)

    def test_original_prompts_and_non_repeated_wiki_tokens(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.add_special_tokens = add_special_tokens
                return list(map(int, text.split()))
            def __len__(self):
                return 32
        tokenizer = Tokenizer()
        self.assertEqual(token_sequences([[1, 2], [3]], tokenizer, ['4 5', '6 7 8'], 4),
                         [[1, 2, 4, 5], [3, 6, 7, 8]])
        self.assertFalse(tokenizer.add_special_tokens)
        with self.assertRaises(ValueError):
            token_sequences([[1]], tokenizer, ['4'], 4)
        with self.assertRaises(ValueError):
            token_sequences([[32]], tokenizer, [], 1)

    def test_original_expert_shards_reproduce_unchanged_hf_layer(self):
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
        torch.manual_seed(401)
        config = Qwen3MoeConfig(hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2, vocab_size=32)
        # CPU tests compare loading and HF layer semantics; production fixes grouped_mm.
        config._attn_implementation = 'sdpa'
        config._experts_implementation = 'eager'
        model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
        original = {}
        for name, value in model.state_dict().items():
            if name.endswith('mlp.experts.gate_up_proj'):
                prefix = name.removesuffix('gate_up_proj')
                for e, weight in enumerate(value):
                    gate, up = weight.chunk(2, dim=0)
                    original[prefix + f'{e}.gate_proj.weight'] = gate.contiguous()
                    original[prefix + f'{e}.up_proj.weight'] = up.contiguous()
            elif name.endswith('mlp.experts.down_proj'):
                prefix = name.removesuffix('down_proj')
                for e, weight in enumerate(value):
                    original[prefix + f'{e}.down_proj.weight'] = weight.contiguous()
            else:
                original[name] = value.contiguous()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shards = [{}, {}]
            index = {}
            for i, (name, value) in enumerate(original.items()):
                shard = i % 2
                shards[shard][name] = value.clone()
                index[name] = f'shard{shard}.safetensors'
            for i, values in enumerate(shards):
                save_file(values, str(directory / f'shard{i}.safetensors'))
            (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
            config.save_pretrained(directory)
            checkpoint = IndexedCheckpoint(directory)
            hidden = torch.randn(1, 8, 16).to(torch.bfloat16)
            positions = torch.arange(8).unsqueeze(0)
            rotary = Qwen3MoeRotaryEmbedding(config)
            with torch.inference_mode():
                for i, reference in enumerate(model.model.layers):
                    loaded = load_hf_layer(checkpoint, config, i, 'cpu')
                    for name, parameter in reference.named_parameters():
                        self.assertTrue(torch.equal(parameter, dict(loaded.named_parameters())[name]), name)
                    args = dict(position_ids=positions, position_embeddings=rotary(hidden, positions), use_cache=False)
                    expected = reference(hidden, **args)
                    actual = loaded(hidden, **args)
                    self.assertTrue(torch.equal(actual, expected), f'layer {i}')
                    hidden = expected
            self.assertEqual(len(checkpoint.used), sum(name.startswith('model.layers.') for name in original))
            missing = dict(index)
            missing['bad'] = '../outside.safetensors'
            (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': missing}))
            with self.assertRaises(ValueError):
                IndexedCheckpoint(directory)

    def test_queue_serializes_layers_and_locks_gpu_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'num_hidden_layers': 48}))
            args = Namespace(checkpoint=root, work=root / 'work', prompts=root / 'ids.json',
                wiki=root / 'wiki.parquet', output=root / 'profile.json',
                capture_layers=[0, 24, 47], device='cuda:0', lock=root / 'gpu.lock', queue=root / 'queue')
            make_queue(args)
            queue = json.loads((args.queue / 'queue_all.json').read_text())
            self.assertEqual(len(queue), 51)
            self.assertEqual(len({row['name'] for row in queue}), 51)
            for i, row in enumerate(queue):
                if i:
                    self.assertEqual(row['after'], [queue[i - 1]['name']])
                stage = row['command'][row['command'].index('--stage') + 1]
                if stage in ('embed', 'layer'):
                    self.assertEqual(row['command'][:2], ['flock', str(args.lock)])
                else:
                    self.assertNotEqual(row['command'][0], 'flock')
                self.assertIn('PYTHONPATH', row['env'])


if __name__ == '__main__':
    unittest.main()
