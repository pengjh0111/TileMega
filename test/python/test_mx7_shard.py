"""CPU packing contracts: reconstruct TP1 and retain its exact arithmetic."""
import copy
import json
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch

import torch

from tilemega.serving.export import Decoder, metadata_only, validate_tp
from tilemega.serving.weights import _packed_cpu, _packed_gpu, _shard, load_weights
from tilemega.serving.plan import Buffer


class Checkpoint:
    def __init__(self, tensors):
        self.tensor = tensors.__getitem__


class ShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.weights = {name: torch.randn(*shape).to(torch.bfloat16) for name, shape in (
            ('q', (64, 32)), ('k', (32, 32)), ('v', (32, 32)), ('gate', (64, 32)),
            ('up', (64, 32)), ('o', (32, 64)), ('down', (32, 64)), ('head', (64, 32)))}
        cls.weights['gamma'] = torch.randn(32).to(torch.bfloat16)
        cls.recipes = [
            dict(kind='alias', source='o'),
            dict(kind='alias', source='down'),
            dict(kind='alias', source='head'),
            dict(kind='qkv_group_interleave', sources=['q', 'k', 'v'], hkv=8, qperkv=2, head_dim=4),
            dict(kind='gate_up_interleave', sources=['gate', 'up'], u=16),
        ]
        text = subprocess.check_output(['git', 'show', '6cab45b96:python/tilemega/serving/weights.py'],
                                       text=True, cwd=Path(__file__).resolve().parents[2])
        cls.old = types.ModuleType('tilemega.serving._mx7_reference_weights')
        cls.old.__package__ = 'tilemega.serving'
        exec(compile(text, '<MX7 TP1 frozen weights>', 'exec'), cls.old.__dict__)

    def test_each_recipe_reconstructs_the_frozen_tp1(self):
        for world in (2, 4):
            for recipe in self.recipes:
                axis = 1 if recipe.get('source') in ('o', 'down') else 0
                reference = self.old._packed_gpu(recipe, self.weights.__getitem__)
                parts = []
                for rank in range(world):
                    local = copy.deepcopy(recipe)
                    local['shard'] = dict(axis=axis, parts=world)
                    if local['kind'] == 'qkv_group_interleave':
                        local['hkv'] //= world
                    value = _packed_gpu(local, self.weights.__getitem__, rank)
                    self.assertTrue(torch.equal(value, _packed_cpu(local, Checkpoint(self.weights), rank)))
                    parts.append(value)
                self.assertTrue(torch.equal(torch.cat(parts, dim=axis), reference), (world, recipe))

    def test_recursive_fold_and_pages_match_tp1_packing_after_slice(self):
        for world in (2, 4):
            for rank in range(world):
                alias = dict(kind='alias', source='head', shard=dict(axis=0, parts=world))
                recipe = dict(kind='tile_pages', layout='xcore_mma', tile_n=16, tile_k=32,
                              source=dict(kind='fold_rmsnorm', norm='gamma', source=alias))
                expected_recipe = copy.deepcopy(recipe)
                del expected_recipe['source']['source']['shard']
                tensors = dict(self.weights)
                tensors['head'] = _shard(tensors['head'], alias['shard'], rank)
                expected = self.old._packed_gpu(expected_recipe, tensors.__getitem__)
                self.assertTrue(torch.equal(_packed_gpu(recipe, self.weights.__getitem__, rank), expected))
                self.assertTrue(torch.equal(_packed_cpu(recipe, Checkpoint(self.weights), rank), expected))

    def test_all_unsharded_paths_match_frozen_implementation(self):
        recipes = copy.deepcopy(self.recipes)
        recipes += [dict(kind='fold_rmsnorm', norm='gamma', source=r) for r in self.recipes
                    if r.get('source') not in ('o', 'down')]
        recipes += [dict(kind='tile_pages', layout='xcore_mma', tile_n=16, tile_k=32, source=r)
                    for r in copy.deepcopy(recipes)]
        for recipe in recipes:
            self.assertTrue(torch.equal(_packed_gpu(recipe, self.weights.__getitem__),
                                        self.old._packed_gpu(recipe, self.weights.__getitem__)))
            self.assertTrue(torch.equal(_packed_cpu(recipe, Checkpoint(self.weights)),
                                        self.old._packed_cpu(recipe, Checkpoint(self.weights))))

    def test_invalid_shards(self):
        for shard, rank in [(dict(axis=0, parts=3), 0), (dict(axis=1, parts=2), 2),
                            (dict(axis=2, parts=2), 0)]:
            with self.assertRaises(ValueError):
                _shard(self.weights['head'], shard, rank)

    def test_loader_requires_matching_rank_and_world(self):
        recipe = dict(kind='alias', source='head', shard=dict(axis=0, parts=2))
        plan = types.SimpleNamespace(buffers=[Buffer('head', 1, 0, 32 * 32, 0, json.dumps(recipe))])
        checkpoint = Checkpoint(self.weights)
        checkpoint.tensor_sha = lambda name: name
        with patch('tilemega.serving.weights.Checkpoint', return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, 'requires tp'):
                load_weights('unused', plan, device='cpu')
            with self.assertRaisesRegex(ValueError, 'differs from TP'):
                load_weights('unused', plan, device='cpu', tp=(0, 4))
            for rank in (0, 1):
                output = load_weights('unused', plan, device='cpu', tp=(rank, 2))
                self.assertTrue(torch.equal(output['head'], self.weights['head'][rank * 32:(rank + 1) * 32]))

    def test_export_dimensions_and_tied_head(self):
        base = dict(architectures=['Qwen3ForCausalLM'], hidden_size=32, intermediate_size=64,
                    num_attention_heads=8, num_key_value_heads=4, head_dim=4,
                    num_hidden_layers=1, vocab_size=64, tie_word_embeddings=True)
        for world in (2, 4):
            config = dict(base, tilemega_tp=world)
            validate_tp(config, world)
            with torch.device('meta'):
                model = Decoder(config)
            self.assertEqual(model.model.layers[0].self_attn.q_proj.weight.shape, (32 // world, 32))
            self.assertEqual(model.model.layers[0].self_attn.k_proj.weight.shape, (16 // world, 32))
            self.assertEqual(model.model.layers[0].mlp.down_proj.weight.shape, (32, 64 // world))
            self.assertEqual(model.lm_head.weight.shape, (64 // world, 32))
            self.assertEqual(model.model.embed_tokens.weight.shape, (64, 32))
            self.assertIsNot(model.lm_head.weight, model.model.embed_tokens.weight)
            self.assertTrue(metadata_only(config))
        with self.assertRaises(ValueError):
            validate_tp(dict(base, architectures=['OlmoeForCausalLM']), 2)
        with self.assertRaises(ValueError):
            validate_tp(dict(base, vocab_size=63), 2)


if __name__ == '__main__':
    unittest.main()
