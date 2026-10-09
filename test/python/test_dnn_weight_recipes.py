"""DNN recipes against independent module expressions and checkpoint loading."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from tilemega.serving.plan import Buffer
from tilemega.serving.weights import _packed_cpu, _packed_gpu, _recipe_sources, load_weights


class Source:
    def __init__(self, values):
        self.values = values

    def tensor(self, name):
        return self.values[name]


class DnnRecipes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261009)
        torch.set_num_threads(2)

    def packed(self, recipe, values, dependencies=None):
        source = Source(values)
        cpu = _packed_cpu(recipe, source)
        self.assertTrue(torch.equal(cpu, _packed_gpu(recipe, source.tensor)))
        self.assertEqual(set(_recipe_sources(recipe)), set(values if dependencies is None else dependencies))
        return cpu

    def test_conv_bn(self):
        for channels, padded in [(3, 4), (3, 8), (16, 16), (24, 24)]:
            for bias in [False, True]:
                values = dict(w=(torch.randn(7, channels, 3, 3) * .08).bfloat16(),
                    gamma=(torch.rand(7) + .5).bfloat16(), beta=torch.randn(7).bfloat16(),
                    mean=torch.randn(7) * .2, variance=torch.rand(7) + .5)
                common = dict(gamma='gamma', beta='beta', mean='mean', variance='variance', epsilon=1e-5)
                if bias:
                    common['bias'] = 'b'; values['b'] = (torch.randn(7) * .1).bfloat16()
                recipe = dict(kind='conv_bn_fold', source='w', padded_channels=padded, **common)
                folded = self.packed(recipe, values, ['w', 'gamma', 'mean', 'variance'])
                self.assertEqual(tuple(folded.shape), (7, 3, 3, padded))
                self.assertTrue(torch.count_nonzero(folded[..., channels:]) == 0)
                bn = self.packed(dict(kind='bn_bias', **common), values,
                    ['gamma', 'beta', 'mean', 'variance'] + (['b'] if bias else []))
                self.assertEqual(bn.dtype, torch.float32)
                x = (torch.randn(2, channels, 7, 9) * .15).bfloat16().float()
                for stride in [1, 2]:
                    reference = F.batch_norm(F.conv2d(x, values['w'].float(),
                        values['b'].float() if bias else None, stride, 1),
                        values['mean'], values['variance'], values['gamma'].float(), values['beta'].float(),
                        training=False, eps=1e-5)
                    actual = F.conv2d(x, folded[..., :channels].permute(0, 3, 1, 2).float(), bn, stride, 1)
                    self.assertTrue(torch.all((actual-reference).abs() <= .016 + .016*reference.abs()))

    def test_deferred_layernorm(self):
        values = dict(w=(torch.randn(19, 24) * .15).bfloat16(),
            gamma=(torch.rand(24)+.5).bfloat16(), beta=(torch.randn(24)*.1).bfloat16(),
            bias=(torch.randn(19)*.1).bfloat16())
        base = dict(kind='fold_layernorm', source='w', gamma='gamma', beta='beta', bias='bias')
        parts = {name: self.packed(dict(base, part=name), values) for name in ['weight', 'u', 'v']}
        self.assertEqual(parts['weight'].dtype, torch.bfloat16)
        self.assertEqual(parts['u'].dtype, torch.float32)
        self.assertEqual(parts['v'].dtype, torch.float32)
        self.assertTrue(torch.equal(parts['u'], parts['weight'].float().sum(1)))
        unrounded = (values['w'].float()*values['gamma'].float()).sum(1)
        self.assertFalse(torch.equal(parts['u'], unrounded))
        x = torch.cat((torch.randn(11, 24), torch.full((1, 24), 2.0))).bfloat16().float()
        reference = F.linear(F.layer_norm(x, [24], values['gamma'].float(), values['beta'].float(), 1e-5),
                             values['w'].float(), values['bias'].float())
        mean, variance = x.mean(1, keepdim=True), x.var(1, unbiased=False, keepdim=True)
        actual = torch.rsqrt(variance+1e-5) * (F.linear(x, parts['weight'].float())-mean*parts['u'])+parts['v']
        self.assertTrue(torch.all((actual-reference).abs() <= .016 + .016*reference.abs()))
        torch.testing.assert_close(actual[-1], reference[-1], atol=2e-5, rtol=0)

    def test_qkv_and_gate(self):
        for part in ['weight', 'bias']:
            shape = (12, 13) if part == 'weight' else (12,)
            values = {name: (torch.randn(shape)*.15).bfloat16() for name in ['q', 'k', 'v']}
            packed = self.packed(dict(kind='qkv_concat_bias', part=part,
                sources=['q', 'k', 'v'], heads=3, head_dim=4), values)
            for head in range(3):
                for slot, name in enumerate(['q', 'k', 'v']):
                    self.assertTrue(torch.equal(packed[(head*3+slot)*4:(head*3+slot+1)*4],
                        values[name][head*4:(head+1)*4]))
            if part == 'weight':
                x = torch.randn(5, 13)
                independent = [F.linear(x, values[name].float()).reshape(5, 3, 4) for name in ['q', 'k', 'v']]
                expected = torch.stack(independent, dim=2).reshape(5, 36)
                torch.testing.assert_close(F.linear(x, packed.float()), expected, atol=1e-6, rtol=1e-6)
        for shape in [(64,), (64, 13), (64, 3, 3, 8)]:
            values = dict(w=torch.randn(shape).bfloat16())
            for unit in [1, 8, 16]:
                packed = self.packed(dict(kind='gate_pair_interleave', source='w', u=unit), values)
                first, second = values['w'].chunk(2)
                for block in range(32//unit):
                    self.assertTrue(torch.equal(packed[2*block*unit:(2*block+1)*unit], first[block*unit:(block+1)*unit]))
                    self.assertTrue(torch.equal(packed[(2*block+1)*unit:(2*block+2)*unit], second[block*unit:(block+1)*unit]))

    def test_checkpoint_dtype_and_nested_pages(self):
        values = dict(w=torch.randn(32, 16).bfloat16(), gamma=torch.rand(16).bfloat16(),
                      beta=torch.randn(16).bfloat16(), bias=torch.randn(32).bfloat16())
        base = dict(kind='fold_layernorm', source='w', gamma='gamma', beta='beta', bias='bias')
        recipes = dict(weight=dict(base, part='weight'), u=dict(base, part='u'),
                       v=dict(base, part='v'), bias=dict(kind='linear_bias', source='bias'),
                       legacy=dict(kind='alias', source='w'),
                       pages=dict(kind='tile_pages', tile_n=16, tile_k=16, source=dict(base, part='weight')))
        expected = {name: _packed_cpu(recipe, Source(values)) for name, recipe in recipes.items()}
        buffers = [Buffer(name, 1, 0 if value.dtype == torch.bfloat16 else 1, value.numel(), 0,
                          json.dumps(recipes[name])) for name, value in expected.items()]
        plan = SimpleNamespace(buffers=buffers)
        with tempfile.TemporaryDirectory() as directory:
            save_file(values, str(Path(directory)/'model.safetensors'))
            actual = load_weights(directory, plan, device='cpu')
        for name in expected:
            self.assertEqual(actual[name].dtype, expected[name].dtype)
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_invalid_geometry(self):
        source = Source(dict(w=torch.ones(18, 8).bfloat16(), gamma=torch.ones(8), beta=torch.zeros(8),
                             variance=torch.full((8,), -1.), mean=torch.zeros(8)))
        invalid = [dict(kind='gate_pair_interleave', source='w', u=8),
            dict(kind='fold_layernorm', source='w', gamma='gamma', beta='beta', part='missing'),
            dict(kind='conv_bn_fold', source='w', gamma='gamma', mean='mean', variance='variance',
                 epsilon=1e-5, padded_channels=8),
            dict(kind='qkv_concat_bias', sources=['w']*3, heads=2, head_dim=4, part='weight')]
        for recipe in invalid:
            with self.assertRaises(ValueError):
                _packed_cpu(recipe, source)


if __name__ == '__main__':
    unittest.main()
