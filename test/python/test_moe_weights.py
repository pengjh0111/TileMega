"""Expert packing, shared deployment weights and bounded source lifetimes."""
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref

import torch
from safetensors.torch import save_file
from tilemega.serving.plan import Buffer
from tilemega.serving.weights import _packed_cpu, _packed_gpu, _recipe_sources, load_weights


class Source:
    def __init__(self, values):
        self.values = values

    def tensor(self, name):
        return self.values[name]


class MoeWeights(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(931)
        self.prefix = 'model.layers.0.mlp.experts.'
        self.values = {self.prefix+f'{e}.{part}_proj.weight': torch.randn(shape).bfloat16()
            for e in range(5) for part,shape in [('gate',(64,128)),('up',(64,128)),('down',(128,64))]}
        self.norm = 'model.layers.0.post_attention_layernorm.weight'
        self.values[self.norm] = torch.rand(128).bfloat16()

    def recipe(self, part, tiled=None, folded=False):
        value = dict(kind='expert_stack', prefix=self.prefix, part=part,
                     experts=5, hidden=128, intermediate=64, u=16)
        if folded:
            value = dict(kind='fold_rmsnorm', source=value, norm=self.norm)
        if tiled:
            value = dict(kind='tile_pages', source=value, tile_n=tiled[0], tile_k=tiled[1])
        return value

    def reference(self, part, folded=False):
        matrices = []
        for expert in range(5):
            stem = self.prefix+f'{expert}.'
            if part == 'gate_up':
                gate,up = (self.values[stem+kind+'_proj.weight'] for kind in ('gate','up'))
                matrix = torch.cat([block for first in range(0,64,16)
                    for block in (gate[first:first+16],up[first:first+16])])
            else:
                matrix = self.values[stem+'down_proj.weight']
            if folded:
                matrix = (matrix.float()*self.values[self.norm].float()).bfloat16()
            matrices.append(matrix)
        return torch.stack(matrices)

    def untile(self, packed, tn, tk, n, k):
        result = torch.empty((5,n,k), dtype=torch.bfloat16)
        tile = packed.reshape(5,n//tn,k//tk,tn*tk)
        for row in range(tn):
            for col in range(tk):
                if tk < 64:
                    address = (row%8)*tk+col
                    physical = (row//8)*8*tk+(address ^ (((address//64)%(tk//8))*8))
                else:
                    physical = ((col//64)*(tn//8)+row//8)*512+(row%8)*64+((col//8)%8 ^ row%8)*8+col%8
                result[:,row::tn,col::tk] = tile[:,:,:,physical]
        return result

    def test_all_layouts_and_fold(self):
        for part in ('gate_up','down'):
            for folded in (False,True) if part == 'gate_up' else (False,):
                expected = self.reference(part,folded)
                for geometry in (None,(32,16),(32,32),(64,64),(128,64)):
                    recipe = self.recipe(part,geometry,folded)
                    actual = _packed_cpu(recipe,Source(self.values))
                    self.assertTrue(torch.equal(actual,_packed_gpu(recipe,self.values.__getitem__)))
                    if geometry:
                        actual = self.untile(actual,*geometry,*expected.shape[1:])
                    self.assertTrue(torch.equal(actual,expected),(part,folded,geometry))
                    sources = {self.prefix+f'{e}.{kind}_proj.weight' for e in range(5)
                        for kind in (('gate','up') if part=='gate_up' else ('down',))}
                    if folded:
                        sources.add(self.norm)
                    self.assertEqual(set(_recipe_sources(recipe)),sources)
        actual = _packed_cpu(self.recipe('gate_up',(32,128)),Source(self.values))
        self.assertTrue(torch.equal(self.untile(actual,32,128,128,128),self.reference('gate_up')))

    def test_expert_source_lifetime(self):
        for part in ('gate_up','down'):
            live = [];maximum = 0;calls = []
            def source(name):
                nonlocal maximum
                gc.collect()
                tensor = self.values[name].clone();calls.append(name)
                live.append(weakref.ref(tensor))
                maximum = max(maximum,sum(item() is not None for item in live))
                return tensor
            actual = _packed_gpu(self.recipe(part,(32,32),part=='gate_up'),source)
            expected = self.reference(part,part=='gate_up')
            self.assertTrue(torch.equal(self.untile(actual,32,32,*expected.shape[1:]),expected))
            self.assertLessEqual(maximum,2)
            self.assertEqual(len(calls),11 if part=='gate_up' else 5)
            gc.collect();self.assertTrue(all(item() is None for item in live))

    def test_shared_deployment_and_legacy_aliases(self):
        recipes = dict(gate_up=self.recipe('gate_up',(32,32),True), down=self.recipe('down',(32,32)),
            router=dict(kind='fold_rmsnorm', norm=self.norm,
                source=dict(kind='alias',source='model.layers.0.mlp.gate.weight')),
            embedding=dict(kind='alias',source='model.embed_tokens.weight'))
        values = dict(self.values, **{'model.layers.0.mlp.gate.weight':torch.randn(5,128).bfloat16(),
                                      'model.embed_tokens.weight':torch.randn(19,128).bfloat16()})
        expected = {name:_packed_cpu(recipe,Source(values)) for name,recipe in recipes.items()}
        buffers = [Buffer(name,1,0,value.numel(),0,json.dumps(recipes[name]))
                   for name,value in expected.items()]
        # Both phases point at the same recipe and storage; an additional name
        # with the same recipe is also deduplicated by content and recipe.
        alias = Buffer('gate_up_copy',1,0,expected['gate_up'].numel(),0,json.dumps(recipes['gate_up']))
        prefill = SimpleNamespace(buffers=buffers)
        decode = SimpleNamespace(buffers=buffers+[alias])
        with tempfile.TemporaryDirectory() as directory:
            save_file(values,str(Path(directory)/'model.safetensors'))
            actual = load_weights(directory,prefill,decode,device='cpu')
            wrong = Buffer('gate_up',1,0,expected['gate_up'].numel(),0,json.dumps(self.recipe('gate_up')))
            with self.assertRaisesRegex(ValueError,'plans disagree'):
                load_weights(directory,prefill,SimpleNamespace(buffers=[wrong]),device='cpu')
        for name,value in expected.items():
            self.assertTrue(torch.equal(actual[name],value),name)
        self.assertEqual(actual['gate_up'].data_ptr(),actual['gate_up_copy'].data_ptr())

    def test_invalid_expert_recipes(self):
        leaf = self.recipe('gate_up')
        invalid = [dict(leaf,experts=0),dict(leaf,u=17),dict(leaf,prefix='missing-dot'),dict(leaf,part='other'),
            self.recipe('gate_up',(16,32)),self.recipe('gate_up',(32,24)),
            self.recipe('down',(32,32),True),self.recipe('gate_up',(256,32))]
        for recipe in invalid:
            with self.assertRaises(ValueError):
                _packed_cpu(recipe,Source(self.values))
        for name in (self.prefix+'0.gate_proj.weight',self.norm):
            values = dict(self.values);values[name] = values[name].float()
            with self.assertRaises(ValueError):
                _packed_cpu(self.recipe('gate_up',folded=True),Source(values))


if __name__ == '__main__':
    unittest.main()
