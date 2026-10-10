"""Header-only recipe checks; no tensor library or model weights are loaded."""
import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tilemega.moe.checkpoints import expected_tensors, index_check


class MoeRecipeMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(model_type='qwen3_moe', hidden_size=32, moe_intermediate_size=32,
            num_experts=4, num_experts_per_tok=2, num_hidden_layers=1, vocab_size=64,
            num_attention_heads=4, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)
        (self.root/'config.json').write_text(json.dumps(self.config))
        header = {}; stop = 0
        for name, shape in expected_tensors(self.config).items():
            past = stop;stop += 2*math.prod(shape)
            header[name] = dict(dtype='BF16', shape=shape, data_offsets=[past, stop])
        raw = json.dumps(header).encode()
        with (self.root/'shard.safetensors').open('wb') as stream:
            stream.write(struct.pack('<Q', len(raw)));stream.write(raw)
            stream.truncate(8+len(raw)+stop)
        (self.root/'model.safetensors.index.json').write_text(json.dumps(dict(
            metadata=dict(total_size=stop), weight_map={name:'shard.safetensors' for name in header})))
        self.bytes = stop
        self.norm = 'model.layers.0.post_attention_layernorm.weight'
        self.stack = dict(kind='expert_stack', prefix='model.layers.0.mlp.experts.',
            experts=4, hidden=32, intermediate=32, u=16, part='gate_up')
        self.packed = dict(kind='tile_pages', tile_n=32, tile_k=16,
            source=dict(kind='fold_rmsnorm', norm=self.norm, source=self.stack))
        self.qkv = dict(kind='qkv_group_interleave', hkv=1, qperkv=4, head_dim=8,
            sources=['model.layers.0.self_attn.'+part+'_proj.weight' for part in ('q','k','v')])

    def test_exact_shapes_and_packed_extents(self):
        recipes = dict(experts=self.packed, qkv=self.qkv,
            tiled_qkv=dict(kind='tile_pages', tile_n=32, tile_k=16, source=self.qkv))
        result = index_check(self.root, recipes, self.bytes,
            recipe_elements=dict(experts=8192, qkv=1536, tiled_qkv=2048))
        self.assertEqual(result['recipe_shapes'], dict(experts=[8192],qkv=[48,32],tiled_qkv=[2048]))
        self.assertIn(self.norm, result['recipe_sources'])
        # The expert packer normalizes before tiling for either wrapper order.
        opposite = dict(kind='fold_rmsnorm',norm=self.norm,source=dict(
            kind='tile_pages',tile_n=32,tile_k=16,source=self.stack))
        self.assertEqual(index_check(self.root,dict(e=opposite))['recipe_shapes']['e'],[8192])
        down = dict(self.stack,part='down')
        self.assertEqual(index_check(self.root,dict(down=down))['recipe_shapes']['down'],[4,32,32])

    def test_geometry_corruption_is_rejected_before_loading(self):
        for key, value in [('hidden',16),('intermediate',16),('experts',2),('u',True),('u',0)]:
            recipe = copy.deepcopy(self.packed);recipe['source']['source'][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                index_check(self.root,dict(e=recipe))
        for tn,tk in [(16,16),(32,256),(0,16),(True,16)]:
            recipe = dict(self.packed,tile_n=tn,tile_k=tk)
            with self.subTest(tile=(tn,tk)),self.assertRaises(ValueError):
                index_check(self.root,dict(e=recipe))
        for change in [dict(qperkv=8),dict(hkv=2),dict(head_dim=16),
                dict(hkv=2,qperkv=4,head_dim=4),dict(sources=self.qkv['sources'][:2])]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                index_check(self.root,dict(q=dict(self.qkv,**change)))
        for recipe in [dict(kind='fold_rmsnorm',norm='model.layers.0.self_attn.q_norm.weight',source=self.stack),
                dict(kind='fold_rmsnorm',norm=self.norm,source=dict(self.stack,part='down')),
                dict(kind='tile_pages',tile_n=32,tile_k=16,source=self.packed),
                dict(kind='fold_rmsnorm',norm=self.norm,source=dict(kind='tile_pages',tile_n=32,tile_k=16,source=self.qkv)),
                dict(kind='unknown',source=self.stack)]:
            with self.assertRaises(ValueError):index_check(self.root,dict(e=recipe))

    def test_manifest_size_and_source_corruption_are_rejected(self):
        for extents in [dict(e=8191),dict(e=8192.,),{},dict(e=8192,extra=1)]:
            with self.subTest(extents=extents),self.assertRaises(ValueError):
                index_check(self.root,dict(e=self.packed),recipe_elements=extents)
        with self.assertRaises(ValueError):index_check(self.root,recipe_elements=dict(e=1))
        with self.assertRaises(ValueError):index_check(self.root,dict(e=dict(kind='alias',source='missing')))


if __name__ == '__main__':
    unittest.main()
