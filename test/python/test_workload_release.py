"""Release comparisons reject missing weights and changed execution sidecars."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tilemega.bench.arms import check_binaries, register

spec = importlib.util.spec_from_file_location('release_driver',
    Path(__file__).resolve().parents[2]/'scripts/check_serving_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseInputs(unittest.TestCase):
    def shard(self, path):
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [4], 'data_offsets': [0,8]}}).encode()
        path.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*8)

    def test_partial_download_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            (root/'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map':{'a':'a.safetensors','b':'b.safetensors'}}))
            self.shard(root/'b.safetensors')
            with self.assertRaisesRegex(ValueError,'missing local checkpoint shard'):
                release.checkpoint_files(root)
            self.shard(root/'a.safetensors')
            self.assertEqual(len(release.checkpoint_files(root)),2)
            (root/'a.safetensors').write_bytes((root/'a.safetensors').read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete safetensors data'):
                release.checkpoint_files(root)

    def test_auto_execution_sidecar_is_part_of_registered_input(self):
        with tempfile.TemporaryDirectory() as name:
            so=Path(name)/'plan.so'; so.write_bytes(b'not-loaded')
            serving=Path(str(so)+'.serving.json')
            serving.write_text(json.dumps({'decode_mode':'L1','decode_loop':0}))
            arm=register(dict(kind='tm',prefill=str(so),decode=str(so)))
            check_binaries(arm)
            serving.write_text(json.dumps({'decode_mode':'L2','decode_loop':1}))
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                check_binaries(arm)

    def test_byte_model_counts_query_width_and_actual_decode_trajectory(self):
        cfg=dict(hidden_size=8,intermediate_size=16,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,head_dim=4,vocab_size=32)
        weights=2*(8*8*4+4*4*8+3*8*16)+8*32
        self.assertEqual(release.step_bytes(cfg,dict(isl=4,osl=2,batch=3)),
                         2*weights+4*3*2*2*4*5)
        # Extra output extends the historical KV trajectory; it does not
        # multiply the once-per-decode-step weight traffic.
        self.assertEqual(release.step_bytes(cfg,dict(isl=4,osl=4,batch=3))-
                         release.step_bytes(cfg,dict(isl=4,osl=2,batch=3)),4*3*2*2*4)


if __name__=='__main__':unittest.main()
