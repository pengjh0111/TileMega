"""CPU checks for geometry classes which must not be silently approximated."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('forward_primitives',
    Path(__file__).resolve().parents[2]/'scripts/dnn1_forward_primitive_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RequiredKeys(unittest.TestCase):
    def pool_inputs(self, root, row):
        inventory = root/'inventory.json'
        layouts = root/'layouts.json'
        inventory.write_text(json.dumps({'rows': [row]}))
        layouts.write_text(json.dumps([dict(model='resnet18', batch=b,
            producer_stage=21, typed_output_stride=512) for b in (1,16)]))
        return inventory, layouts

    def test_cross_image_dense_tile_keeps_real_representative(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            row=dict(batch=1, model='resnet18', pool_producer_stage=21,
                producer_body='conv', rows_per_image=49, pool_channels=512)
            required,_=probe.pool_keys(*self.pool_inputs(root,row))
            matches=[r for r in required if r['key']['tm']==64 and r['key']['tn']==32
                and r['key']['live_rows']==34 and r['key']['row_base_mod_image']==15]
            self.assertEqual(len(matches),1)
            self.assertEqual(matches[0]['key']['parts'],2)
            self.assertEqual(matches[0]['representative'],
                dict(row_base=64, allocation_batch=2, channel=0))

    def test_unrepresented_spatial_edge_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            row=dict(batch=1, model='resnet18', pool_producer_stage=21,
                producer_body='depthwise', rows_per_image=100, pool_channels=512,
                output_width=10, output_height=10, choices=[dict(tile_w=8,tile_h=8,parts=4)])
            with self.assertRaisesRegex(ValueError,'boundary class'):
                probe.pool_keys(*self.pool_inputs(root,row))

    def test_rowstat_tail_requires_own_key(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for model in ('resnet18','mbv1','mbv2','bert','nafnet'):
                for batch in (1,16):
                    stages=[]
                    if model=='bert':
                        stages=[dict(index=0,body='gemm',m=129,n=768,k=768,
                            dnn1_geometry=dict(has_row_stats=True,gated=False,
                            store_kind='vector_16byte',rows=129,store_columns=768,output_pitch=768))]
                    (root/f'{model}_b{batch}.json').write_text(json.dumps(dict(stages=stages)))
            with self.assertRaisesRegex(ValueError,'tail rows'):
                probe.rowstat_keys(root)

    def test_frozen_target_reconstructs_cross_image_case(self):
        with tempfile.TemporaryDirectory() as name:
            target=Path(name)/'target.json'
            key=dict(pool_rows=49,parts=2,pool_channels=512,tm=64,tn=32,
                live_rows=34,width=32,row_base_mod_image=15,stride=512,
                tw=0,th=0,image_width=0,live_tile_width=0)
            target.write_text(json.dumps(dict(sm90_forward_constants=dict(schema=2,
                keys=[['pool_finish_store_ns',probe.key_string(key),'ns']]))))
            rows,_=probe.target_keys(target,'pool')
            self.assertEqual(rows[0]['key'],key)
            self.assertEqual(rows[0]['representative'],
                dict(row_base=64,allocation_batch=2,channel=0))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'CPU PyTorch oracle dependency')
    def test_bf16_zero_sign_and_padding_are_distinct(self):
        oracle=probe.sca_oracle(32,32)
        self.assertEqual(oracle['input_physical_bf16_words'][0],49062)
        self.assertEqual(oracle['expected_physical_bf16_words'][0],0x8000)
        self.assertEqual(oracle['expected_physical_bf16_words'][32],0)
        self.assertEqual(oracle['scale_bf16_words'][0],0)


if __name__=='__main__':unittest.main()
