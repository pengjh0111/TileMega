"""CPU contracts for frozen-reference, stage-boundary and payload experiments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def module(name):
    spec=importlib.util.spec_from_file_location('mx8_'+name,ROOT/'experiments/mx8'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class PayloadTest(unittest.TestCase):
    def setUp(self):
        self.bw=module('bandwidth')
        self.config=dict(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,intermediate_size=64,vocab_size=128)

    def test_dense_tp_divides_weights_and_kv_once(self):
        full=self.bw.logical_payload(self.config,4)
        half=self.bw.logical_payload(self.config,4,tp=2)
        expected_parameters=2*(32*(4+2*2)*8+32*4*8+3*32*64)+32*128
        self.assertEqual(full['weight_bytes_per_rank'],2*expected_parameters)
        self.assertEqual(full['kv_bytes_per_rank'],4*4*2*2*8*576)
        self.assertEqual(half['logical_bytes_per_rank'],full['logical_bytes_per_rank']/2)

    def test_moe_uses_per_layer_distinct_experts(self):
        config=dict(self.config,num_experts=64,num_experts_per_tok=8)
        low=self.bw.logical_payload(config,4,active_per_layer=[8,16])
        high=self.bw.logical_payload(config,4,active_per_layer=[16,16])
        self.assertEqual(high['weight_bytes_per_rank']-low['weight_bytes_per_rank'],2*3*32*64*8)
        with self.assertRaises(ValueError):self.bw.logical_payload(config,4)
        with self.assertRaises(ValueError):self.bw.logical_payload(config,4,active_per_layer=[64,64])

    def test_inventory_reuses_four_systems_and_latest_tp8(self):
        rows=self.bw.frozen_timings()
        self.assertEqual(len(rows),220)
        self.assertEqual({r['system'] for r in rows},{'tilemega','vllm','sglang','torch.compile'})
        self.assertFalse(any(r['timing_remeasured'] for r in rows))
        row=next(r for r in rows if (r['model'],r['tp'],r['batch'],r['system'])==('qwen3_1_7b',8,16,'tilemega'))
        self.assertEqual(row['timing_source'],'docs/results/MX7_tp8_final_c550.csv')


class SequenceTest(unittest.TestCase):
    def test_overlay_preserves_production_runtime_and_ticket_allocation(self):
        transform=module('build_sequence').transform_runtime
        for runtime in ('DecodeRuntimeV2.cuh','MoeRuntimeV3.cuh'):
            path=ROOT/'include/tilemega/Codegen/tasks/mx'/runtime
            original=path.read_text();changed=transform(original)
            self.assertIn('return Mx8Prepare(s);',changed)
            self.assertIn('Mx8Release(s);\n',changed)
            self.assertIn('return Mx8Launch(s, first, steps, iteration, stream);',changed)
            self.assertEqual(path.read_text(),original)
            self.assertEqual(changed.count('atomic'),original.count('atomic'))

    def test_pending_definitions_are_immutable(self):
        queue=module('queue')
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            definition=dict(kind='test',command=['true'],inputs=[])
            queue.submit(root,'same',definition);queue.submit(root,'same',definition)
            with self.assertRaises(ValueError):queue.submit(root,'same',dict(definition,command=['false']))
            self.assertEqual(json.loads((root/'jobs/same/definition.json').read_text()),definition)


if __name__=='__main__':unittest.main()
