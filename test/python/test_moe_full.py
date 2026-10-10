"""Full-model construction command and shared deployment allocation accounting."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from tilemega.moe.full import compile_command, memory_report, resolve_target, preflight


class MoeFullTest(unittest.TestCase):
    def fixture(self):
        config = dict(num_hidden_layers=2, num_key_value_heads=1,
                      hidden_size=128, num_attention_heads=8, head_dim=64)
        weight = dict(name='w', dtype='bf16', role='external', constant=100,
                      recipe=dict(kind='alias', source='w.weight'))
        decode = dict(batch_lo=2, batch_hi=2, capacity=128, seq=1,
            past_hi=127, phase='decode', memory_arena_bytes=64,
            buffers=[weight, dict(name='a', dtype='f32', role='internal',
                                 constant=3, per_seq=2, per_past=1,
                                 per_total=1, per_batch=5),
                     dict(name='reused', dtype='bf16', role='internal',
                          constant=32, dm_arena_offset=0),
                     dict(name='tokens', dtype='i32', role='external',
                          per_batch=128)])
        prefill = copy.deepcopy(decode)
        prefill.update(seq=64, past_hi=0, phase='prefill')
        return config, [decode, prefill]

    def test_shared_weights_state_and_distinct_plan_workspaces(self):
        config, manifests = self.fixture()
        result = memory_report(config, manifests, 2, 128)
        self.assertEqual(result['shared_weight_bytes'], 200)
        self.assertEqual([v['internal_bytes'] for v in result['phases']],
                         [64+270*4, 64+205*4])
        state = 2*2*2*1*128*64*2 + 2*128*4 + 2*128*64*2
        self.assertEqual(result['request_state_bytes'], state)
        self.assertEqual(result['estimated_allocation_bytes'],
                         200+state+2*64+(270+205)*4)
        self.assertEqual(result['evidence'], 'inferred')

    def test_incompatible_phase_recipes_and_workloads_are_rejected(self):
        config, manifests = self.fixture()
        manifests[1]['buffers'][0]['recipe']['source'] = 'other.weight'
        with self.assertRaisesRegex(ValueError, 'disagree on weight'):
            memory_report(config, manifests, 2, 128)
        config, manifests = self.fixture()
        with self.assertRaisesRegex(ValueError, 'workload differs'):
            memory_report(config, manifests, 1, 128)

    def test_moe_cli_enables_handoffs_without_changing_dense_defaults(self):
        from tilemega.cli import read_config
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=root/'model'; model.mkdir()
            path=root/'run.json'
            for model_type, explicit, expected in [('llama',None,0),
                    ('qwen3_moe',None,1),('qwen3_moe',0,0)]:
                (model/'config.json').write_text(json.dumps(dict(model_type=model_type)))
                data=dict(model=dict(path=str(model)))
                if explicit is not None:data['features']=dict(nonpaged_la=explicit)
                path.write_text(json.dumps(data))
                self.assertEqual(read_config(path)['features']['nonpaged_la'],expected)

    def test_commands_use_bound_past_range_and_common_weight_layout(self):
        args = SimpleNamespace(compiler=Path('/compiler'), capacity=128,
                               pg='pages', target=Path('/target.json'))
        for phase, expected in [('decode', '1:127'), ('prefill', '0:0')]:
            command = compile_command(args, '/bridge', '/plan.cu', phase, 2)
            options = dict(zip(command[4::2], command[5::2]))
            self.assertEqual(options['--past-range'], expected)
            self.assertEqual(options['--nonpaged-weight-layout'], 'tiled')
            self.assertEqual(options['--pg'], 'pages')
            self.assertNotIn('--measure-cmd', options)

    def test_device_binding_uses_native_profile_and_probed_resources(self):
        from tilemega.fingerprint import ROOT
        for arch in ('sm_80', 'sm_90'):
            profile = json.loads((ROOT/f'configs/targets/{arch}.json').read_text())
            device = dict(arch_tag=arch, sm_major=int(arch[3:])//10,
                sm_minor=int(arch[3:])%10, caps=profile['caps'],
                resources=dict(num_sms=137, l2_bytes=12345678,
                               dram_capacity_bytes=80*1024**3))
            with tempfile.TemporaryDirectory() as directory, \
                    patch('tilemega.dnn.cli.gpu_lock', return_value=nullcontext()), \
                    patch('tilemega.moe.full.subprocess.run',
                          return_value=SimpleNamespace(stdout=json.dumps(device))) as run:
                args=SimpleNamespace(target='auto', compiler=Path('/compiler'), out=Path(directory))
                target=json.loads(resolve_target(args).read_text())
                self.assertEqual(target['arch_tag'], arch)
                self.assertEqual(target['resources']['num_sms'], 137)
                self.assertEqual(target['resources']['l2_bytes'], 12345678)
                self.assertEqual(target['resources']['dram_capacity_bytes'], 80*1024**3)
                self.assertEqual(target['calibration'], profile['calibration'])
                self.assertEqual(run.call_args.args[0], ['/compiler', 'probe', 'device'])

    def test_explicit_target_needs_no_device(self):
        with patch('tilemega.moe.full.subprocess.run') as run:
            self.assertEqual(resolve_target(SimpleNamespace(target='/target.json')),
                             Path('/target.json'))
            run.assert_not_called()

    def test_preflight_reports_memory_before_rejecting_allocation(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('tilemega.dnn.cli.gpu_lock', return_value=nullcontext()), \
                patch('torch.cuda.mem_get_info', return_value=(1024, 2048)):
            root=Path(directory); (root/'B1').mkdir()
            (root/'B1/memory.json').write_text(json.dumps(dict(estimated_allocation_bytes=4096)))
            with self.assertRaisesRegex(MemoryError, 'insufficient GPU memory'):
                preflight(SimpleNamespace(out=root, batch=[1]))
            receipt=json.loads((root/'memory-preflight.json').read_text())
            self.assertEqual(receipt['evidence'], 'inferred')
            self.assertTrue(receipt['batches'][0]['exceeds_available'])


if __name__ == '__main__':
    unittest.main()
