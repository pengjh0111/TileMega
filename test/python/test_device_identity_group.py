"""Group mode of the device identity wrapper (R5 B0.5): one identity per card."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_r16_with_device_identity as wrapper


UUIDS = ['73c1a37e-d3c0-dc24-b81d-6a5563ebe677', '344fa712-6ca8-3673-b04b-33b9c3cbffac',
         '701b606f-5e57-e434-5e2b-d5cc774c3c9e', '5b5e235a-09a2-fe8b-7754-916b7eb85976']
PCIS = ['0000:0f:00.0', '0000:10:00.0', '0000:41:00.0', '0000:44:00.0']


def device(k):
    return dict(cuda_uuid=UUIDS[k], cuda_pci_bus_id=PCIS[k], numa_node=0, sm_count=132,
                total_memory=85520809984, probe_pid=1,
                cuda_driver_queries_without_context=True)


class GroupIdentityTests(unittest.TestCase):
    def wrapper_argv(self, out, group, uuids, pcis, nodes):
        return ['run_r16_with_device_identity.py', '--out', str(out),
                '--expected-uuid', ','.join(uuids), '--expected-pci', ','.join(pcis),
                '--numa-node', ','.join(nodes), '--guard-policy', str(self.policy),
                '--allow-no-measurements', '--', 'true']

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.policy = Path(self.directory.name) / 'policy.json'
        self.policy.write_text(json.dumps(dict(allowed_gpu_indices=list(range(8)))))
        self.addCleanup(self.directory.cleanup)

    def run_wrapper(self, group, uuids, pcis, nodes, probed):
        out = Path(self.directory.name) / 'out'
        env = dict(TILEMEGA_DEVICE_GROUP=','.join(str(g) for g in group),
                   CUDA_VISIBLE_DEVICES=','.join(str(g) for g in group),
                   TILEMEGA_GPU_LOCK_HELD='1')
        with patch.dict(os.environ, env, clear=True), \
                patch.object(sys, 'argv', self.wrapper_argv(out, group, uuids, pcis, nodes)), \
                patch.object(wrapper, 'require_group_guard', lambda policy, g: None), \
                patch.object(wrapper, 'run_probe', lambda args: probed), \
                patch.object(wrapper, 'nvml_uuid', lambda index: 'GPU-' + UUIDS[group.index(index)]):
            code = wrapper.main()
        return code, out

    def test_each_card_is_verified_against_its_own_expectation(self):
        group = [0, 1, 2, 3]
        code, out = self.run_wrapper(group, UUIDS, PCIS, ['0'] * 4, [device(k) for k in range(4)])
        self.assertEqual(code, 0)
        receipt = json.loads((out / 'physical-device-identity.json').read_text())
        self.assertEqual(receipt['group'], group)
        self.assertEqual([d['gpu_index'] for d in receipt['devices']], group)
        self.assertEqual([d['cuda_uuid'] for d in receipt['devices']], UUIDS)
        self.assertTrue(all(d['nvml_uuid'].startswith('GPU-') for d in receipt['devices']))

    def test_a_list_shorter_than_the_group_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_wrapper([0, 1, 2, 3], UUIDS[:3], PCIS, ['0'] * 4,
                             [device(k) for k in range(4)])

    def test_one_mismatched_card_fails_the_whole_group(self):
        group = [0, 1, 2, 3]
        probed = [device(k) for k in range(4)]
        # Card 2 reports a different physical device than the one predeclared.
        probed[2] = dict(probed[2], cuda_uuid=UUIDS[3])
        with self.assertRaises(ValueError):
            self.run_wrapper(group, UUIDS, PCIS, ['0'] * 4, probed)
        probed = [device(k) for k in range(4)]
        probed[1] = dict(probed[1], numa_node=1)
        with self.assertRaises(ValueError):
            self.run_wrapper(group, UUIDS, PCIS, ['0'] * 4, probed)
        probed = [device(k) for k in range(4)]
        probed[0] = dict(probed[0], cuda_pci_bus_id='0000:86:00.0')
        with self.assertRaises(ValueError):
            self.run_wrapper(group, UUIDS, PCIS, ['0'] * 4, probed)

    def test_single_card_mode_still_takes_scalar_values(self):
        out = Path(self.directory.name) / 'single'
        env = dict(TILEMEGA_DEVICE_INDEX='0', CUDA_VISIBLE_DEVICES='0',
                   TILEMEGA_GPU_LOCK_HELD='1', TILEMEGA_GPU_LOCK='/tmp/tilemega_gpu0.lock')
        argv = ['run_r16_with_device_identity.py', '--out', str(out), '--expected-uuid', UUIDS[0],
                '--expected-pci', PCIS[0], '--numa-node', '0', '--guard-policy', str(self.policy),
                '--allow-no-measurements', '--', 'true']
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                patch.object(wrapper, 'require_guard', lambda policy, index: None), \
                patch.object(wrapper, 'run_probe', lambda args: device(0)), \
                patch.object(wrapper, 'nvml_uuid', lambda index: 'GPU-' + UUIDS[0]):
            self.assertEqual(wrapper.main(), 0)
        receipt = json.loads((out / 'physical-device-identity.json').read_text())
        # The single-card receipt keeps its flat shape, so frozen R16 data stays comparable.
        self.assertNotIn('group', receipt)
        self.assertEqual(receipt['gpu_index'], 0)
        self.assertEqual(receipt['cuda_uuid'], UUIDS[0])


if __name__ == '__main__':
    unittest.main()
