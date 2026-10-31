"""Group mode (R5 B0): one unit locks every card it spans and owns its workers."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tilemega.bench.gpu_guard import external, idle, nonce_owners
from tilemega.serving.baseline_common import require_guard


POLICY = dict(max_util_pct=5, idle_power_w=80, power_margin_w=20, max_hidden_mib=1024)


def row(index, owners):
    return dict(gpu_index=index, owners=owners, utilization_pct=0, power_w=70,
                hidden_mib=0, free_mib=80000, throttle_mask=0)


class GroupGuardTests(unittest.TestCase):
    def test_require_guard_accepts_a_matching_group_and_counts_it(self):
        env = dict(TILEMEGA_DEVICE_GROUP='0,1,2,3', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   TILEMEGA_GPU_LOCK_HELD='1')
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(require_guard(), ['0', '1', '2', '3'])
            self.assertEqual(require_guard(4), ['0', '1', '2', '3'])
            # A group of the wrong size is a scheduling error, not a slow round.
            with self.assertRaises(ValueError):
                require_guard(2)

    def test_require_guard_rejects_a_group_that_does_not_match_the_mask(self):
        for env in (dict(TILEMEGA_DEVICE_GROUP='0,1', CUDA_VISIBLE_DEVICES='0,2'),
                    dict(TILEMEGA_DEVICE_GROUP='0,0', CUDA_VISIBLE_DEVICES='0,0'),
                    dict(TILEMEGA_DEVICE_GROUP='0,x', CUDA_VISIBLE_DEVICES='0,x')):
            with patch.dict(os.environ, dict(env, TILEMEGA_GPU_LOCK_HELD='1'), clear=True):
                with self.assertRaises(ValueError):
                    require_guard()

    def test_single_card_mode_is_unchanged(self):
        with patch.dict(os.environ, dict(TILEMEGA_DEVICE_INDEX='5', CUDA_VISIBLE_DEVICES='5',
                                         TILEMEGA_GPU_LOCK_HELD='1'), clear=True):
            self.assertEqual(require_guard(), ['5'])
            self.assertEqual(require_guard(1), ['5'])

    def test_a_stamped_worker_outside_the_process_tree_is_not_foreign(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / '4242').mkdir()
            (proc / '4242' / 'environ').write_bytes(b'PATH=/usr/bin\0TILEMEGA_RUN_NONCE=abc\0')
            with patch('tilemega.bench.gpu_guard.proc_files',
                       lambda root, name: [proc / '4242' / name]):
                self.assertEqual(nonce_owners('abc'), {4242})
                self.assertEqual(nonce_owners('other'), set())
                self.assertEqual(nonce_owners(None), set())

    def test_an_unstamped_owner_on_any_card_voids_the_group(self):
        with patch('tilemega.bench.gpu_guard.nvml_owners', lambda own: own), \
                patch('tilemega.bench.gpu_guard.nonce_owners', lambda nonce: {4242}), \
                patch('tilemega.bench.gpu_guard.descendants', lambda pid: {os.getpid()}):
            self.assertEqual(external(row(0, {4242: 1000}), {os.getpid()}, nonce='abc'), set())
            self.assertTrue(idle(row(0, {4242: 1000}), POLICY, nonce='abc'))
            # GPU 2 of the group carries a process this unit never started.
            self.assertEqual(external(row(2, {999999: 1000}), {os.getpid()}, nonce='abc'),
                             {999999})
            self.assertFalse(idle(row(2, {999999: 1000}), POLICY, nonce='abc'))


if __name__ == '__main__':
    unittest.main()
