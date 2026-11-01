"""CPU-only checks for shared C550 card locks and group telemetry."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tilemega.bench import gpu_guard as guard


POLICY = dict(samples=1, interval_s=0, idle_power_w=100, idle_vram_mib=860,
              power_margin_w=20, max_util_pct=5, max_hidden_mib=1024,
              hidden_violations=3)


def telemetry(owners=None):
    return dict(owners=owners or {}, utilization_pct=0, power_w=60,
                used_mib=860, free_mib=64000)


class GroupGuardTests(unittest.TestCase):
    def test_group_locks_are_sorted_and_shared_with_single_card_jobs(self):
        with patch.dict(os.environ, {'TILEMEGA_DEVICE_GROUP': '7,6,5,4'}, clear=True), \
                patch.object(guard.fcntl, 'flock') as lock:
            with guard.device_locks(guard.device_indices()):
                self.assertEqual([Path(c.args[0].name).name for c in lock.call_args_list],
                                 [f'tilemega_mx_gpu{g}.lock' for g in (4, 5, 6, 7)])
                self.assertTrue(all(c.args[1] == guard.fcntl.LOCK_EX for c in lock.call_args_list))

    def run_guard(self, group=None, foreign=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root / 'policy.json'
            policy.write_text(json.dumps(POLICY))
            env = dict(TILEMEGA_DEVICE_INDEX='6', TILEMEGA_GPU_LOCK=str(root / 'single.lock'))
            if group:
                env['TILEMEGA_DEVICE_GROUP'] = group
            child = MagicMock(pid=999999)
            child.wait.return_value = 0
            def query(index):
                return telemetry({424242: 1000} if str(index) == foreign else {})
            argv = ['gpu_guard', '--policy', str(policy), '--out', str(root / 'out'),
                    '--timeout-s', '1', '--', 'true']
            with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                    patch.object(guard.mx_smi, 'device', side_effect=query) as smi, \
                    patch.object(guard.subprocess, 'Popen', return_value=child) as launch, \
                    patch.object(guard.fcntl, 'flock'), patch.object(guard, 'stop'), \
                    patch.object(guard, 'descendants', return_value={os.getpid()}), \
                    patch.object(guard, 'session_members', return_value=set()), \
                    patch.object(guard, 'device_openers', return_value=set()):
                result = guard.main()
                return result, smi.call_args_list, launch.call_count

    def test_one_occupied_card_rejects_the_whole_group(self):
        code, calls, launched = self.run_guard('4,5,6,7', foreign='7')
        self.assertEqual(code, 75)
        self.assertEqual(launched, 0)
        self.assertEqual([c.args[0] for c in calls], ['4', '5', '6', '7'])

    def test_all_cards_checked_before_and_after(self):
        code, calls, launched = self.run_guard('4,5,6,7')
        self.assertEqual((code, launched), (0, 1))
        self.assertEqual([c.args[0] for c in calls], ['4', '5', '6', '7'] * 2)

    def test_single_card_mode_retains_its_index_and_lock(self):
        code, calls, launched = self.run_guard()
        self.assertEqual((code, launched), (0, 1))
        self.assertEqual([c.args[0] for c in calls], ['6', '6'])

    def test_group_rejects_forbidden_cards(self):
        for value in ('0,4', '4,5,6', '4,x'):
            with patch.dict(os.environ, {'TILEMEGA_DEVICE_GROUP': value}, clear=True):
                with self.assertRaises(ValueError):
                    guard.device_indices()


if __name__ == '__main__':
    unittest.main()
