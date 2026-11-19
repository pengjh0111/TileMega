"""User GPU restrictions reject timed and untimed jobs before any telemetry."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tilemega.bench import gpu_guard, scheduler


class UserGpuRestriction(unittest.TestCase):
    def test_default_policy_and_allowed_device(self):
        gpu_guard.require_allowed_gpu({}, '0')
        gpu_guard.require_allowed_gpu({'allowed_gpu_indices': [7]}, '7')

    def test_external_constraint_is_live_and_cannot_be_relaxed_by_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'constraints.json'
            path.write_text(json.dumps({'allowed_gpu_indices': [7]}))
            policy = {'allowed_gpu_indices': [0, 7], 'gpu_allowlist_file': str(path)}
            with self.assertRaisesRegex(ValueError, 'active user GPU restriction'):
                gpu_guard.require_allowed_gpu(policy, '0')
            gpu_guard.require_allowed_gpu(policy, '7')
            path.write_text(json.dumps({'allowed_gpu_indices': [6]}))
            with self.assertRaises(ValueError):
                gpu_guard.require_allowed_gpu(policy, '7')
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                gpu_guard.require_allowed_gpu(policy, '7')

    def test_invalid_lists_fail_closed(self):
        for allowed in ([], ['7'], [True], [-1], '7'):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                gpu_guard.require_allowed_gpu({'allowed_gpu_indices': allowed}, '7')

    def test_guard_rejects_timed_and_untimed_work_before_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root/'policy.json'
            policy.write_text(json.dumps({'allowed_gpu_indices': [7]}))
            for timing in ('0', '1'):
                args = ['gpu_guard', '--policy', str(policy), '--out', str(root/'unused'),
                        '--timeout-s', '1', '--', sys.executable, '-c', 'raise AssertionError()']
                with patch.dict(os.environ, {'TILEMEGA_DEVICE_INDEX': '0',
                       'CUDA_VISIBLE_DEVICES': '0', 'TILEMEGA_TIMING': timing}), \
                     patch.object(sys, 'argv', args), \
                     patch.object(gpu_guard, 'gpu', side_effect=AssertionError('must not query GPU')):
                    with self.assertRaisesRegex(ValueError, 'active user GPU restriction'):
                        gpu_guard.main()
            self.assertFalse((root/'unused').exists())

    def test_scheduler_rejects_old_device_before_creating_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root/'policy.json'
            policy.write_text(json.dumps({'allowed_gpu_indices': [7]}))
            with patch.dict(os.environ, {'TILEMEGA_DEVICE_INDEX': '1'}), \
                 patch.object(scheduler, 'gpu', side_effect=AssertionError('must not query GPU')):
                with self.assertRaisesRegex(ValueError, 'active user GPU restriction'):
                    scheduler.main(['--policy', str(policy), '--queue-dir', str(root/'queue'),
                                    '--out', str(root/'out')])
            self.assertFalse((root/'out').exists())


class StrictGpuOwnership(unittest.TestCase):
    def test_host_pid_matches_descendant_and_start_time(self):
        row = {'owners': {42: 10, 201: 20, 202: 30}}
        known = {202: '100'}
        with patch.object(gpu_guard, 'host_proc', return_value=(Path('/host/proc'), 200)), \
             patch.object(gpu_guard, 'descendant_starts', return_value={200: '1', 201: '8'}), \
             patch.object(gpu_guard, 'process_start', side_effect=lambda p, root: {201: '8', 202: '111'}[p]):
            self.assertEqual(gpu_guard.strict_foreign(row, 200, known), {42, 202})

    def test_first_unresolved_running_owner_invalidates_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = root/'policy.json'
            policy.write_text(json.dumps(dict(strict_host_owners=True, samples=2, interval_s=0.01,
                                               max_util_pct=2, max_hidden_mib=1024, hidden_violations=1)))
            idle = dict(owners={}, utilization_pct=0, hidden_mib=0, free_mib=65536,
                        throttle_mask=0, thermal_counters_us=[0, 0])
            busy = dict(idle, owners={41: 200})
            args = ['gpu_guard', '--policy', str(policy), '--out', str(root/'attempt'),
                    '--timeout-s', '1', '--', sys.executable, '-c', 'pass']
            env = {'TILEMEGA_DEVICE_INDEX': '0', 'CUDA_VISIBLE_DEVICES': '0',
                   'TILEMEGA_GPU_LOCK': str(root/'lock'), 'TILEMEGA_TIMING': '0',
                   'TILEMEGA_GPU_PID_MODE': 'strict'}
            with patch.dict(os.environ, env), patch.object(sys, 'argv', args), \
                 patch.object(gpu_guard, 'host_proc', return_value=(Path('/host/proc'), 200)), \
                 patch.object(gpu_guard, 'gpu', side_effect=[dict(idle), dict(idle), busy]), \
                 patch.object(gpu_guard, 'strict_foreign', return_value={41}), \
                 patch.object(gpu_guard, 'stop') as stop, patch.object(subprocess, 'Popen') as launch:
                child = launch.return_value
                child.pid = 12345678
                child.wait.side_effect = subprocess.TimeoutExpired('child', 0.01)
                self.assertEqual(gpu_guard.main(), 75)
            stop.assert_called_once()
            self.assertIs(stop.call_args.args[0], child)
            result = json.loads((root/'attempt/guard_result.json').read_text())
            self.assertEqual(result['reason'], 'running interference')
