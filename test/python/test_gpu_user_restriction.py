"""User GPU restrictions reject timed and untimed jobs before any telemetry."""
from pathlib import Path
import json
import os
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
