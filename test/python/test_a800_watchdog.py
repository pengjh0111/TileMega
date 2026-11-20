"""The allocation watchdog terminates an idle child and leaves other cells resumable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tilemega.bench import gpu_guard, scheduler


class WatchdogTests(unittest.TestCase):
    def test_missing_benchmark_progress_stops_own_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policy = dict(cuda_visible_devices='0', max_util_pct=2, max_hidden_mib=1024,
                          samples=1, interval_s=0.02, hidden_violations=1)
            (root/'policy.json').write_text(json.dumps(policy))
            argv = ['gpu_guard', '--policy', str(root/'policy.json'), '--out', str(root/'raw'),
                    '--timeout-s', '2', '--heartbeat-timeout-s', '0.08', '--',
                    sys.executable, '-c', 'import time; time.sleep(60)']
            env = dict(CUDA_VISIBLE_DEVICES='0', TILEMEGA_DEVICE_INDEX='3',
                       TILEMEGA_GPU_LOCK=str(root/'gpu.lock'), TILEMEGA_TIMING='0')
            row = dict(owners={}, utilization_pct=0, hidden_mib=0, free_mib=32768)
            with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                    patch.object(gpu_guard, 'gpu', side_effect=lambda *args: dict(row)):
                self.assertEqual(gpu_guard.main(), 124)
            raw = root/'raw'
            result = json.loads((raw/'guard_result.json').read_text())
            self.assertEqual(result['reason'], 'benchmark heartbeat timed out')
            self.assertEqual(result['acceptance'], 'rejected')
            self.assertIsNone(gpu_guard.process_start(int((raw/'pgid').read_text())))

    def test_allocation_exits_without_an_idle_device(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = root/'queue'
            queue.mkdir()
            policy = dict(max_util_pct=2, max_hidden_mib=1024,
                          samples=1, interval_s=1, hidden_violations=1)
            (root/'policy.json').write_text(json.dumps(policy))
            (queue/'queue_test.json').write_text(json.dumps([
                dict(name='model', gpu=True, timeout_s=30, command=['false'])]))
            row = dict(owners={}, utilization_pct=99, hidden_mib=0, free_mib=32768)
            args = ['--policy', str(root/'policy.json'), '--queue-dir', str(queue),
                    '--out', str(root/'output'), '--deadline-hours', '1',
                    '--release-on-blocked', '--only', 'model']
            with patch.object(scheduler, 'gpu', return_value=row), \
                    patch.object(scheduler.time, 'sleep') as sleep, \
                    patch.object(subprocess, 'Popen') as start:
                scheduler.main(args)
            sleep.assert_not_called()
            start.assert_not_called()
            record = json.loads((root/'output/state.json').read_text())['model']
            self.assertEqual((record['status'], record['attempts']), ('pending', 0))


    def test_memory_snapshot_is_confirmed_without_allowing_persistent_use(self):
        for remains_hidden in (False, True):
            with self.subTest(remains_hidden=remains_hidden), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                policy = dict(strict_host_owners=True, confirm_hidden_memory=True,
                              max_util_pct=2, max_hidden_mib=1024, samples=1,
                              interval_s=0.02, hidden_violations=1)
                (root/'policy.json').write_text(json.dumps(policy))
                argv = ['gpu_guard', '--policy', str(root/'policy.json'), '--out', str(root/'raw'),
                        '--timeout-s', '2', '--', '/bin/true']
                env = dict(CUDA_VISIBLE_DEVICES='0', TILEMEGA_DEVICE_INDEX='0',
                           TILEMEGA_GPU_LOCK=str(root/'gpu.lock'), TILEMEGA_TIMING='0')
                idle = dict(owners={}, utilization_pct=0, hidden_mib=4, free_mib=32768,
                            thermal_counters_us=[0, 0])
                mismatch = dict(idle, hidden_mib=4300)
                readings = [dict(idle), dict(mismatch), dict(mismatch if remains_hidden else idle), dict(idle)]
                with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                        patch.object(gpu_guard, 'host_proc', return_value=(Path('/proc'), 200)), \
                        patch.object(gpu_guard, 'gpu', side_effect=readings), \
                        patch.object(gpu_guard, 'strict_foreign', return_value=set()), \
                        patch.object(gpu_guard, 'stop'), patch.object(subprocess, 'Popen') as launch:
                    launch.return_value.pid = 12345678
                    launch.return_value.wait.side_effect = [subprocess.TimeoutExpired('child', 0.02), 0]
                    self.assertEqual(gpu_guard.main(), 75 if remains_hidden else 0)
                evidence = (root/'raw/guard.jsonl').read_text()
                self.assertIn('running_confirm', evidence)

    def test_exited_verified_owner_requires_an_immediate_clean_sample(self):
        pid = 234567890
        for known_owner, remains_visible, expected in ((True, False, 0), (True, True, 75), (False, False, 75)):
            with self.subTest(known=known_owner, visible=remains_visible), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                policy = dict(strict_host_owners=True, confirm_retired_owners=True,
                    max_util_pct=2, max_hidden_mib=1024, samples=1, interval_s=.02, hidden_violations=1)
                (root/'policy.json').write_text(json.dumps(policy))
                argv = ['gpu_guard', '--policy', str(root/'policy.json'), '--out', str(root/'raw'),
                        '--timeout-s', '2', '--', '/bin/true']
                env = dict(CUDA_VISIBLE_DEVICES='0', TILEMEGA_DEVICE_INDEX='0',
                           TILEMEGA_GPU_LOCK=str(root/'gpu.lock'), TILEMEGA_TIMING='0')
                idle = dict(owners={}, utilization_pct=0, hidden_mib=4, free_mib=32768,
                            thermal_counters_us=[0, 0])
                stale = dict(idle, owners={pid: 492})

                def owners(row, guard_pid, known):
                    if known_owner:
                        known[pid] = '123'
                    return set(row['owners'])

                samples = [dict(idle), dict(stale), dict(stale if remains_visible else idle), dict(idle)]
                with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                        patch.object(gpu_guard, 'host_proc', return_value=(Path('/proc'), 200)), \
                        patch.object(gpu_guard, 'gpu', side_effect=samples), \
                        patch.object(gpu_guard, 'strict_foreign', side_effect=owners), \
                        patch.object(gpu_guard, 'stop'), patch.object(subprocess, 'Popen') as launch:
                    launch.return_value.pid = 12345678
                    launch.return_value.wait.side_effect = [subprocess.TimeoutExpired('child', .02), 0]
                    self.assertEqual(gpu_guard.main(), expected)
                self.assertEqual('running_confirm' in (root/'raw/guard.jsonl').read_text(), known_owner)


if __name__ == '__main__':
    unittest.main()
