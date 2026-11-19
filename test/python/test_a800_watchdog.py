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


if __name__ == '__main__':
    unittest.main()
