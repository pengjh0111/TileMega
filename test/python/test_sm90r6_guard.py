import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
from sm90r6_scheduler import GuardLauncher
BOOTSTRAP = '''
import sys,time
from tilemega.bench import gpu_guard
gpu_guard.gpu=lambda index:dict(time=time.time(),gpu_index=0,owners={},hidden_mib=0,
    utilization_pct=int(sys.argv.pop(1)),power_w=0,free_mib=80000,throttle_mask=0)
# Keep a constant utilization for every telemetry sample.
row=gpu_guard.gpu('0');gpu_guard.gpu=lambda index:dict(row,time=time.time())
from sm90r6_guard import main
raise SystemExit(main())
'''


class PreflightLocks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / 'gpu.lock'
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps(dict(samples=1, interval_s=.1, idle_power_w=0,
            power_margin_w=1, max_util_pct=5, max_hidden_mib=1, hidden_violations=1)))

    def command(self, index, mocked=False, utilization=0):
        out = self.root / str(index); marker = self.root / ('child'+str(index))
        child = ('import fcntl,os,pathlib; '
                 'f=open(os.environ["TILEMEGA_GPU_LOCK"],"a"); '
                 '\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                 '\nexcept BlockingIOError: pathlib.Path('+repr(str(marker))+').touch()'
                 '\nelse: raise RuntimeError("guard did not retain physical lock")')
        prefix = [sys.executable, '-c', BOOTSTRAP, str(utilization)] if mocked else [
            sys.executable, str(ROOT / 'scripts/sm90r6_guard.py')]
        command = prefix + ['--policy', str(self.policy), '--out', str(out),
                            '--timeout-s', '10', '--', sys.executable, '-c', child]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,[ROOT/'python',ROOT/'scripts'])),
            TILEMEGA_DEVICE_INDEX='0', CUDA_VISIBLE_DEVICES='0', TILEMEGA_TIMING='0',
            TILEMEGA_GPU_LOCK=str(self.lock), TILEMEGA_PREFLIGHT_LOCK_TIMEOUT_S='.4')
        env.pop('TILEMEGA_DEVICE_GROUP', None)
        return command, env, out, marker

    def test_five_fresh_blocked_guards_never_start_preflight_or_child(self):
        with self.lock.open('a') as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            for index in range(5):
                command, env, out, marker = self.command(index)
                command = [command[0], '-m', 'tilemega.bench.gpu_guard', *command[2:]]
                process = GuardLauncher().Popen(command, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 75, stderr.decode())
                self.assertEqual(json.loads((out/'guard_result.json').read_text())['acceptance'], 'rejected')
                self.assertFalse((out/'guard.jsonl').exists())
                self.assertFalse((out/'pgid').exists())
                self.assertFalse(marker.exists())

    def test_five_fresh_handoffs_hold_lock_through_child(self):
        for index in range(5):
            command, env, out, marker = self.command(index, mocked=True)
            with self.lock.open('a') as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                deadline = time.monotonic()+5
                while not (out/'lock_policy.json').exists() and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertFalse(marker.exists())
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr.decode())
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads((out/'guard_result.json').read_text())['code'], 0)
            with self.lock.open('a') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_occupied_preflight_still_rejects_after_acquiring_lock(self):
        command, env, out, marker = self.command(0, mocked=True, utilization=100)
        process = subprocess.run(command, env=env, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 75, process.stderr.decode())
        self.assertEqual(json.loads((out/'guard_result.json').read_text())['reason'], 'preflight occupied')
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
