"""Check that an override bypasses only the selected CPU build lock."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BuildLockTest(unittest.TestCase):
    def run_case(self, override):
        directory = ROOT / 'runs/sm90x1/test-build-lock'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            d = Path(temporary)
            requested, alternative = d / 'requested.lock', d / 'override.lock'
            env = dict(os.environ)
            env.pop('TILEMEGA_NVCC_LOCK', None)
            if override:
                env['TILEMEGA_NVCC_LOCK'] = str(alternative)
            with requested.open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_sm90r1_build.py'),
                    '--receipt', str(d / 'receipt.json'), '--max-gib', '1', '--timeout', '10',
                    '--lock', str(requested), '--', '/bin/true'], env=env)
                try:
                    if override:
                        self.assertEqual(process.wait(timeout=5), 0)
                    else:
                        deadline = time.monotonic() + 5
                        while not (d / 'receipt.json').exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue((d / 'receipt.json').exists())
                        self.assertIsNone(process.poll())
                        fcntl.flock(held, fcntl.LOCK_UN)
                        self.assertEqual(process.wait(timeout=5), 0)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)

    def test_override_ignores_held_explicit_lock(self):
        self.run_case(True)

    def test_default_keeps_explicit_lock(self):
        self.run_case(False)


if __name__ == '__main__':
    unittest.main()
