"""Fresh blocking-lock fixtures without CUDA, GPU queries, or model work."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_fifo_background_python as fifo
import dnn1_aitemplate_collect as controller


class FIFOTest(unittest.TestCase):
    def test_strict_idle_and_reservations_survive_lock_queue_admission(self):
        row = dict(gpu_index=0)
        with patch.object(fifo.background, 'main_reservations', return_value=[]):
            self.assertTrue(fifo.admitted(row, lambda *a: True, {}, scheduler=True))
            self.assertFalse(fifo.admitted(row, lambda *a: False, {}, scheduler=True))
        with patch.object(fifo.background, 'main_reservations', return_value=['native']):
            self.assertFalse(fifo.admitted(row, lambda *a: True, {}, scheduler=True))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(controller, 'main_reservations', return_value=[]), \
                patch.object(controller.base, 'held', return_value=True), \
                patch.object(controller, 'resource_snapshot', return_value=dict(
                    owners={}, used_mib=0, free_mib=80000)):
            self.assertFalse(controller.available_card(Path(folder), 0))
            self.assertTrue(controller.available_card(Path(folder), 0, fifo=True))

    def test_five_fresh_processes_block_until_original_holder_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(5):
                lock, ready, acquired = [root / f'{index}.{name}' for name in ('lock', 'ready', 'acquired')]
                with lock.open('a') as holder:
                    fcntl.flock(holder, fcntl.LOCK_EX)
                    code = ('import sys,fcntl;from pathlib import Path;'
                        f'sys.path.insert(0,{str(ROOT / "scripts")!r});'
                        'import dnn1_fifo_background_python as f\n'
                        'def guard():\n'
                        f' with open({str(lock)!r},"a") as stream:\n'
                        f'  Path({str(ready)!r}).write_text("waiting")\n'
                        '  fcntl.flock(stream,fcntl.LOCK_EX)\n'
                        f'  Path({str(acquired)!r}).write_text("acquired")\n'
                        ' return 0\n'
                        f'f.settling.main=lambda:f.background.nonblocking_guard(guard,Path({str(root)!r}))\n'
                        'raise SystemExit(f.main())\n')
                    process = subprocess.Popen([sys.executable, '-c', code])
                    try:
                        deadline = time.monotonic() + 5
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue(ready.exists())
                        time.sleep(.05)
                        self.assertFalse(acquired.exists())
                        self.assertIsNone(process.poll())
                        fcntl.flock(holder, fcntl.LOCK_UN)
                        self.assertEqual(process.wait(timeout=5), 0)
                        self.assertTrue(acquired.exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()


if __name__ == '__main__':
    unittest.main()
