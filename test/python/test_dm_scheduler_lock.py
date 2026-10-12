import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"python"))
from tilemega.bench import locking as scheduler


class SharedLock(unittest.TestCase):
    def test_wait_does_not_consume_execution_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            lock=Path(folder)/'gpu.lock'
            held=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)
            fcntl.flock(held,fcntl.LOCK_EX)
            timer=threading.Timer(.3,lambda:fcntl.flock(held,fcntl.LOCK_UN))
            timer.start()
            begin=time.monotonic()
            command,descriptor=scheduler.acquire_command_lock(
                ['flock',str(lock),'timeout','.1',sys.executable,'-c','print("executed")'],str(lock))
            try:
                self.assertGreaterEqual(time.monotonic()-begin,.2)
                result=subprocess.run(command,pass_fds=(descriptor,),cwd=folder,
                    capture_output=True,text=True,timeout=2)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.strip(),'executed')
                self.assertFalse((Path(folder)/str(descriptor)).exists())
                competitor=os.open(lock,os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:os.close(competitor)
            finally:
                os.close(descriptor);timer.join();os.close(held)

    def test_dnn_child_reuses_guard_ownership_without_inherited_fd(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            held = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(held, fcntl.LOCK_EX)
            try:
                env = dict(os.environ, TILEMEGA_GPU_LOCK=str(lock),
                           TILEMEGA_GPU_LOCK_HELD='1',
                           PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'python'))
                for _ in range(5):
                    result = subprocess.run([sys.executable, '-c',
                        'from tilemega.dnn.cli import gpu_lock\nwith gpu_lock(): print("guard-owned")'],
                        env=env, capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), 'guard-owned')
                competitor = os.open(lock, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(competitor)
            finally:
                os.close(held)

    def test_other_commands_are_unchanged(self):
        command=['python3','-c','pass']
        self.assertEqual(scheduler.acquire_command_lock(command,'unused'),(command,None))


if __name__=='__main__':unittest.main()
