"""Five fresh CPU signal fixtures; no CUDA or GPU process is launched."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from dnn1_owned_controller import OwnedController


class OwnedTest(unittest.TestCase):
    def test_pid_reuse_is_not_signalled(self):
        with tempfile.TemporaryDirectory() as folder:
            owner = OwnedController(folder)
            with patch.object(owner, 'record', return_value=dict(pid=123, start='new')), \
                    patch('dnn1_owned_controller.os.kill') as kill:
                owner.send(dict(pid=123, start='old', role='child'), signal.SIGTERM, [])
                kill.assert_not_called()

    def test_five_fresh_SIGTERM_scopes_drain_children_and_leave_foreign_process(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        self.addCleanup(lambda: foreign.kill() if foreign.poll() is None else None)
        self.addCleanup(lambda: foreign.wait() if foreign.poll() is not None else None)
        with tempfile.TemporaryDirectory() as folder:
            for index in range(5):
                out = Path(folder) / str(index)
                out.mkdir()
                code = ('import sys,subprocess,time; from pathlib import Path; '
                    f'sys.path.insert(0,{str(ROOT / "scripts")!r}); '
                    'from dnn1_owned_controller import OwnedController\n'
                    f'with OwnedController({str(out)!r},grace_s=.5) as owner:\n'
                    ' p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)\n'
                    ' owner.register(p)\n'
                    f' Path({str(out / "ready")!r}).write_text(str(p.pid))\n'
                    ' while True:time.sleep(.02)\n')
                parent = subprocess.Popen([sys.executable, '-c', code])
                deadline = time.monotonic() + 5
                while not (out / 'ready').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((out / 'ready').exists())
                parent.send_signal(signal.SIGTERM)
                parent.wait(timeout=5)
                result = json.loads((out / 'controller_cleanup.json').read_text())
                self.assertTrue(result['all_owned_nonce_drained'])
                self.assertTrue(result['captured_owned_processes'])
                self.assertNotIn(foreign.pid, [r['pid'] for r in result['signals']])
                self.assertIsNone(foreign.poll())
                self.assertEqual(result['remaining'], [])
        foreign.terminate()
        foreign.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
