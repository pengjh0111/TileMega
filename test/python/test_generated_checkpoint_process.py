import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'generated_checkpoint', ROOT / 'scripts/check_generated_models.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


class GeneratedCheckpointProcessTest(unittest.TestCase):
    def test_timeout_reaps_compiler_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            child_pid = folder / 'child.pid'
            program = (
                'import signal,subprocess,sys,time\n'
                'p=subprocess.Popen([sys.executable,"-c",'
                '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])\n'
                'open(sys.argv[1],"w").write(str(p.pid))\n'
                'time.sleep(60)\n')
            pid = None
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    checkpoint.execute([sys.executable, '-c', program, str(child_pid)],
                                       ROOT, folder, 'compile', timeout=1)
                pid = int(child_pid.read_text())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    path = Path('/proc', str(pid), 'stat')
                    if not path.exists() or path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                        break
                    time.sleep(.02)
                else:
                    self.fail('compiler descendant survived its timeout')
                self.assertTrue((folder / 'compile.command.json').is_file())
            finally:
                if pid is not None:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
