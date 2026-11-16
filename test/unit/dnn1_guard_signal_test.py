"""CPU regression for interruption while a separately sessioned child is live."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def alive(pid):
    status = Path('/proc', str(pid), 'stat')
    return status.exists() and status.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / 'fake_guard.py'
        script.write_text('''import fcntl,importlib.util,os,subprocess,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('entry',sys.argv[1]);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
out=Path(sys.argv[3]);out.mkdir()
def fake_main():
 with (out/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  child="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path("+repr(str(out/'child.pid'))+").write_text(str(os.getpid()));time.sleep(60)"
  subprocess.Popen([sys.executable,'-c',child],start_new_session=True)
  while True:time.sleep(.05)
entry.run(fake_main)
''')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), CUDA_VISIBLE_DEVICES='')
        for index in range(5):
            out = root / str(index)
            process = subprocess.Popen([sys.executable, str(script),
                str(ROOT / 'scripts/dnn1_guard_entry.py'), '--out', str(out)], env=env)
            try:
                deadline = time.monotonic() + 10
                while not (out / 'child.pid').is_file():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('CPU guard fixture did not start')
                    time.sleep(.02)
                child = int((out / 'child.pid').read_text())
                assert alive(child)
                with (out / 'lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        raise AssertionError('fixture guard never held its lock')
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=10) == 143
                deadline = time.monotonic() + 2
                while alive(child) and time.monotonic() < deadline:
                    time.sleep(.02)
                assert not alive(child), 'separate-session payload survived guard interruption'
                assert json.loads((out / 'guard_result.json').read_text())['acceptance'] == 'rejected'
                record = json.loads((out / 'dnn1_guard_interruption.json').read_text())
                assert str(child) in record['captured_owned_pids']
                with (out / 'lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        print('PASS: 5 fresh CPU guards clean separate-session children before releasing locks')


if __name__ == '__main__':
    main()
