#!/usr/bin/env python3
"""CPU checks for completion-event recovery after the client is interrupted."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RecoveryTests(unittest.TestCase):
    def test_attach_accepts_an_already_completed_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = root / 'queue'; queue.mkdir()
            out = root / 'events'; out.mkdir()
            (queue / 'queue_one.json').write_text(json.dumps([dict(name='one')]))
            (out / 'state.json').write_text(json.dumps(dict(one=dict(status='done'))))
            scheduler = root / 'scheduler.py'
            scheduler.write_text('import signal; print("ready",flush=True); signal.pause()\n')
            worker = subprocess.Popen([sys.executable, str(scheduler), '--out', str(out)],
                                      stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(worker.stdout.readline().strip(), 'ready')
                (out / 'scheduler.pid').write_text(str(worker.pid))
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_queue.py')),
                    '--attach', '--queue-dir', str(queue), '--out', str(out),
                    '--policy', str(root / 'unused-policy.json')], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads((out / 'completion.json').read_text())['passed'])
                worker.wait(timeout=5)
            finally:
                if worker.poll() is None: worker.kill(); worker.wait()
                worker.stdout.close()

    def test_attach_rejects_an_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = root / 'queue'; queue.mkdir()
            out = root / 'events'; out.mkdir()
            (queue / 'queue_one.json').write_text(json.dumps([dict(name='one')]))
            worker = subprocess.Popen([sys.executable, '-c',
                'import signal; print("ready",flush=True); signal.pause()'], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(worker.stdout.readline().strip(), 'ready')
                (out / 'scheduler.pid').write_text(str(worker.pid))
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_queue.py')),
                    '--attach', '--queue-dir', str(queue), '--out', str(out),
                    '--policy', str(root / 'unused-policy.json')], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(worker.poll())
            finally:
                worker.terminate(); worker.wait(timeout=5); worker.stdout.close()


if __name__ == '__main__':
    unittest.main()
