"""Storage failure before launch must leave the pending attempt untouched."""
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from tilemega.bench.scheduler import main, prepare_attempt, prefer_ready_cpu

class DiskAdmission(unittest.TestCase):
    def test_withdrawal_after_directory_listing_keeps_other_jobs_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);queue = root/'queue';queue.mkdir()
            withdrawn = queue/'queue_withdrawn.json'
            withdrawn.write_text(json.dumps([dict(name='withdrawn')]))
            (queue/'queue_keep.json').write_text(json.dumps([dict(
                name='keep',gpu=False,timing=False,timeout_s=5,
                command=[sys.executable,'-c','pass'])]))
            policy = root/'policy.json';policy.write_text('{}')
            original = Path.glob
            def withdraw_after_listing(directory, pattern):
                paths = list(original(directory, pattern))
                if directory == queue and withdrawn.exists():withdrawn.unlink()
                return iter(paths)
            env = dict(TILEMEGA_GPU_LOCK=str(root/'gpu.lock'),
                       TILEMEGA_DEVICE_INDEX='3',CUDA_VISIBLE_DEVICES='')
            with patch.object(Path,'glob',withdraw_after_listing), patch.dict(os.environ,env):
                main(['--queue-dir',str(queue),'--out',str(root/'out'),
                      '--policy',str(policy)])
            state = json.loads((root/'out/state.json').read_text())
            self.assertEqual(state['keep']['status'],'done')
            self.assertNotIn('withdrawn',state)

    def test_high_priority_cpu_gate_precedes_lower_priority_gpu(self):
        gate = dict(name='gate', gpu=False, priority=-2)
        gpu = dict(name='baseline', gpu=True, priority=0)
        self.assertIs(prefer_ready_cpu([gate, gpu], gpu), gate)
        self.assertIs(prefer_ready_cpu([dict(gate, priority=3), gpu], gpu), gpu)
        self.assertIs(prefer_ready_cpu([gate, gpu], None), gate)
        # A rejected/blocked GPU candidate must not be recovered from ready.
        self.assertIsNone(prefer_ready_cpu([gpu], None))
        tied = dict(gate, name='z-gate', priority=0)
        self.assertIs(prefer_ready_cpu([gpu, tied], gpu), gpu)

    def test_no_space_does_not_invent_an_attempt(self):
        record = dict(status='pending', attempts=0)
        with patch.object(Path, 'mkdir', side_effect=OSError(errno.ENOSPC, 'full')):
            self.assertIsNone(prepare_attempt(Path('/out/gpu7'), 'job', record))
        self.assertEqual(record, dict(status='pending', attempts=0))

    def test_real_directory_allocates_exactly_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = dict(status='pending', attempts=0)
            folder = prepare_attempt(Path(tmp)/'gpu7', 'job', record)
            self.assertTrue(folder.is_dir())
            self.assertEqual(record['attempts'], 1)

    def test_permissions_are_not_misreported_as_no_space(self):
        record = dict(status='pending', attempts=0)
        with patch.object(Path, 'mkdir', side_effect=OSError(errno.EACCES, 'denied')):
            with self.assertRaises(OSError):
                prepare_attempt(Path('/out/gpu7'), 'job', record)
        self.assertEqual(record['attempts'], 0)

if __name__ == '__main__':
    unittest.main()
