"""Exercise GPU allocation and recovery without starting hardware work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import dnn4_intermittent as runner


class IntermittentTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.session = self.root/'session'
        self.queue = self.root/'queue'
        self.out = self.root/'scheduler'
        self.now = 1000
        self.calls = []
        self.codes = []
        runner.save(self.session/'authorization.json', dict(status='active', allowed_gpu_indices=[0]))
        self.steps = [dict(name=name, gpu=True, env=dict(CUDA_VISIBLE_DEVICES='0',
            TILEMEGA_DEVICE_INDEX='0'), command=['/root/python/worker.py'],
            after=[] if i == 0 else ['one']) for i, name in enumerate(['one', 'two'])]
        self.write_steps()
        self.command = ['/root/python/guard.py', '-m', 'tilemega.bench.scheduler',
            '--queue-dir', str(self.queue), '--out', str(self.out), '--deadline-hours', '6']
        for target, value in [('SESSION', self.session), ('RUN', self.launch),
                ('time', SimpleNamespace(time=lambda: self.now, sleep=self.sleep))]:
            context = patch.object(runner, target, value)
            context.start();self.addCleanup(context.stop)

    def write_steps(self):
        runner.save(self.queue/'queue_unit.json', self.steps)

    def sleep(self, duration):
        self.assertLessEqual(duration, 30)
        self.now += duration

    def launch(self, command, **kwargs):
        folder = Path(command[command.index('--out')+1])
        queue = Path(command[command.index('--queue-dir')+1])
        steps = runner.read(queue/'queue_single.json')
        self.assertEqual(len(steps), 1)
        name = steps[0]['name']
        attempt = folder.parent/'raw'/name/'attempt1'
        code = self.codes.pop(0) if self.codes else 0
        started = self.now;self.now += 1
        runner.save(attempt/'guard_result.json', dict(code=code))
        state = dict(status='done' if code == 0 else 'not_run', exit_code=code,
            out=str(attempt), started=started, ended=self.now,
            retry_at=self.now+300 if code == 75 else 0)
        runner.save(folder/'state.json', {name: state})
        self.calls.append((name, started, self.now))
        return subprocess.CompletedProcess(command, 0)

    def test_one_reservation_and_gap_then_reuse_completed_receipts(self):
        runner.dispatch(self.command)
        self.assertEqual([r[0] for r in self.calls], ['one', 'two'])
        self.assertGreaterEqual(self.calls[1][1]-self.calls[0][2], 30)
        runner.dispatch(self.command)
        self.assertEqual(len(self.calls), 2)
        receipt = runner.read(self.out/'intermittent.json')
        self.assertEqual(len(receipt['source_states']), 2)

    def test_guard_deferral_preserves_cooldown_and_uses_fresh_directory(self):
        self.codes = [75, 0, 0]
        runner.dispatch(self.command)
        self.assertGreaterEqual(self.calls[1][1]-self.calls[0][2], 300)
        receipt = runner.read(self.out/'intermittent.json')
        self.assertIn('reservation2', receipt['source_states'][0])
        rejected = self.out/'jobs/01/raw/one/attempt1/guard_result.json'
        self.assertEqual(runner.read(rejected)['code'], 75)

    def test_failure_does_not_retry_or_start_dependent_process(self):
        self.codes = [86]
        with self.assertRaisesRegex(RuntimeError, 'did not complete'):
            runner.dispatch(self.command)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(ValueError, 'only guard-deferred'):
            runner.dispatch(self.command)

    def test_other_gpu_original_python_entry_and_paused_state_reject(self):
        self.steps[0]['env']['CUDA_VISIBLE_DEVICES'] = '3';self.write_steps()
        with self.assertRaisesRegex(ValueError, 'GPU0'):
            runner.dispatch(self.command)
        self.steps[0]['env']['CUDA_VISIBLE_DEVICES'] = '0'
        self.steps[0]['command'] = ['/root/repo/test.py'];self.write_steps()
        with self.assertRaisesRegex(ValueError, 'queue changed'):
            runner.dispatch(self.command)
        runner.save(self.session/'authorization.json', dict(status='paused', allowed_gpu_indices=[0]))
        with self.assertRaisesRegex(ValueError, 'not active'):
            runner.dispatch(self.command)
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
