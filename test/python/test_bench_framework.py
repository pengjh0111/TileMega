import copy
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tilemega.bench import anchor
from tilemega.bench.arms import register, check_binaries
from tilemega.bench.gpu_guard import external, idle, stop
from tilemega.bench.make_queue import prepare
from tilemega.bench.report import metrics, discernible
from tilemega.bench.scheduler import ready_steps, timing_pause
from tilemega.bench.status import read_status
from tilemega.bench.identity_join import checked_join
from tilemega.build.identity import digest


class GuardTests(unittest.TestCase):
    def test_idle_limits(self):
        policy = dict(max_util_pct=5, idle_power_w=21.81, power_margin_w=30, max_hidden_mib=1024)
        row = dict(owners={}, utilization_pct=5, power_w=51.81, hidden_mib=1024, free_mib=12288)
        self.assertTrue(idle(row, policy, 12288))
        for key, value in [('utilization_pct', 6), ('power_w', 52), ('hidden_mib', 1025), ('free_mib', 12287)]:
            self.assertFalse(idle(dict(row, **{key: value}), policy, 12288))
        with patch('tilemega.bench.gpu_guard.external', return_value={123}):
            self.assertFalse(idle(row, policy, 12288))

    def test_own_and_dead(self):
        # mx-smi lists container-local PIDs only: any listed owner outside the
        # job is foreign, as is a listed owner count above the job's device openers.
        with patch('tilemega.bench.gpu_guard.device_openers', side_effect=lambda pids: set(pids)):
            self.assertEqual(external(dict(owners={os.getpid(): 100}), {os.getpid()}), set())
            self.assertEqual(external(dict(owners={os.getpid(): 100, 999999999: 100}), {os.getpid()}),
                             {999999999, -1})
        with patch('tilemega.bench.gpu_guard.device_openers', return_value=set()):
            self.assertEqual(external(dict(owners={os.getpid(): 100}), {os.getpid()}), {-1})

    def test_cleanup_after_parent_exit(self):
        worker = 'import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(os.getpid(),flush=True);signal.pause()'
        parent = 'import subprocess,sys,signal;p=subprocess.Popen([sys.executable,"-c",' + repr(worker) + '],stdout=subprocess.PIPE,text=True);print(p.stdout.readline().strip(),flush=True);signal.pause()'
        process = subprocess.Popen([sys.executable, '-c', parent], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        child = int(process.stdout.readline())
        pidfd = None

        def exited():
            # Without pidfd_open (this Python build), a zombie or a vanished
            # /proc entry is the exit. The container's init does not reap.
            try:
                return Path('/proc', str(child), 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
            except OSError:
                return True
        try:
            if hasattr(os, 'pidfd_open'):
                pidfd = os.pidfd_open(child)
            stop(process)
            if pidfd is not None:
                self.assertTrue(select.select([pidfd], [], [], 2)[0])
            else:
                deadline = time.monotonic() + 2
                while not exited() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(exited())
            self.assertIsNotNone(process.poll())
        finally:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            if pidfd is not None:
                os.close(pidfd)


class QueueTests(unittest.TestCase):
    def test_status_includes_new_queue_files_and_guard_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, queue = root / 'scheduler', root / 'queue'
            out.mkdir(); queue.mkdir()
            state = dict(finished=dict(status='done', attempts=1),
                         guarded=dict(status='pending', attempts=2, exit_code=75, retry_at=200),
                         active=dict(status='running', attempts=1, pid=123))
            (out / 'state.json').write_text(json.dumps(state))
            (out / 'control.json').write_text(json.dumps(dict(timing_paused=True, reason='six hours')))
            (queue / 'queue_new.json').write_text(json.dumps([dict(name='new_check')]))
            before = (out / 'state.json').read_bytes()
            value = read_status(out, queue, now=100)
            self.assertEqual(value['counts'], dict(done=1, pending=1, running=1, queued=1))
            self.assertTrue(value['timing_paused'])
            rows = {row['name']: row for row in value['steps']}
            self.assertIn('GPU guard retry', rows['guarded']['note'])
            self.assertIn('100s', rows['guarded']['note'])
            self.assertEqual(rows['active']['note'], 'pid=123')
            self.assertEqual((out / 'state.json').read_bytes(), before)

    def test_dependencies_and_cooldown(self):
        steps = {'a': dict(name='a', priority=2), 'b': dict(name='b', priority=1, after=['a']),
                 'c': dict(name='c', after_any=['a'])}
        state = {name: dict(status='pending', retry_at=0) for name in steps}
        self.assertEqual([s['name'] for s in ready_steps(steps, state, 0)[0]], ['a'])
        state['a']['status'] = 'failed'
        ready, skipped = ready_steps(steps, state, 0)
        self.assertEqual(skipped, ['b'])
        self.assertEqual([s['name'] for s in ready], ['c'])
        state['a']['status'] = 'done'
        state['b']['retry_at'] = 10
        self.assertEqual([s['name'] for s in ready_steps(steps, state, 0)[0]], ['c'])

    def test_status_filters_history_to_explicit_queues(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = dict(old_failure=dict(status='failed'),
                         current=dict(status='running', attempts=1, pid=7),
                         accepted=dict(status='done', attempts=1))
            (root / 'state.json').write_text(json.dumps(state))
            queue = root / 'queue_new.json'
            queue.write_text(json.dumps([dict(name='current'), dict(name='new')]))
            before = (root / 'state.json').read_bytes()
            value = read_status(root, queue_files=[queue])
            self.assertEqual(value['counts'], dict(running=1, queued=1))
            self.assertEqual([row['name'] for row in value['steps']], ['current', 'new'])
            self.assertEqual((root / 'state.json').read_bytes(), before)
    def test_six_hours_pause_requires_explicit_resume(self):
        control = {}
        timing_pause(control, True, False, 1, 21600)
        timing_pause(control, True, False, 21601, 21600)
        self.assertTrue(control['timing_paused'])
        timing_pause(control, True, True, 21602, 21600)
        self.assertTrue(control['timing_paused'])

    def test_preparation_freezes_workload_and_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ids.json').write_text(json.dumps([[1] * 128] * 16))
            for phase in ('p', 'd'):
                (root / phase).write_bytes(b'binary')
            config = dict(schema='tilemega.bench.v1', measurement_policy='policy.json', cells={
                'newmodel_B16': dict(model_path='model', prompt_ids='ids.json',
                    workload=dict(isl=128, osl=2048, batch=16), arms=[dict(
                        label='test', kind='tm', prefill='p', decode='d', mode='L1', decode_loop=0)])})
            steps = prepare(config, root, root / 'runs/test')
            self.assertEqual(len(steps), 3)
            frozen = json.loads((root / 'runs/test/arms.json').read_text())
            arm = frozen['cells']['newmodel_B16']['arms'][0]
            check_binaries(arm)
            (root / 'd').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'binary changed'):
                check_binaries(arm)
            invalid = copy.deepcopy(config)
            invalid['cells']['newmodel_B16']['workload']['isl'] = 64
            with self.assertRaisesRegex(ValueError, 'prompt table'):
                prepare(invalid, root, root / 'runs/invalid')


class MeasurementTests(unittest.TestCase):
    def test_identity_join_excludes_trace_and_other_executions(self):
        def row(artifact, trace=False):
            execution = dict(artifact_id=artifact, executor='L1', loop=False, trace=trace)
            execution['execution_id'] = digest(execution)
            return dict(execution_identity=execution)
        self.assertIn('left', checked_join(row('one'), row('one')))
        with self.assertRaisesRegex(ValueError, 'different artifact'):
            checked_join(row('one'), row('two'))
        with self.assertRaisesRegex(ValueError, 'trace timing'):
            checked_join(row('one', True), row('one', True))
        modified = row('one')
        modified['execution_identity']['loop'] = True
        with self.assertRaisesRegex(ValueError, 'modified'):
            checked_join(row('one'), modified)

    def test_shared_workload_in_commands(self):
        cell = dict(model_path='/model', prompt_ids='/ids', workload=dict(isl=128, osl=2048, batch=16))
        for kind in ('tm', 'vllm'):
            arm = dict(kind=kind, root='/root', python=sys.executable, prefill='/p', decode='/d',
                       mode='L2', decode_loop=1, prefill_mode='L1')
            cmd, env = anchor.command(arm, cell, Path('/out'), '/policy')
            self.assertIn('2048', cmd)
            self.assertIn('/ids', cmd)
            self.assertEqual(env['PYTHONPATH'], '/root/python')

    def test_full_model_metrics_and_noise(self):
        measured = metrics(dict(e2e_seconds=2.1, ttft_seconds=.1),
            dict(batch=16, osl=1024), step_bytes=1e9, dram_gbps=980)
        self.assertAlmostEqual(measured['tpot_seconds'], 2 / 1023)
        self.assertAlmostEqual(measured['output_tokens_per_second'], 16 * 1024 / 2.1)
        self.assertAlmostEqual(measured['mbu'], 1023 / (2 * 980))
        self.assertFalse(discernible([1, 1, 1], [.997, .997, .997]))
        self.assertFalse(discernible([.98, 1.02, 1], [.99, .99, .99]))
        self.assertTrue(discernible([1, 1, 1], [.99, .99, .99]))

    def test_round_requires_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'GPU guard'):
                anchor.run_round({}, 'cell', 0, Path('/unused'))

    def test_75_invalidates_whole_round(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'ids').write_text('[[1]]')
            arm = dict(label='test', kind='vllm', root=str(root), python=sys.executable)
            data = dict(measurement_policy='/policy', cells={'cell': dict(arms=[arm, dict(arm, label='other')],
                prompt_ids=str(root / 'ids'), model_path='/model', workload=dict(batch=1, osl=1024))})
            with patch.dict(os.environ, TILEMEGA_GPU_LOCK_HELD='1'), patch(
                    'tilemega.bench.anchor.subprocess.run', return_value=subprocess.CompletedProcess([], 75)) as run:
                self.assertEqual(anchor.run_round(data, 'cell', 0, root / 'out'), 75)
                self.assertEqual(run.call_count, 1)
            self.assertTrue(json.loads((root / 'out/round.json').read_text())['invalidated'])


if __name__ == '__main__':
    unittest.main()
