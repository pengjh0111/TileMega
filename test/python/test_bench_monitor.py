import copy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest

from tilemega.bench.monitor import Observer, Changes, snapshot


def value():
    return dict(out='/scheduler', scheduler_pid=10, control={}, steps={
        'active': dict(timeout_s=300), 'old': dict(timeout_s=30)},
        state={'active': dict(status='running', attempts=1, pid=11, started=100),
               'old': dict(status='failed', attempts=1, exit_code=1)},
        counts=dict(running=1, failed=1))


class MonitorTest(unittest.TestCase):
    def test_history_and_normal_guard_wait_are_not_new_failures(self):
        observer = Observer()
        initial = value()
        self.assertEqual([r['event'] for r in observer.inspect(initial, 100, lambda pid: 'start')],
                         ['baseline'])
        later = copy.deepcopy(initial)
        later['state']['active'].update(status='pending', exit_code=75, retry_at=1000)
        self.assertEqual(observer.inspect(later, 200, lambda pid: 'start'), [])

    def test_new_failure_and_completion_are_distinct_from_acceptance(self):
        observer = Observer(value())
        later = value()
        later['state']['active'].update(status='failed', exit_code=124)
        later['counts'] = dict(failed=2)
        events = observer.inspect(later, 200)
        self.assertEqual([r['event'] for r in events], ['failure', 'completed'])
        self.assertEqual(events[0]['exit_code'], 124)
        self.assertFalse(events[1]['all_done'])
        self.assertEqual(observer.inspect(later, 201), [])

    def test_exited_or_reused_pid_requires_confirmation(self):
        observer = Observer()
        data = value()
        observer.inspect(data, 100, lambda pid: 'original')
        self.assertEqual(observer.inspect(data, 101, lambda pid: None), [])
        events = observer.inspect(data, 221, lambda pid: 'reused')
        self.assertEqual({r['step'] for r in events}, {'scheduler', 'active'})
        self.assertEqual(observer.inspect(data, 222, lambda pid: 'reused'), [])

    def test_configured_timeout_not_log_silence_defines_overrun(self):
        observer = Observer(value())
        self.assertEqual(observer.inspect(value(), 519, lambda pid: 'start'), [])
        events = observer.inspect(value(), 521, lambda pid: 'start')
        self.assertEqual([r['event'] for r in events], ['timeout_overrun'])
        data = value()
        data['steps']['active']['gpu'] = True
        self.assertEqual(Observer(value()).inspect(data, 521, lambda pid: 'start'), [])

    def test_queue_filter_and_directory_watch_survive_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = root / 'queue.json'
            queue.write_text(json.dumps([dict(name='new'), dict(name='selected')]))
            state_path = root / 'state.json'
            original = json.dumps(dict(selected=dict(status='done'), unrelated=dict(status='running')))
            state_path.write_text(original)
            (root / 'scheduler.pid').write_text(str(os.getpid()))
            result = snapshot(root, [queue])
            self.assertEqual(result['counts'], dict(queued=1, done=1))
            self.assertNotIn('unrelated', result['state'])
            self.assertEqual(state_path.read_text(), original)
            changes = Changes([root])
            try:
                self.assertGreaterEqual(changes.fd, 0)
                temp = root / 'state.tmp'
                temp.write_text('{}')
                temp.replace(state_path)
                changes.wait(.1)
            finally:
                changes.close()

    def test_event_wait_ignores_baseline_and_preserves_resume_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / 'events.jsonl'
            events.write_text(json.dumps(dict(event='baseline')) + '\n' +
                              json.dumps(dict(event='failure', step='new')) + '\n')
            command = [sys.executable, '-m', 'tilemega.bench.monitor', 'next',
                       '--events', str(events), '--timeout-s', '.1']
            result = json.loads(subprocess.check_output(command, text=True))
            self.assertEqual(result['step'], 'new')
            self.assertEqual(result['next_offset'], events.stat().st_size)
            resumed = json.loads(subprocess.check_output(command + [
                '--offset', str(result['next_offset'])], text=True))
            self.assertEqual(resumed['event'], 'wait_timeout')

    def test_six_hour_pause_is_reported_once(self):
        observer = Observer(value())
        data = value()
        data['control'] = dict(timing_paused=True, reason='no idle timing window for six hours')
        self.assertEqual([r['event'] for r in observer.inspect(data, 200, lambda pid: 'start')],
                         ['timing_paused'])
        self.assertEqual(observer.inspect(data, 201, lambda pid: 'start'), [])

    def test_watcher_records_terminal_event_without_editing_scheduler_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scheduler = root / 'scheduler'
            scheduler.mkdir()
            events = root / 'monitor/events.jsonl'
            queue = root / 'queue.json'
            queue.write_text(json.dumps([dict(name='check', timeout_s=60)]))
            state = scheduler / 'state.json'
            state.write_text(json.dumps(dict(check=dict(status='running', pid=os.getpid()))))
            (scheduler / 'scheduler.pid').write_text(str(os.getpid()))
            command = [sys.executable, '-m', 'tilemega.bench.monitor', 'watch',
                       '--out', str(scheduler), '--queue-file', str(queue), '--events', str(events)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(select.select([process.stdout], [], [], 3)[0])
                self.assertEqual(json.loads(process.stdout.readline())['event'], 'baseline')
                final = json.dumps(dict(check=dict(status='failed', exit_code=1)))
                temp = scheduler / 'state.tmp'
                temp.write_text(final)
                temp.replace(state)
                output, error = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0, error)
                self.assertEqual([json.loads(line)['event'] for line in output.splitlines()],
                                 ['failure', 'completed'])
                self.assertEqual(state.read_text(), final)
                self.assertTrue(events.with_suffix('.cursor.json').is_file())
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()


if __name__ == '__main__':
    unittest.main()
