"""Reject invalid timing and changed evidence before a B5 selection resume."""
import json
from pathlib import Path
import tempfile
import unittest

from tilemega.tp.resume_tp import checked, completed_timing, check_unlaunched_retry
from tilemega.tp.solve_tp import ref


class ResumeTests(unittest.TestCase):
    def test_changed_archived_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text('{}')
            identity = ref(path)
            self.assertEqual(checked(identity), path)
            path.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'changed archived evidence'):
                checked(identity)

    def test_only_preflight_rejection_leaves_the_retry_unused(self):
        with tempfile.TemporaryDirectory() as directory:
            retry = Path(directory)
            execution = retry / 'timing_execution/execution'
            guard = execution / 'guard'
            guard.mkdir(parents=True)
            (execution / 'PROCESS.json').write_text(json.dumps(dict(returncode=75)))
            (guard / 'guard_result.json').write_text(json.dumps(dict(code=75, reason='preflight occupied GPU3')))
            check_unlaunched_retry(retry)
            (guard / 'pgid').write_text('1234')
            with self.assertRaisesRegex(ValueError, 'launched timing retry'):
                check_unlaunched_retry(retry)

    def test_guard_invalidated_times_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def record(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return ref(path)
            timing = dict(guard=record('guard.json', dict(code=75)),
                          driver=record('driver.json', {}),
                          processes=[record(f'p{i}.json', dict(PID=i + 1, returncode=0)) for i in range(2)],
                          measurements=[record(f'm{i}.json', dict(timed_tokens_identical=True,
                                                               rank_tokens_identical=True)) for i in range(2)])
            with self.assertRaisesRegex(ValueError, 'invalid timing guard'):
                completed_timing(timing)
            timing['guard'] = record('guard_pass.json', dict(code=0))
            completed_timing(timing)
            timing['processes'][1] = timing['processes'][0]
            with self.assertRaisesRegex(ValueError, 'fresh timing processes'):
                completed_timing(timing)


if __name__ == '__main__':
    unittest.main()
