"""Reject invalid timing and changed evidence before a B5 selection resume."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tilemega.tp.resume_tp import checked, completed_timing, check_unlaunched_retry, finish_qualified_final
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

    def test_external_interference_does_not_hide_a_numerical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / 'final/correctness'
            (failed / 'execution/guard').mkdir(parents=True)
            (failed / 'execution/PROCESS.json').write_text(json.dumps(dict(returncode=75)))
            (failed / 'execution/guard/guard_result.json').write_text(json.dumps(dict(code=75, reason='running interference GPU3')))
            (failed / 'hf_check.json').write_text(json.dumps({'pass': False}))
            (root / 'selected_pin.json').write_text(json.dumps(dict(mx_decode_v2={'sentinel': 1}, evidence=dict(selection={'selected': 0}))))
            candidate = dict(choice=dict(predicted_ns=1), timing=dict(fresh_process_ns=[1, 1]))
            cell = SimpleNamespace(out=root, cost_sha='a' * 64)
            with patch('tilemega.tp.resume_tp.select_measured', return_value={'selected': 0}), patch('tilemega.tp.resume_tp.pin', return_value={'mx_decode_v2': {'sentinel': 1}}):
                with self.assertRaisesRegex(ValueError, 'HF correctness failure'):
                    finish_qualified_final(SimpleNamespace(tp=4), cell, [candidate], [], {})

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
