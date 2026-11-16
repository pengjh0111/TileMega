#!/usr/bin/env python3
"""Discard partial cohorts and preserve transient GPU rejection semantics."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import dnn4_cohort


class CohortRetry(unittest.TestCase):
    def run_failure(self, code):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root/'unexpected-second-child'
            steps = [dict(name='first', command=[sys.executable, '-c', f'raise SystemExit({code})'],
                          cwd=str(root), env={}, timeout_s=10),
                     dict(name='second', command=[sys.executable, '-c',
                          'from pathlib import Path;Path('+repr(str(marker))+').touch()'],
                          cwd=str(root), env={}, timeout_s=10)]
            plan = root/'plan.json'
            plan.write_text(json.dumps(dict(frozen={}, steps=steps)))
            expected = SystemExit if code == 75 else RuntimeError
            with patch('tilemega.serving.baseline_common.require_guard'), self.assertRaises(expected) as caught:
                dnn4_cohort.execute(plan, root/'out')
            if code == 75:
                self.assertEqual(caught.exception.code, 75)
            result = json.loads((root/'out/result.json').read_text())
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(len(result['processes']), 1)
            self.assertEqual(result['processes'][0]['exit_code'], code)
            self.assertFalse(marker.exists())

    def test_transient_occupancy_retries_whole_cohort(self):
        self.run_failure(75)

    def test_hard_failure_remains_terminal(self):
        self.run_failure(1)


if __name__ == '__main__':
    unittest.main()
