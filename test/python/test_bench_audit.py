import json
import tempfile
import unittest
from pathlib import Path

from tilemega.bench.audit import accepted_rounds, compare


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'state.json'
        rows = {}
        for rnd in range(3):
            folder = self.root / f'round{rnd}' / 'attempt2'
            folder.mkdir(parents=True)
            (folder / 'guard.jsonl').write_text(json.dumps(dict(phase='after', exit_code=0)) + '\n')
            arms = {}
            for arm, seconds in [('reference', 1.0), ('candidate', 1.1)]:
                path = folder / arm
                path.mkdir()
                (path / 'command.json').write_text(json.dumps(dict(binaries=dict(decode=dict(sha256=arm)))))
                (path / 'tokens_N16_run1.json').write_text('[1, 2, 3]')
                arms[arm] = dict(exit_code=0, e2e_seconds=seconds, ttft_seconds=.1)
            (folder / 'round.json').write_text(json.dumps(dict(cell='cell', round=rnd,
                workload=dict(batch=1, osl=16), arms=arms)))
            rows[f'round{rnd}'] = dict(status='done', attempts=2, out=str(folder))
            # A discarded attempt must not contribute even if it has a round file.
            (folder.parent / 'attempt1').mkdir()
            (folder.parent / 'attempt1/round.json').write_text('invalid discarded data')
        self.state.write_text(json.dumps(rows))

    def test_final_attempt_only(self):
        result = accepted_rounds(self.state, 'round')
        row = compare(result, 'reference', 'candidate')[0]
        self.assertTrue(row['discernible'])
        self.assertTrue(row['tokens_identical'])
        self.assertEqual(len(result['accepted_attempts']), 3)

    def test_reject_guard_failure(self):
        (self.root / 'round0/attempt2/guard.jsonl').write_text(json.dumps(dict(phase='after', exit_code=75)))
        with self.assertRaisesRegex(ValueError, 'guard'):
            accepted_rounds(self.state, 'round')

    def test_reject_binary_change(self):
        p = self.root / 'round0/attempt2/reference/command.json'
        p.write_text(json.dumps(dict(binaries=dict(decode=dict(sha256='different')))))
        with self.assertRaisesRegex(ValueError, 'binary identity'):
            accepted_rounds(self.state, 'round')

    def test_reject_incomplete_round(self):
        rows = json.loads(self.state.read_text())
        rows['round0']['status'] = 'pending'
        self.state.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'completed'):
            accepted_rounds(self.state, 'round')


if __name__ == '__main__':
    unittest.main()
