# SPDX-License-Identifier: BSD-3-Clause
"""Clean scheduler receipts remain separate from historical queues."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_r16_candidates import read_states, collect

class SchedulerRootsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'runs/r16/scheduler'
        self.clean = self.root / 'runs/r16/scheduler-rf2-final-v1'

    def state(self, directory, gpu, name):
        path = directory / f'gpu{gpu}/state.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({name: {'status': 'done'}}))

    def test_default_preserves_old_root(self):
        self.state(self.old, 7, 'old')
        self.state(self.clean, 3, 'new')
        state, cards = read_states(self.root)
        self.assertEqual(set(state), {'old'})
        self.assertEqual(cards, {'old': 7})

    def test_explicit_clean_root_and_repeated_path(self):
        self.state(self.old, 7, 'old')
        self.state(self.clean, 3, 'new')
        state, cards = read_states(self.root, [self.clean, self.clean])
        self.assertEqual(set(state), {'new'})
        self.assertEqual(cards, {'new': 3})
        self.assertEqual(collect(self.root, [], [self.clean])['rejections'], [])

    def test_duplicate_job_across_roots_is_rejected(self):
        self.state(self.old, 7, 'same')
        self.state(self.clean, 3, 'same')
        with self.assertRaisesRegex(ValueError, 'ambiguous scheduler job name'):
            read_states(self.root, [self.old, self.clean])

if __name__ == '__main__':
    unittest.main()
