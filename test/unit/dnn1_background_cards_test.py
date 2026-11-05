"""CPU admission fixtures for main-priority spare GPU borrowing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dnn1_background_cards as cards


class BorrowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / 'runs/dnn2/scan/queue'
        self.queue.mkdir(parents=True)
        (self.queue / 'queue_fixture.json').write_text(json.dumps([
            dict(name='main', gpu=True, env={'CUDA_VISIBLE_DEVICES': '1'})]))
        self.state = self.queue.parent / 'scheduler/state.json'
        self.state.parent.mkdir()

    def reasons(self, gpu=1, held=False):
        with patch.object(cards.base, 'held', return_value=held):
            return cards.main_reservations(gpu, self.root)

    def test_prepared_queue_without_state_reserves_card(self):
        self.assertTrue(self.reasons())

    def test_running_and_pending_guard_reserve_card(self):
        for status in ('running', 'pending', 'retry_wait'):
            self.state.write_text(json.dumps({'main': {'status': status}}))
            self.assertTrue(self.reasons())

    def test_completed_history_does_not_reserve_idle_card(self):
        self.state.write_text(json.dumps({'main': {'status': 'done'}}))
        self.assertEqual(self.reasons(), [])

    def test_scoreboard_flock_reserves_even_before_queue_appears(self):
        self.assertTrue(self.reasons(gpu=0, held=True))

    def test_other_gpu_queue_does_not_reserve_card(self):
        self.assertEqual(self.reasons(gpu=0), [])

    def test_partial_state_write_fails_closed(self):
        self.state.write_text('{')
        self.assertTrue(self.reasons())

    def test_preferred_cards_ignore_main_spare_reservations(self):
        self.assertEqual(self.reasons(gpu=5, held=True), [])


if __name__ == '__main__':
    unittest.main()
