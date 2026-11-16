"""CPU fixtures retain transient boundary observations without lowering gates."""
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from dnn1_baseline_boundaries import idle_boundary, final_boundary
import dnn1_background_python as provider


class BoundaryTest(unittest.TestCase):
    def observe(self, sequence, timeout=10):
        ticks = [0.]
        self.calls = []
        def sample(_):
            i = len(self.calls); row = sequence[min(i, len(sequence)-1)]
            self.calls.append(row); return row
        result = idle_boundary({}, timeout_s=timeout, observe=sample,
            clock=lambda: ticks[0], sleep=lambda delay: ticks.__setitem__(0, ticks[0]+delay))
        return result

    def test_memory_high_reading_resets_streak_and_is_preserved(self):
        sequence = [dict(owners={}, used_mib=m) for m in (0, 4, 1, 0)]
        result = self.observe(sequence)
        self.assertEqual(len(result['samples']), 4)
        self.assertIs(final_boundary(result), sequence[-1])
        self.assertEqual(len(self.calls), 4)

    def test_owner_resets_streak_even_with_zero_memory(self):
        result = self.observe([dict(owners={}, used_mib=0),
            dict(owners={'123': 1}, used_mib=0), dict(owners={}, used_mib=1),
            dict(owners={}, used_mib=0)])
        self.assertEqual(len(result['samples']), 4)

    def test_persistent_four_mib_times_out_and_is_never_accepted(self):
        result = self.observe([dict(owners={}, used_mib=4)], timeout=.5)
        self.assertEqual(result['status'], 'timeout')
        with self.assertRaises(ValueError): final_boundary(result)

    def test_one_good_observation_is_insufficient(self):
        result = self.observe([dict(owners={}, used_mib=0)], timeout=0)
        self.assertEqual(result['status'], 'timeout')

    def test_missing_owner_telemetry_fails_closed(self):
        result = self.observe([dict(used_mib=0)])
        self.assertEqual(result['status'], 'telemetry_unavailable')


class ProviderTest(unittest.TestCase):
    def test_busy_physical_lock_remains_held_by_its_original_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'physical.lock'
            with path.open('a') as owner, path.open('a') as borrower:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                code = provider.nonblocking_guard(lambda: fcntl.flock(borrower, fcntl.LOCK_EX), Path(temp))
                self.assertEqual(code, 75)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(borrower, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_scheduler_yields_to_queued_main_at_every_idle_check(self):
        row = {'gpu_index': 1}
        with patch.object(provider, 'main_reservations', return_value=['queued main']), \
             patch.object(provider.base, 'held', return_value=False):
            self.assertFalse(provider.admitted(row, lambda *a: True, {}, scheduler=True))

    def test_scheduler_does_not_queue_behind_physical_lock(self):
        with patch.object(provider, 'main_reservations', return_value=[]), \
             patch.object(provider.base, 'held', return_value=True):
            self.assertFalse(provider.admitted({'gpu_index': 0}, lambda *a: True, {}, scheduler=True))

    def test_guard_lock_is_nonblocking_and_function_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            original = fcntl.flock
            operations = []
            def fake(fd, op):
                operations.append(op)
                raise BlockingIOError('held by another guard')
            with patch.object(fcntl, 'flock', side_effect=fake) as patched:
                code = provider.nonblocking_guard(lambda: fcntl.flock(9, fcntl.LOCK_EX), Path(temp))
                self.assertIs(fcntl.flock, patched)
            self.assertIs(fcntl.flock, original)
            self.assertEqual(code, 75)
            self.assertEqual(operations, [fcntl.LOCK_EX | fcntl.LOCK_NB])
            self.assertTrue((Path(temp) / 'guard_result.json').exists())


if __name__ == '__main__':
    unittest.main()
