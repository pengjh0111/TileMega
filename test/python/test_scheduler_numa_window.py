"""A busy NUMA timing window must leave correctness work schedulable."""
import fcntl
from pathlib import Path
import tempfile
import unittest

from tilemega.bench.scheduler import numa_window_available


class NumaWindow(unittest.TestCase):
    def test_busy_node_defers_only_timing_and_preserves_other_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with (folder/'tilemega_numa1.lock').open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                timed = dict(gpu=True, env={'TILEMEGA_NUMA_NODE': '1'})
                numerical = dict(timed, timing=False)
                other = dict(gpu=True, env={'TILEMEGA_NUMA_NODE': '0'})
                self.assertFalse(numa_window_available(timed, {}, folder))
                self.assertTrue(numa_window_available(numerical, {}, folder))
                self.assertTrue(numa_window_available(other, {}, folder))
                ready = [timed, numerical]
                self.assertIs(next(s for s in ready if numa_window_available(s, {}, folder)), numerical)
            self.assertTrue(numa_window_available(timed, {}, folder))

    def test_hint_releases_lock_and_does_not_override_trial_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            step = dict(gpu=True, env={'TILEMEGA_NUMA_NODE': '1'})
            self.assertTrue(numa_window_available(step, {}, folder))
            with (folder/'tilemega_numa1.lock').open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(numa_window_available(step, {}, folder))
                self.assertTrue(numa_window_available(step, {'TILEMEGA_NUMA_CONCURRENCY_TRIAL': '1'}, folder))
                self.assertFalse(numa_window_available(step, {'TILEMEGA_NUMA_CONCURRENCY_VERIFIED': '0'}, folder))


if __name__ == '__main__':
    unittest.main()
