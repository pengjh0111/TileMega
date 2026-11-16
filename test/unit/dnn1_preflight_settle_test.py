import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('settle',
    Path(__file__).resolve().parents[2] / 'scripts/dnn1_preflight_settle.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SettleTest(unittest.TestCase):
    def run_protocol(self, memories, *, owners=None, timeout=120, admission=None):
        now = [0.]
        rows = iter(memories)
        self.observed = []
        def observe():
            value = next(rows)
            row = dict(owners=owners or {}, used_mib=value, free_mib=81000)
            self.observed.append(row)
            return [row]
        def sleep(seconds):
            now[0] += seconds
        return MODULE.settle(observe, required_samples=6, interval_s=5,
            timeout_s=timeout, needs_free_mib=8192,
            admission_ok=admission or (lambda: True),
            clock=lambda: now[0], sleep=sleep)

    def test_high_reading_resets_full_six_sample_requirement(self):
        result = self.run_protocol([0, 1, 0, 0, 4, 0, 1, 0, 1, 0, 0])
        self.assertEqual(result['status'], 'idle_observed')
        self.assertEqual(len(result['samples']), 11)
        self.assertEqual(result['samples'][4]['rows'][0]['used_mib'], 4)
        self.assertEqual(result['final_streak'], 6)
        self.assertEqual(result['final_rows'], result['samples'][-1]['rows'])
        self.assertEqual(len(self.observed), 11)

    def test_any_owner_yields_before_payload_even_at_zero_memory(self):
        result = self.run_protocol([0], owners={123: 0})
        self.assertEqual(result['status'], 'occupied')
        self.assertEqual(len(result['samples']), 1)
        self.assertNotIn('final_rows', result)

    def test_nonconsecutive_idle_observations_never_accumulate(self):
        result = self.run_protocol([0, 0, 4, 0, 0, 4, 0, 0], timeout=35)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(result['final_streak'], 6)
        self.assertNotIn('final_rows', result)

    def test_new_main_reservation_yields_and_retains_prior_samples(self):
        calls = [0]
        def admission():
            calls[0] += 1
            return calls[0] < 3
        result = self.run_protocol([0, 0, 0], admission=admission)
        self.assertEqual(result['status'], 'yield_reservation')
        self.assertEqual(len(result['samples']), 2)
        self.assertNotIn('final_rows', result)

    def test_missing_telemetry_fails_closed(self):
        result = self.run_protocol([])
        self.assertEqual(result['status'], 'telemetry_unavailable')
        self.assertEqual(result['samples'], [])

    def test_late_observation_does_not_pass_timeout(self):
        now = [0]
        def observe():
            now[0] = 121
            return [dict(owners={}, used_mib=0, free_mib=81000)]
        result = MODULE.settle(observe, required_samples=1, interval_s=5,
            timeout_s=120, needs_free_mib=8192, admission_ok=lambda: True,
            clock=lambda: now[0], sleep=lambda _: None)
        self.assertEqual(result['status'], 'timeout')
        self.assertNotIn('final_rows', result)


if __name__ == '__main__':
    unittest.main()
