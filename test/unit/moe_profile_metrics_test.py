"""CPU checks for the public metric denominators and overlap accounting."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from moe_profile_metrics import interval_union, summarize_repeats, summarize_round


class MetricsTest(unittest.TestCase):
    def round(self):
        events = [dict(index=0, observed_ns=10_000_000, ids=[1]),
                  dict(index=1, observed_ns=12_000_000, ids=[7]),
                  dict(index=0, observed_ns=30_000_000, ids=[1, 2, 3]),
                  dict(index=1, observed_ns=32_000_000, ids=[7, 8, 9])]
        return summarize_round(2, 128, 3, 0, 40_000_000, events)

    def test_streamed_latency_and_coalescing(self):
        row = self.round()
        self.assertEqual(row['ttft_request_median_ms'], 11)
        self.assertEqual(row['tpot_request_median_ms'], 10)
        self.assertEqual(row['e2e_ms'], 40)
        self.assertEqual(row['output_tokens_per_s'], 150)
        self.assertFalse(row['all_chunks_single_token'])
        self.assertEqual(row['actual_input_tokens'], 256)

    def test_changed_prefix_and_short_output_fail(self):
        with self.assertRaises(ValueError):
            summarize_round(1, 128, 2, 0, 3,
                            [dict(index=0, observed_ns=1, ids=[1]),
                             dict(index=0, observed_ns=2, ids=[2, 3])])
        with self.assertRaises(ValueError):
            summarize_round(1, 128, 2, 0, 3,
                            [dict(index=0, observed_ns=1, ids=[1])])

    def test_union_is_not_kernel_sum(self):
        self.assertEqual(interval_union([(10, 20), (0, 15), (21, 30)]), 29)
        self.assertEqual(interval_union([]), 0)
        with self.assertRaises(ValueError):
            interval_union([(2, 1)])

    def test_repeats_require_same_actual_work(self):
        rows = [self.round() for _ in range(3)]
        self.assertEqual(summarize_repeats(rows)['e2e_ms_range'], 0)
        with self.assertRaises(ValueError):
            summarize_repeats(rows[:2])
        rows[-1]['actual_output_tokens'] -= 1
        with self.assertRaises(ValueError):
            summarize_repeats(rows)


if __name__ == '__main__':
    unittest.main()
