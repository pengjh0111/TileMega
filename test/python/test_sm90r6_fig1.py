from pathlib import Path
import statistics
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from sm90r6_fig1 import critical_segments


def row(begin, gate, arrive, release):
    return dict(t_begin=begin, m2=gate, t_tasks_end=arrive, t_release=release, past=580)


class CriticalSegments(unittest.TestCase):
    def test_last_arriving_cta_owns_all_segment_marks(self):
        result = critical_segments([row(1000, 5000, 6000, 11000),
                                    row(2000, 4000, 8000, 9000)], 'expert_gate_up')
        self.assertEqual(result, dict(stage=7, barrier_tail=1, gate_up=2, down=4))

    def test_missing_gate_mark_is_not_silently_filled(self):
        with self.assertRaises(ValueError):
            critical_segments([row(1000, 0, 8000, 9000)], 'expert_gate_up')

    def test_subtract_instance_timestamps_before_medians(self):
        rows = [row(1000, 2000, 4000, 5000), row(1000, 11000, 12000, 13000),
                row(1000, 3000, 21000, 22000)]
        down = statistics.median(critical_segments([r], 'expert_gate_up')['down'] for r in rows)
        self.assertEqual(down, 2)
        self.assertNotEqual(down, (statistics.median(r['t_tasks_end'] for r in rows) -
                                  statistics.median(r['m2'] for r in rows)) / 1000)


if __name__ == '__main__':
    unittest.main()
