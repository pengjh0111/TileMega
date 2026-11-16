"""Check P-SEL mirrored order and the measured-noise prediction tiebreak."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'dnn3_select', Path(__file__).resolve().parents[2] / 'scripts/dnn3_select.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def rows(samples):
    used = dict.fromkeys(range(4), 0)
    result = []
    for rank in selection.ORDER:
        result.append(dict(rank=rank, median_ms=samples[rank][used[rank]]))
        used[rank] += 1
    return result


class SelectionTest(unittest.TestCase):
    def test_clear_winner(self):
        d = selection.select(rows([[4, 4], [3, 3], [2, 2], [1, 1]]))
        self.assertEqual(d['winner'], 3)
        self.assertFalse(d['prediction_tiebreak'])

    def test_half_percent_tie_prefers_prediction(self):
        d = selection.select(rows([[1.004, 1.004], [1, 1], [2, 2], [3, 3]]))
        self.assertEqual(d['winner'], 0)
        self.assertEqual(d['fastest'], 1)
        self.assertTrue(d['prediction_tiebreak'])

    def test_range_threshold_uses_both_positions(self):
        d = selection.select(rows([[1.02, 1.04], [1, 1], [2, 2], [3, 3]]))
        self.assertEqual(d['winner'], 0)
        self.assertAlmostEqual(d['pair_threshold'], .04 / 1.03)

    def test_only_the_two_fastest_are_tied(self):
        d = selection.select(rows([[1.008, 1.008], [1.004, 1.004], [1, 1], [3, 3]]))
        self.assertEqual(d['winner'], 1)

    def test_rejects_wrong_order(self):
        bad = rows([[1, 1]] * 4)
        bad[0], bad[1] = bad[1], bad[0]
        with self.assertRaisesRegex(ValueError, 'A B C D D C B A'):
            selection.select(bad)

    def test_rejects_invalid_measurements(self):
        for invalid in (0, -1, float('nan'), float('inf')):
            bad = rows([[1, 1]] * 4)
            bad[0]['median_ms'] = invalid
            with self.assertRaisesRegex(ValueError, 'finite positive'):
                selection.select(bad)


if __name__ == '__main__':
    unittest.main()
