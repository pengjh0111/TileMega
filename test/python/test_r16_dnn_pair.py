"""Use the fixed noise criterion, including tails, for a scoped recommendation."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from analyze_r16_dnn_pair import comparison


class Pair(unittest.TestCase):
    def rows(self, center, tail):
        return [dict(median_ms=center+d, p90_ms=tail+d) for d in (0,.01,-.01)]

    def test_faster_median_and_tail_are_eligible(self):
        value = comparison(self.rows(6,6.1), self.rows(5.7,5.8))
        self.assertTrue(value['eligible_in_measured_grid'])

    def test_noise_does_not_promote_and_tail_regression_blocks(self):
        a = self.rows(6,6.1)
        self.assertFalse(comparison(a,self.rows(5.99,6.09))['eligible_in_measured_grid'])
        self.assertFalse(comparison(a,self.rows(5.7,6.4))['eligible_in_measured_grid'])


if __name__ == '__main__':
    unittest.main()
