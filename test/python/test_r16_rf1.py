"""RF1 significance requires a material request gap beyond both process ranges."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('rf1_analysis', ROOT/'scripts/analyze_r16_rf1.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RF1ComparisonTests(unittest.TestCase):
    def test_large_request_gap_inside_process_noise_is_inconclusive(self):
        result = analysis.comparison([101, 105, 109], [100, 102, 104])
        self.assertGreater(result['on_change_percent'], 0.5)
        self.assertFalse(result['distinguishable'])

    def test_stable_regression_and_improvement_have_opposite_decisions(self):
        regression = analysis.comparison([102, 102.1, 102.2], [100, 100.1, 100.2])
        improvement = analysis.comparison([98, 98.1, 98.2], [100, 100.1, 100.2])
        self.assertTrue(regression['distinguishable'])
        self.assertTrue(regression['on_regression'])
        self.assertTrue(improvement['distinguishable'])
        self.assertFalse(improvement['on_regression'])

    def test_small_stable_gap_does_not_pass_materiality_threshold(self):
        result = analysis.comparison([100.3]*3, [100]*3)
        self.assertFalse(result['distinguishable'])


if __name__ == '__main__':
    unittest.main()
