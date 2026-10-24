import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('joint_critical', Path(__file__).resolve().parents[2] /
    'scripts/analyze_sm90r2_joint_ring_critical.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CriticalSpans(unittest.TestCase):
    def test_local_critical_is_not_latest_absolute_cta(self):
        r = module.summarize_trial([
            dict(cta=0, issue=100, first=120, release=180, final=200),
            dict(cta=1, issue=150, first=155, release=225, final=230)])
        self.assertEqual((r['span_critical_cta'], r['latest_final_cta']), (0, 1))
        self.assertEqual(r['max_local_span_ns'], 100)
        self.assertEqual(r['absolute_envelope_with_launch_skew_ns'], 130)
        self.assertEqual((r['selected_first_ns'], r['selected_main_ns'], r['selected_tail_ns']), (20, 60, 20))

    def test_large_integer_epochs_preserve_single_nanosecond(self):
        t = 2**60
        r = module.summarize_trial([dict(cta=0, issue=t, first=t+1, release=t+2, final=t+3)])
        self.assertEqual(r['max_local_span_ns'], 3)
        self.assertEqual(r['selected_first_ns'], 1)

    def test_bad_span_and_duplicate_cta_reject(self):
        r = dict(cta=0, issue=100, first=120, release=110, final=130)
        with self.assertRaises(ValueError):
            module.summarize_trial([r])
        r['release'] = 125
        with self.assertRaises(ValueError):
            module.summarize_trial([r, r])


if __name__ == '__main__':
    unittest.main()
