"""Validate frozen BF16 arm selection and the two-sided DT-ANCHOR threshold."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'dnn3_anchor', Path(__file__).resolve().parents[2] / 'scripts/dnn3_anchor.py')
anchor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anchor)


class AnchorTest(unittest.TestCase):
    def test_two_sided_boundary(self):
        for measured in (.98, 1, 1.02):
            self.assertEqual(anchor.decision(1, measured)['action'], 'retain_frozen')
        for measured in (.97999, 1.02001):
            self.assertEqual(anchor.decision(1, measured)['action'], 'resample_trt_tc_all_batches')

    def test_invalid_medians(self):
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                anchor.decision(1, invalid)
            with self.assertRaises(ValueError):
                anchor.decision(invalid, 1)

    def test_fastest_ignores_fp16_failed_and_other_cells(self):
        base = dict(model='resnet18', batch='1', precision='bf16', status='pass',
                    arm='tensorrt-bf16', median_ms='2')
        rows = [base, dict(base, precision='fp16', median_ms='.1'),
                dict(base, status='fail', median_ms='.1'), dict(base, batch='16', median_ms='.1'),
                dict(base, model='bert', median_ms='.1'),
                dict(base, arm='torchcompile-max-autotune-freezing', median_ms='1')]
        self.assertEqual(anchor.fastest(rows, 'resnet18', 1)['median_ms'], '1')

    def test_unsupported_fastest_is_not_replaced(self):
        with self.assertRaisesRegex(ValueError, 'separate baseline adapter'):
            anchor.fastest([dict(model='bert', batch='1', precision='bf16', status='pass',
                arm='tvm-bf16', median_ms='1')], 'bert', 1)

    def test_missing_or_invalid_baseline(self):
        with self.assertRaises(ValueError):
            anchor.fastest([], 'bert', 1)
        with self.assertRaises(ValueError):
            anchor.fastest([dict(model='bert', batch='1', precision='bf16', status='pass',
                arm='tensorrt-bf16', median_ms='nan')], 'bert', 1)


if __name__ == '__main__':
    unittest.main()
