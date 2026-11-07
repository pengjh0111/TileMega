"""Reject incomplete final grids and unanchored cross-card comparisons."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'dnn3_results', Path(__file__).resolve().parents[2] / 'scripts/dnn3_results.py')
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


class ResultsTest(unittest.TestCase):
    def test_complete_grid_required(self):
        cells = [dict(model=m, batch=b, status='passed', protocol='P-SEL; winner P-CORR; P-DNN')
                 for m in results.MODELS for b in results.BATCHES]
        results.coverage(cells)
        with self.assertRaises(ValueError):
            results.coverage(cells[:-1])
        with self.assertRaises(ValueError):
            results.coverage(cells[:-1] + cells[:1])
        cells[0]['status'] = 'partial'
        with self.assertRaises(ValueError):
            results.coverage(cells)

    def test_geometric_ratio(self):
        self.assertAlmostEqual(results.geometric([.5, 2]), 1)
        self.assertAlmostEqual(results.geometric([1, 4]), 2)
        for values in ([], [0], [float('nan')]):
            with self.assertRaises(ValueError):
                results.geometric(values)

    def test_retention_requires_same_card_and_both_endpoints(self):
        cell = dict(model='bert', batch=4, gpu=2)
        anchor = dict(status='passed', protocol='P-ANCHOR', model='bert', gpu=2,
                      action='retain_frozen', anchors=[dict(batch=b, action='retain_frozen',
                      frozen_ms=1., median_ms=1.) for b in (1, 16)])
        ref = dict(model='bert', batch='4', precision='bf16', status='pass', median_ms='2')
        self.assertEqual(results.baseline(cell, anchor, [ref], []), ref)
        anchor['gpu'] = 3
        with self.assertRaises(ValueError):
            results.baseline(cell, anchor, [ref], [])
        anchor['gpu'] = 2; anchor['anchors'][0]['median_ms'] = .97
        with self.assertRaises(ValueError):
            results.baseline(cell, anchor, [ref], [])

    def test_resampling_cannot_reuse_frozen_baselines(self):
        cell = dict(model='bert', batch=4, gpu=2)
        anchor = dict(status='passed', protocol='P-ANCHOR', model='bert', gpu=2,
                      action='resample_trt_tc_all_batches', anchors=[dict(batch=b) for b in (1, 16)])
        with self.assertRaisesRegex(ValueError, 'all five batches'):
            results.baseline(cell, anchor, [], [])


if __name__ == '__main__':
    unittest.main()
