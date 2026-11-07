"""Check DT-ANCHOR eligibility and the full four-arm, five-process protocol."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'dnn3_resample', Path(__file__).resolve().parents[2] / 'scripts/dnn3_resample.py')
resample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resample)


class ResampleTest(unittest.TestCase):
    def test_only_measured_endpoint_drift_permits_resampling(self):
        anchor = dict(status='passed', protocol='P-ANCHOR',
                      action='resample_trt_tc_all_batches', anchors=[
                          dict(batch=b, frozen_ms=1., median_ms=1.03) for b in (1, 16)])
        resample.require_resample(anchor)
        anchor['anchors'][0]['median_ms'] = 1.
        resample.require_resample(anchor)
        anchor['anchors'][1]['median_ms'] = 1.02
        with self.assertRaises(ValueError):
            resample.require_resample(anchor)
        anchor['anchors'][1]['median_ms'] = .97
        resample.require_resample(anchor)
        anchor['status'] = 'partial'
        with self.assertRaises(ValueError):
            resample.require_resample(anchor)

    def test_three_rounds_cover_each_bf16_arm_in_mirrored_order(self):
        rounds = [[a['arm'] for a in arms] for arms in resample.order()]
        self.assertEqual(len(rounds[0]), 4)
        self.assertEqual(rounds[0], rounds[2])
        self.assertEqual(rounds[1], list(reversed(rounds[0])))
        self.assertTrue(all(a['precision'] == 'bf16' and a['system'] != 'tvm'
                            for arms in resample.order() for a in arms))
        self.assertEqual(resample.BATCHES, (1, 2, 4, 8, 16))


if __name__ == '__main__':
    unittest.main()
