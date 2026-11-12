"""Reference metric variation must never hide changed or inaccurate outputs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from dnn4_bert_model_cohort import validate_bert_checks


def cohort():
    check = dict(reference='exported BF16 checkpoint promoted to FP32',
                 synthetic_weights=False, batch=16, passed=True, artifact_id='artifact',
                 checker_sha256='checker', bridge_sha256='bridge', archive_sha256='archive',
                 source_checkpoint_sha256='weights', input_tensors_sha256='inputs')
    check['cases'] = [dict(epoch=e, mode=m, outputs=[
        dict(name=name, cosine_min=.99995, max_error=.016679663,
             bf16_max_error=.02734375, output_sha256=digest*64)
        for name, digest in [('layer_norm_24', 'a'), ('tanh', 'b')]])
        for e in range(3) for m in (1, 2)]
    return [copy.deepcopy(check) for _ in range(5)]


class BertCohortTest(unittest.TestCase):
    def test_metric_variation_preserves_output_gate(self):
        checks = cohort()
        checks[3]['cases'][0]['outputs'][1]['max_error'] = .016677532
        result = validate_bert_checks(checks, 'artifact')
        metric = next(r for r in result['metric_ranges']
                      if r['output'] == 'tanh' and r['metric'] == 'max_error')
        self.assertLess(metric['min'], metric['max'])

    def test_corruptions(self):
        changes = [
            lambda c: c.pop(),
            lambda c: c[4].update(passed=False),
            lambda c: c[4].update(artifact_id='other'),
            lambda c: c[4].update(source_checkpoint_sha256='other'),
            lambda c: c[4].update(synthetic_weights=True),
            lambda c: c[4]['cases'].pop(),
            lambda c: c[4]['cases'][0].update(mode=2),
            lambda c: c[4]['cases'][0]['outputs'].pop(),
            lambda c: c[4]['cases'][0]['outputs'][0].update(output_sha256='c'*64),
            lambda c: c[4]['cases'][0]['outputs'][0].update(output_sha256='bad'),
            lambda c: c[4]['cases'][0]['outputs'][0].update(cosine_min=.99899),
            lambda c: c[4]['cases'][0]['outputs'][0].update(max_error=float('nan')),
            lambda c: c[4]['cases'][0]['outputs'][0].update(bf16_max_error=-1),
        ]
        for change in changes:
            with self.subTest(change=change):
                checks = cohort()
                change(checks)
                with self.assertRaises(ValueError):
                    validate_bert_checks(checks, 'artifact')


if __name__ == '__main__':
    unittest.main()
