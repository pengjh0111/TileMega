"""Reference metric variation must never hide changed or inaccurate outputs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from dnn4_bert_model_cohort import validate_bert_checks, validate_bert_output_identity


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
    def test_two_distinct_outputs_match_reference(self):
        checks = cohort()
        reference = copy.deepcopy(checks[0])
        reference['artifact_id'] = 'incumbent'
        reference['cases'][0]['outputs'][0]['max_error'] += .000001
        result = validate_bert_output_identity(checks, reference, 'artifact')
        self.assertEqual(result['named_output_sha256'], {
            'layer_norm_24': 'a'*64, 'tanh': 'b'*64})

    def test_identity_preserves_output_names_cases_and_inputs(self):
        changes = [
            lambda r: r.update(passed=False),
            lambda r: r.update(input_tensors_sha256='other'),
            lambda r: r.update(source_checkpoint_sha256='other'),
            lambda r: r.update(checker_sha256='other'),
            lambda r: r['cases'].pop(),
            lambda r: r['cases'][0]['outputs'].reverse(),
            lambda r: r['cases'][0]['outputs'][0].update(output_sha256='b'*64),
            lambda r: r['cases'][0]['outputs'][1].update(output_sha256='a'*64),
            lambda r: r['cases'][5]['outputs'][1].update(output_sha256='c'*64),
        ]
        for change in changes:
            with self.subTest(change=change):
                checks = cohort()
                reference = copy.deepcopy(checks[0])
                change(reference)
                with self.assertRaises(ValueError):
                    validate_bert_output_identity(checks, reference, 'artifact')

    def test_identity_retains_five_process_precision_and_determinism(self):
        for change in (lambda c: c.pop(),
                       lambda c: c[4]['cases'][0]['outputs'][1].update(
                           output_sha256='c'*64),
                       lambda c: c[4]['cases'][0]['outputs'][1].update(
                           cosine_min=.99899)):
            checks = cohort()
            reference = copy.deepcopy(checks[0])
            change(checks)
            with self.assertRaises(ValueError):
                validate_bert_output_identity(checks, reference, 'artifact')

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
