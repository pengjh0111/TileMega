"""A zero process exit must not certify a failed or mismatched DNN gate."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from validate_r16_dnn_numeric_gate import validate_payload


class NumericGate(unittest.TestCase):
    def setUp(self):
        self.config = dict(export_files={'exported_program.pt2': 'archive',
                                        'inputs_b1.safetensors': 'inputs'})
        self.gate = dict(passed=True, evidence='verified', artifact_id='binary',
            checker_sha256='checker', source_checkpoint_sha256='checkpoint',
            bridge_sha256='bridge', archive_sha256='archive', input_tensors_sha256='inputs',
            batch=1, cases=[dict(mode=1), dict(mode=2)])

    def validate(self, gate):
        return validate_payload(gate, self.config, 'binary', 'checker', 'checkpoint', 'bridge')

    def test_complete_true_gate_passes(self):
        self.assertTrue(self.validate(self.gate)['passed'])

    def test_false_or_nonboolean_pass_cannot_certify_timing(self):
        for value in (False, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(dict(self.gate, passed=value))

    def test_binary_checkpoint_and_inputs_cannot_be_substituted(self):
        for key in ('artifact_id', 'archive_sha256', 'input_tensors_sha256',
                    'source_checkpoint_sha256', 'bridge_sha256', 'checker_sha256', 'batch'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(dict(self.gate, **{key: 'other'}))

    def test_both_executors_are_required(self):
        gate = copy.deepcopy(self.gate)
        gate['cases'] = [dict(mode=1)]
        with self.assertRaises(ValueError):
            self.validate(gate)


if __name__ == '__main__':
    unittest.main()
