# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the real native parser without importing torch or probing a GPU."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r16_argument_contract import validate_native_arguments


class NativeArgumentContractTests(unittest.TestCase):
    runner = ROOT / 'python/tilemega/serving/r16_run.py'

    def arguments(self, label):
        return ['--prefill', '/unused/prefill.so', '--decode', '/unused/decode.so',
                '--prompt-ids', '/unused/prompts.json', '--batch', '16',
                '--out', '/unused/results', '--system-label', label,
                '--mode', 'L2', '--decode-loop', '1', '--check-executors', '--hf-check']

    def test_existing_native_aliases_are_accepted(self):
        for label in ('TileMega', 'TM-H1H4', 'TM-noWGMMA'):
            result = validate_native_arguments(self.runner, self.arguments(label))
            self.assertEqual(result['parsed']['system_label'], label)
            self.assertEqual(result['parsed']['decode_loop'], 1)
            self.assertTrue(result['parsed']['check_executors'])

    def test_outer_candidate_labels_cannot_enter_native_parser(self):
        for label in ('TM-H3-stage', 'TM-H3-cluster4', 'TM-paged-mma'):
            with self.assertRaisesRegex(ValueError, 'invalid choice'):
                validate_native_arguments(self.runner, self.arguments(label))

    def test_actual_parser_rejects_missing_inputs_and_unknown_options(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            validate_native_arguments(self.runner, [])
        with self.assertRaisesRegex(ValueError, 'unrecognized arguments'):
            validate_native_arguments(self.runner, self.arguments('TileMega') + ['--invented'])

    def test_guard_and_runtime_prefixes_are_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / 'runner.py'
            runner.write_text('def main():\n'
                              '    parser = argparse.ArgumentParser()\n'
                              '    parser.add_argument("--value", required=True)\n'
                              '    args = parser.parse_args()\n'
                              '    raise RuntimeError("runtime must not execute")\n')
            self.assertEqual(validate_native_arguments(runner, ['--value', 'ok'])
                             ['parsed']['value'], 'ok')

    def test_unrecognized_prefix_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / 'runner.py'
            runner.write_text('def main():\n    probe_gpu()\n'
                              '    parser = argparse.ArgumentParser()\n'
                              '    args = parser.parse_args()\n')
            with self.assertRaisesRegex(ValueError, 'unsupported statements'):
                validate_native_arguments(runner, [])


if __name__ == '__main__':
    unittest.main()
