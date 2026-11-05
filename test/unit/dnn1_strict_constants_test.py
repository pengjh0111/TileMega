"""CPU-only contracts for sampled probe-process ownership evidence."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('strict_constants', ROOT/'scripts/dnn1_strict_constants.py')
strict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(strict)


class StrictWindows(unittest.TestCase):
    def test_empty_samples_around_one_stable_owner_are_allowed(self):
        strict.validate_window(dict(errors=[], samples=[dict(owners={}),
            dict(owners={'321': {}}), dict(owners={'321': {}}), dict(owners={})]))

    def test_unobserved_overlapping_changed_or_failed_samples_are_rejected(self):
        for rows, errors in (([], []), ([{}], []),
                ([{'321': {}, '654': {}}], []), ([{'321': {}}, {'654': {}}], []),
                ([{'321': {}}], ['telemetry failure'])):
            with self.subTest(rows=rows, errors=errors):
                with self.assertRaisesRegex(RuntimeError, 'stable observed owner'):
                    strict.validate_window(dict(errors=errors, samples=[dict(owners=r) for r in rows]))


if __name__ == '__main__':
    unittest.main()
