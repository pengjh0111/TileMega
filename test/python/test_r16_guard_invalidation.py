# SPDX-License-Identifier: BSD-3-Clause
"""A later guard audit must exclude an otherwise successful original receipt."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from r16_measurement_validation import completed_guard


class GuardInvalidationTest(unittest.TestCase):
    def test_successful_original_is_preserved_but_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw = json.dumps(dict(code=0, acceptance='strict'))
            (root/'guard_result.json').write_text(raw)
            event = dict(status='done', exit_code=0)
            self.assertEqual(completed_guard(root, event)['code'], 0)
            (root/'guard_invalidation.json').write_text(json.dumps(dict(
                valid=False, reason='required physical GPU lock was absent')))
            with self.assertRaisesRegex(ValueError, 'physical GPU lock'):
                completed_guard(root, event)
            self.assertEqual((root/'guard_result.json').read_text(), raw)

    def test_malformed_invalidation_never_accepts_original(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'guard_result.json').write_text('{"code":0}')
            (root/'guard_invalidation.json').write_text('not JSON')
            with self.assertRaises(ValueError):
                completed_guard(root, dict(status='done', exit_code=0))


if __name__ == '__main__':
    unittest.main()
