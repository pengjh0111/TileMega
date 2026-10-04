import json
import unittest
from unittest.mock import patch

import advance


class FixedPlanTests(unittest.TestCase):
    def jobs(self):
        sources = {m: {p: {'bridge': '/fixture/' + m + '/' + p}
                         for p in ('prefill', 'decode')} for m in ('llama', 'qwen3')}
        with patch.object(advance.Path, 'read_text', return_value=json.dumps(sources)):
            return advance.make_jobs()

    def test_fixed_geometry_domain_is_bounded(self):
        rows = self.jobs()
        self.assertEqual(len(rows), 31)
        self.assertEqual(sum(r['phase'] == 'prefill' for r in rows), 4)
        self.assertEqual(sum(r['label'] == 'N-R12bh-120' for r in rows), 3)
        self.assertFalse(any(r['cell'] == 'qwen3_B16' and r['label'] == 'N-R12bh-120' for r in rows))

    def test_pdl_is_only_on_registered_pdl_arms(self):
        for row in self.jobs():
            expected = 'auto' if row['label'].endswith(('-pdl0', '-pdl')) else 'off'
            self.assertEqual(row['overrides']['pdl'], expected)
            self.assertEqual(row['overrides']['watchdog'], 0)

    def test_seed_last_arriver_is_an_explicit_control(self):
        rows = self.jobs()
        for row in rows:
            if row['label'] in ('PS-120', 'PSA-120'):
                self.assertEqual(row['overrides']['paged_la_splitk'], int(row['label'] == 'PSA-120'))
                self.assertEqual(row['overrides']['weight_layout'], 'tiled')

    def test_prefill_uses_the_portable_r13_decision(self):
        for row in self.jobs():
            if row['phase'] == 'prefill':
                self.assertIn('/PF-R10-noWD/', row['manifest'])
                self.assertEqual(row['projection'], 'identity')


if __name__ == '__main__':
    unittest.main()
