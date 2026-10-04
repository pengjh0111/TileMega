import json
from pathlib import Path
import unittest

import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_preregistered_vllm_is_the_installed_version(self):
        data = json.loads((bootstrap.HERE / 'predictions_sm120.json').read_text())
        self.assertEqual(data['vllm_version'], '0.29.0')
        self.assertFalse(data['sm89_phase_d_complete_at_start'])
        self.assertEqual(len(data['predictions']), 14)

    def test_guard_changes_only_local_idle_power(self):
        original = json.loads((bootstrap.FRAME / 'guard_policy.json').read_text())
        local = json.loads((bootstrap.HERE / 'guard_policy.json').read_text())
        original.pop('idle_power_w')
        local.pop('idle_power_w')
        self.assertEqual(original, local)

    def test_queue_orders_calibration_before_first_execution(self):
        rows = json.loads((bootstrap.HERE / 'queue_e0_e2a.json').read_text())
        rows = {row['name']: row for row in rows}
        self.assertEqual(rows['E1_TL2']['after'], ['E1_llama', 'E1_qwen3'])
        for suite in 'bcdef':
            row = rows['E2a_MB-1' + suite]
            self.assertEqual(row['after'], ['E1_TL2'])
            self.assertTrue(row['gpu'])
        self.assertIn('E0_test_build', rows['E0_units']['after_any'])
        self.assertNotIn('E0_units', rows['E1_llama']['after'])
        self.assertIn('E0_environment', rows['E1_llama']['after'])

    def test_sources_are_not_legacy_worktree(self):
        self.assertEqual(bootstrap.ROOT, Path('/root/tilemega-r13-sm120'))
        self.assertNotEqual(bootstrap.BUILD, Path('/root/tilemega-sm120-build'))
        data = json.loads((bootstrap.HERE / 'start.json').read_text())
        self.assertTrue(data['checkpoint_not_final'])
        self.assertTrue(data['no_push'])


if __name__ == '__main__':
    unittest.main()
