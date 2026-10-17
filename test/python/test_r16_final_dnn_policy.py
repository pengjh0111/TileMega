# SPDX-License-Identifier: BSD-3-Clause
import sys
import copy
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from collect_r16_final_dnn import PAIRS, baseline_policy, choose, settled


def option(name, median, p90):
    return dict(proof=dict(candidate_id=name, checkpoint_sha256='checkpoint',
        archive_sha256='archive', input_sha256='inputs'), device=dict(uuid='gpu'),
        rows=[dict(median_ms=median, p90_ms=p90)]*3)


class FinalDnnPolicyTest(unittest.TestCase):
    def baseline_fixture(self):
        cells = [dict(model=model, selected=dict(archive_sha256='archive',
            checkpoint_sha256='checkpoint'), physical_device=dict(uuid='gpu',boot_id='boot'))
            for model in PAIRS]
        rows = [dict(model=model, system=system, precision=precision, compile_mode=mode,
            protocol_complete=True, fresh_processes=3, exported_program_sha256='archive',
            source_checkpoint_sha256='checkpoint', physical_gpu_uuid='gpu',hardware_boot_id='boot',
            median_ms=10, p90_ms=12) for model in PAIRS
            for system,precision,mode in [('tensorrt','bf16',''), ('tensorrt','fp16',''),
                ('torchcompile','bf16','max-autotune'),('torchcompile','bf16','reduce-overhead')]]
        return rows, cells

    def test_pending_is_not_a_completed_selection(self):
        self.assertFalse(settled(['job'], {}))
        self.assertFalse(settled(['job'], {'job':dict(status='running')}))
        self.assertTrue(settled(['job'], {'job':dict(status='failed')}))

    def test_p90_regression_keeps_the_incumbent(self):
        incumbent = option('old', 10, 12)
        winner, decisions = choose([incumbent, option('new', 8, 15)])
        self.assertIs(winner, incumbent)
        self.assertFalse(decisions[0]['eligible_in_measured_grid'])

    def test_clear_improvement_can_replace_the_incumbent(self):
        improved = option('new', 8, 10)
        winner, _ = choose([option('old', 10, 12), improved])
        self.assertIs(winner, improved)

    def test_indistinguishable_keeps_the_incumbent(self):
        incumbent = option('old', 10, 12)
        winner, _ = choose([incumbent, option('new', 9.99, 11.99)])
        self.assertIs(winner, incumbent)

    def test_input_and_device_changes_cannot_select_a_faster_arm(self):
        for field in ('checkpoint_sha256', 'archive_sha256', 'input_sha256', 'device'):
            changed = option('new', 8, 10)
            if field == 'device':
                changed['device'] = dict(uuid='other')
            else:
                changed['proof'][field] = 'other'
            with self.subTest(field=field), self.assertRaises(ValueError):
                choose([option('old', 10, 12), changed])

    def test_baseline_attribution_preference_is_independent_of_speed(self):
        rows, cells = self.baseline_fixture()
        historical = dict(rows[0], physical_gpu_uuid='',hardware_boot_id='',median_ms=1)
        selected, excluded = baseline_policy(rows+[historical], cells)
        self.assertEqual(len(selected), 20)
        self.assertTrue(all(row['median_ms'] == 10 for row in selected))
        self.assertEqual(len(excluded), 1)

    def test_ambiguous_baseline_triplets_are_rejected(self):
        rows, cells = self.baseline_fixture()
        with self.assertRaises(ValueError):
            baseline_policy(rows+[copy.deepcopy(rows[0])], cells)

    def test_export_identity_is_required_even_without_historical_checkpoint_hash(self):
        rows, cells = self.baseline_fixture()
        rows[0]['source_checkpoint_sha256'] = ''
        self.assertEqual(len(baseline_policy(rows, cells)[0]), 20)
        rows[0]['exported_program_sha256'] = 'other'
        with self.assertRaises(ValueError):
            baseline_policy(rows, cells)


if __name__ == '__main__':
    unittest.main()
