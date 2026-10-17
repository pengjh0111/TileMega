# SPDX-License-Identifier: BSD-3-Clause
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from collect_r16_final_dnn import choose, settled


def option(name, median, p90):
    return dict(proof=dict(candidate_id=name, checkpoint_sha256='checkpoint',
        archive_sha256='archive', input_sha256='inputs'), device=dict(uuid='gpu'),
        rows=[dict(median_ms=median, p90_ms=p90)]*3)


class FinalDnnPolicyTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
