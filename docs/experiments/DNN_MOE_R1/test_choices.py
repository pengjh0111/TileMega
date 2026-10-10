import copy
import unittest
from choose_dm import audit, distinguishably_faster, seal


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        self.winner = dict(identity_id='a' * 64, role='solver_candidate',
                           correctness_passed=True)
        self.search = dict(resnet18_B1=dict(source='solver_second_level',
                                          winner='a' * 64, candidates=[self.winner]))

    def test_faster_control_does_not_change_winner(self):
        selection = seal(self.search)
        original = copy.deepcopy(selection)
        matrix = dict(resnet18_B1={
            'TM': dict(identity_id='a' * 64, correctness_passed=True,
                       rounds=[0, 1, 2], latency_ms=[10., 10.01, 10.02]),
            'TM-L1': dict(identity_id='b' * 64, role='control',
                          correctness_passed=True, rounds=[0, 1, 2],
                          latency_ms=[8., 8.01, 8.02])})
        result = audit(selection, matrix)['cells']['resnet18_B1']
        self.assertTrue(result['faster_controls'][0]['analysis_required'])
        self.assertFalse(result['selection_changed'])
        self.assertEqual(selection, original)

    def test_control_or_failed_path_cannot_be_selected(self):
        for change in [dict(role='control'), dict(correctness_passed=False)]:
            changed = copy.deepcopy(self.search)
            changed['resnet18_B1']['candidates'][0].update(change)
            with self.assertRaises(ValueError):
                seal(changed)

    def test_incomplete_or_noisy_data_never_claims_a_speedup(self):
        with self.assertRaises(ValueError):
            distinguishably_faster([1.], [2.])
        self.assertFalse(distinguishably_faster([9., 10., 11.], [10., 11., 12.]))

    def test_identity_change_is_rejected(self):
        selection = seal(self.search)
        with self.assertRaises(ValueError):
            audit(selection, dict(resnet18_B1={'TM': dict(identity_id='b' * 64)}))


if __name__ == '__main__':
    unittest.main()
