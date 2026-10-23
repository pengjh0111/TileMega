import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from promote_sm90r2_calibration import check_metrics, KINDS


class PromotionContractTest(unittest.TestCase):
    def fixtures(self):
        candidates = [dict(instruction_sha256=str(i), binary_sha256=str(i),
            protocol_complete=True, processes=1, step_error_pct=0,
            correctness=dict(pass_=True)) for i in range(8)]
        ranking = dict(model='qwen3_0_6b', batch=1, ranking_pass=True,
            complete=True, retrospective=False, reused_training_receipts=[],
            prediction_alias_inconsistencies=[], distinct_programs=8,
            top3_contains_fastest=True, kendall_tau=.8, candidates=candidates)
        receipt = dict(model='qwen3_0_6b', batch=1, class_pass=True, binary_sha256='0')
        classes = [dict(kind=kind, error_pct='0') for kind in KINDS]
        return ranking, receipt, classes

    def test_complete_development_metrics(self):
        ranking, receipt, classes = self.fixtures()
        self.assertEqual(check_metrics(ranking, receipt, classes), ('qwen3_0_6b', 1))

    def test_ranking_cannot_promote_retrospective_or_training_results(self):
        for key, value in [('retrospective', True), ('reused_training_receipts', ['used']),
                           ('prediction_alias_inconsistencies', [['a', 'b']]),
                           ('top3_contains_fastest', False), ('kendall_tau', .6999),
                           ('kendall_tau', float('nan')), ('distinct_programs', 7)]:
            ranking, receipt, classes = self.fixtures();ranking[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_metrics(ranking, receipt, classes)

    def test_step_and_class_limits_remain_fixed(self):
        for error in [10.0001, -10.0001, float('nan'), float('inf')]:
            ranking, receipt, classes = self.fixtures()
            ranking['candidates'][0]['step_error_pct'] = error
            with self.subTest(step_error=error), self.assertRaises(ValueError):
                check_metrics(ranking, receipt, classes)
        for error in ['15.0001', '-15.0001', 'nan']:
            ranking, receipt, classes = self.fixtures();classes[0]['error_pct'] = error
            with self.subTest(class_error=error), self.assertRaises(ValueError):
                check_metrics(ranking, receipt, classes)

    def test_all_classes_must_name_a_ranked_program(self):
        for change in ['missing', 'duplicate', 'different_binary', 'wrong_cell']:
            ranking, receipt, classes = self.fixtures()
            if change == 'missing':classes.pop()
            if change == 'duplicate':classes[0] = copy.deepcopy(classes[1])
            if change == 'different_binary':receipt['binary_sha256'] = 'unranked'
            if change == 'wrong_cell':receipt['batch'] = 16
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_metrics(ranking, receipt, classes)

    def test_8b_requires_three_fresh_processes(self):
        ranking, receipt, classes = self.fixtures()
        ranking['model'] = receipt['model'] = 'llama3_1_8b'
        with self.assertRaises(ValueError):check_metrics(ranking, receipt, classes)
        for row in ranking['candidates']:row['processes'] = 3
        self.assertEqual(check_metrics(ranking, receipt, classes), ('llama3_1_8b', 1))


if __name__ == '__main__':
    unittest.main()
