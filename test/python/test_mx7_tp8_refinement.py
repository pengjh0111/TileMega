"""Measured depth refinement preserves layout and global solver provenance."""
import copy
import importlib.util
from pathlib import Path
import unittest

from tilemega.tp.selection_tp import k_best, pin, stage_lists
from test_mx7_selection import fixture

source = Path(__file__).resolve().parents[2] / 'experiments/mx7/tp8_depth_refinement.py'
spec = importlib.util.spec_from_file_location('mx7_depth_refinement', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RefinementTests(unittest.TestCase):
    def domain(self):
        rows, constants = fixture()
        for row in rows:
            row['key'].update(model='qwen3_1_7b_tp8', tp=8)
        for group in constants.values():
            for row in group:
                row['key']['tp'] = 8
        for row in list(rows):
            if row['key']['stage'] == 'attention':
                continue
            for depth in (2, 3):
                next_row = copy.deepcopy(row)
                next_row['key'].update(d=depth, prefetch=depth)
                next_row['median_ns'] += depth
                rows.append(next_row)
        catalog = dict(rows=rows, tp_constants=constants)
        lists = stage_lists(rows, 'qwen3_1_7b_tp8', 1, constants)
        config = pin(next(k_best(lists, 28)), 'a' * 64, 8)['mx_decode_v2']
        return catalog, lists, config

    def test_every_refinement_retains_true_global_candidate_rank(self):
        catalog, lists, config = self.domain()
        expected = {choice['candidate_rank']: choice for choice in k_best(lists, 28)}
        for item in module.refinements(catalog, 1, config):
            choice = item['choice']
            self.assertEqual(choice, expected[choice['candidate_rank']])
            self.assertTrue(module.family_matches(choice, config))
            for stage in module.REGULAR:
                key = choice['stages'][stage]['key']
                self.assertEqual((key['tn'], key['wn']), (config[stage]['tn'], config[stage]['wn']))

    def test_uniform_depths_are_measured_combinations(self):
        catalog, _, config = self.domain()
        choices = {item['name']: item['choice'] for item in module.refinements(catalog, 1, config)}
        for depth in (2, 3):
            choice = choices[f'uniform_d{depth}']
            self.assertEqual({choice['stages'][stage]['key']['d'] for stage in module.REGULAR}, {depth})

    def test_a_different_attention_partition_is_excluded(self):
        _, lists, config = self.domain()
        choice = next(k_best(lists, 28))
        changed = copy.deepcopy(choice)
        changed['stages']['attention']['key']['b_min'] = 8
        self.assertFalse(module.family_matches(changed, config))


if __name__ == '__main__':
    unittest.main()
