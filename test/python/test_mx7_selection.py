"""Independent small-domain oracles for TP communication ranking and k-best."""
import copy
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tilemega.tp.selection_tp import STAGES, k_best, pin, select_measured, stage_lists
from tilemega.tp.solve_tp import check_constant_group, check_qualification


def row(stage, tn=16, cost=10, length=None):
    key = dict(model='qwen3_1_7b_tp2', b=1, tp=2, stage=stage,
               tn=tn, wn=1, d=1, prefetch=1, r_next=100, part=1)
    if stage == 'attention':
        key.update(attn_version=2, prefetch=0, b_min=1, length=length)
    return dict(key=key, median_ns=cost, p90_ns=cost * 1.5, processes=3,
                resources=dict(resident=1, MTreg=100, stack=0, static_smem=0))


def fixture():
    rows = [row(name) for name in STAGES if name != 'attention']
    rows += [row('attention', length=n) for n in (65, 256, 512, 768, 1088)]
    constants = dict(t_ar=[dict(key=dict(tp=2, b=1, tn=16, stage=stage), median_ns=5,
                               p90_ns=7, processes=3) for stage in ('o', 'down')],
                     t_head_x=[dict(key=dict(tp=2, b=1), median_ns=5, p90_ns=7, processes=3)])
    return rows, constants


class SelectionTests(unittest.TestCase):
    def test_tp8_v3_and_v2_have_distinct_complete_length_grids(self):
        rows, constants = fixture()
        for item in rows:
            item['key'].update(model='qwen3_1_7b_tp8', tp=8)
        for group in constants.values():
            for item in group:
                item['key']['tp'] = 8
        for item in list(rows):
            if item['key']['stage'] == 'attention':
                added = copy.deepcopy(item)
                added['key'].pop('b_min')
                added['key'].update(attn_version=3, split=4)
                added['median_ns'] = 5
                rows.append(added)
        choices = stage_lists(rows, 'qwen3_1_7b_tp8', 1, constants)
        self.assertEqual(len(choices['attention']), 2)
        selected = pin(next(k_best(choices, 28)), 'a' * 64, 8)
        self.assertEqual(selected['mx_decode_v2']['attn_version'], 3)
        self.assertEqual(selected['mx_decode_v2']['split'], 4)
        self.assertEqual(selected['mx_decode_v2']['b_min'], 1)

    def test_v3_is_still_rejected_for_tp2_and_tp4(self):
        for tp in (2, 4):
            rows, constants = fixture()
            for item in rows:
                item['key'].update(model=f'qwen3_1_7b_tp{tp}', tp=tp)
                if item['key']['stage'] == 'attention':
                    item['key'].update(attn_version=3, split=4)
            for group in constants.values():
                for item in group:
                    item['key']['tp'] = tp
            with self.assertRaisesRegex(ValueError, 'requires TP8'):
                stage_lists(rows, f'qwen3_1_7b_tp{tp}', 1, constants)

    def test_constants_follow_their_measured_physical_group(self):
        catalog = {'tp_gpu_groups': {'4': [1, 2, 3, 5]}}
        check_constant_group(catalog, 4, [1, 2, 3, 5])
        with self.assertRaisesRegex(ValueError, 'different physical GPU group'):
            check_constant_group(catalog, 4, [1, 3, 4, 5])
        check_constant_group({}, 4, [1, 3, 4, 5])

    def test_selection_refuses_changed_or_ineligible_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'costs.json'
            catalog = dict(kind='mx7_tp_measured_stage_costs', complete_scope=True,
                           tp_constants={'t_ar': [], 't_head_x': []})
            path.write_text(json.dumps(catalog))
            proof = dict(cost_table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         solver_eligible=True)
            qualification = path.with_suffix('.qualification.json')
            qualification.write_text(json.dumps(proof))
            self.assertEqual(check_qualification(path)[0], catalog)
            proof['solver_eligible'] = False
            qualification.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, 'eligible TP costs'):
                check_qualification(path)
            proof['solver_eligible'] = True
            qualification.write_text(json.dumps(proof))
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'eligible TP costs'):
                check_qualification(path)

    def test_communication_is_added_before_sorting(self):
        rows, constants = fixture()
        rows += [row('o', tn=32, cost=12)]
        constants['t_ar'][0]['median_ns'] = 100
        constants['t_ar'].append(dict(key=dict(tp=2, b=1, tn=32, stage='o'),
                                     median_ns=2, p90_ns=3, processes=3))
        choices = stage_lists(rows, 'qwen3_1_7b_tp2', 1, constants)
        self.assertEqual(choices['o'][0]['key']['tn'], 32)
        self.assertEqual(choices['o'][0]['ns'], 14)
        self.assertEqual(choices['head'][0]['ns'], 15)

    def test_k_best_matches_exhaustive_cartesian_scores(self):
        lists = {name: [dict(ns=(i + 1) * (j + 1), key={}) for j in range(2)]
                 for i, name in enumerate(STAGES)}
        factors = (28, 28, 28, 28, 28, 1)
        expected = sorted(sum(lists[name][rank]['ns'] * factors[i]
                              for i, (name, rank) in enumerate(zip(STAGES, ranks)))
                          for ranks in itertools.product(range(2), repeat=6))
        actual = [value['predicted_ns'] for value in k_best(lists, 28)]
        self.assertEqual(actual, expected)

    def test_noisy_p90_and_complete_attention_grid(self):
        rows, constants = fixture()
        rows[0]['noisy'] = True
        lists = stage_lists(rows, 'qwen3_1_7b_tp2', 1, constants)
        self.assertEqual(lists['qkv'][0]['ns'], 15)
        with self.assertRaisesRegex(ValueError, 'incomplete attention'):
            stage_lists(rows[:-1], 'qwen3_1_7b_tp2', 1, constants)
        constants['t_ar'] = []
        with self.assertRaisesRegex(ValueError, 'communication constant'):
            stage_lists(rows, 'qwen3_1_7b_tp2', 1, constants)

    def test_dense_pin_places_selection_evidence_outside_config(self):
        rows, constants = fixture()
        choice = next(k_best(stage_lists(rows, 'qwen3_1_7b_tp2', 1, constants), 28))
        value = pin(choice, 'a' * 64, 2, 'manual')
        self.assertEqual(value['evidence']['tp'], 2)
        self.assertEqual(value['evidence']['build_past_range'], '0:1086')
        self.assertEqual(value['mx_decode_v2']['purpose'], 'manual')
        self.assertNotIn('selection', value['mx_decode_v2'])
        self.assertFalse(value['mx_decode_v2']['default_enabled'])

    def test_exact_one_percent_rule(self):
        self.assertEqual(select_measured([100, 101, 102, 103],
                                        [[100, 100], [99, 99], [100, 100], [100, 100]])['selected'], 1)
        self.assertEqual(select_measured([100, 101, 102, 103],
                                        [[100, 100], [99.1, 99.1], [100, 100], [100, 100]])['selected'], 0)


if __name__ == '__main__':
    unittest.main()
