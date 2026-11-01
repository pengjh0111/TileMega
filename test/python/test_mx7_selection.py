"""Independent small-domain oracles for TP communication ranking and k-best."""
import copy
import itertools
import unittest

from tilemega.tp.selection_tp import STAGES, k_best, pin, select_measured, stage_lists


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
