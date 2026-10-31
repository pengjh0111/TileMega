"""Group claims (R5 B0): a multi-card unit takes NUMA-local cards together."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import sm90r5_dispatch as dispatch


class GroupSelectionTests(unittest.TestCase):
    def test_tp4_prefers_one_numa_node(self):
        policy = {'allowed_timing_gpu_indices': [0, 1, 2, 3, 4, 5, 7]}
        # GPU 6 is excluded, so node 1 cannot supply four cards.
        self.assertEqual(dispatch.groups(policy, 4), [[0, 1, 2, 3]])
        self.assertEqual(dispatch.groups(policy, 4)[0], [0, 1, 2, 3])

    def test_tp2_groups_stay_inside_a_node(self):
        policy = {'allowed_timing_gpu_indices': list(range(8))}
        for group in dispatch.groups(policy, 2):
            self.assertEqual(len({dispatch.numa(gpu) for gpu in group}), 1, group)
        self.assertEqual(dispatch.groups(policy, 2)[0], [0, 1])

    def test_a_group_spans_nodes_only_when_no_node_can_hold_it(self):
        policy = {'allowed_timing_gpu_indices': [0, 1, 4, 5]}
        self.assertEqual(dispatch.groups(policy, 4), [[0, 1, 4, 5]])
        # Two cards still fit one node, so no cross-node pair is offered.
        for group in dispatch.groups(policy, 2):
            self.assertEqual(len({dispatch.numa(gpu) for gpu in group}), 1, group)

    def test_excluded_cards_never_appear(self):
        policy = {'allowed_timing_gpu_indices': list(range(8))}
        for group in dispatch.groups(policy, 2, exclude=(0, 1, 2)):
            self.assertFalse({0, 1, 2} & set(group), group)


if __name__ == '__main__':
    unittest.main()
