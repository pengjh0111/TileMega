import unittest
from tilemega.moe.routing_check import agreement, index_names


class Routing(unittest.TestCase):
    def test_order_and_set_agreement_are_distinct(self):
        reference = [[[[1, 2], [2, 3]], [[0, 1], [0, 2]]]]
        actual = [[[[2, 1], [2, 3]], [[0, 3], [0, 2]]]]
        result = agreement(actual, reference)
        self.assertEqual(result['positions'], 4)
        self.assertEqual(result['exact_set_ratio'], .75)
        self.assertEqual(result['exact_rank_ratio'], .5)
        self.assertEqual(result['per_layer_set_ratio'], [1., .5])

    def test_rejects_truncated_or_misbound_tables(self):
        for a, b in [([], []), ([[1]], [[1]]), ([[[[1]]]], [[[[1, 2]]]])]:
            with self.assertRaises(ValueError):
                agreement(a, b)
        with self.assertRaises(ValueError):
            index_names({'nodes': []}, 1)

    def test_indices_follow_decoder_order(self):
        bridge = {'nodes': [{'target': 'tilemega.moe_experts.default',
                            'inputs': ['h', name, 'w']} for name in ['i1', 'i2']]}
        self.assertEqual(index_names(bridge, 2), ['i1', 'i2'])

if __name__ == '__main__':
    unittest.main()
