import unittest

from prepare_regression import handoff_option


class RecipeTest(unittest.TestCase):
    def test_selected_implementation_uses_selection_policy(self):
        self.assertEqual(handoff_option('last_arriver'), 'auto')
        self.assertEqual(handoff_option('off'), 'off')
        self.assertEqual(handoff_option('auto'), 'auto')

    def test_unknown_handoff_cannot_silently_change_semantics(self):
        with self.assertRaises(ValueError):
            handoff_option('recompute')


if __name__ == '__main__':
    unittest.main()
