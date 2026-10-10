import unittest

from prepare_regression import handoff_option


class RecipeTest(unittest.TestCase):
    def test_selected_implementation_uses_selection_policy(self):
        self.assertEqual(handoff_option('last_arriver', 'pages'), 'off')
        self.assertEqual(handoff_option('off', 'l2'), 'off')
        self.assertEqual(handoff_option('off', 'pages'), 'off')
        with self.assertRaises(ValueError):
            handoff_option('last_arriver', 'l2')
        with self.assertRaises(ValueError):
            handoff_option('auto', 'pages')

    def test_unknown_handoff_cannot_silently_change_semantics(self):
        with self.assertRaises(ValueError):
            handoff_option('recompute', 'pages')


if __name__ == '__main__':
    unittest.main()
