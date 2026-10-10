import unittest

import torch

from tilemega.dnn.check_generated import output_metrics


class DnnOutputMetricsTest(unittest.TestCase):
    def test_fixture_zero_vectors_use_elementwise_contract(self):
        reference = torch.tensor([[0., 0.], [1., 2.]])
        actual = reference.clone()
        actual[1, 0] += .01
        metrics = output_metrics(actual, reference, reference, elementwise=True)
        self.assertLess(metrics['max_error'], .016)
        # Model-level criteria are still independent and unchanged.
        with self.assertRaises(AssertionError):
            output_metrics(actual, reference, reference)
        actual[0, 0] = .1
        with self.assertRaisesRegex(AssertionError, 'elementwise error'):
            output_metrics(actual, reference, reference, elementwise=True)


if __name__ == '__main__':
    unittest.main()
