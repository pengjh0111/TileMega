#!/usr/bin/env python3
"""User-authorized performance continuation must not conceal failed gates."""
import unittest
from stop_loss import outcome

class StopLossTests(unittest.TestCase):
    def test_regression_remains_failed_when_collection_is_allowed(self):
        rows=[dict(kill=False) for _ in range(3)]+[dict(kill=True)]
        self.assertFalse(outcome(rows)['collection_allowed'])
        report=outcome(rows,advisory=True)
        self.assertTrue(report['collection_allowed'])
        self.assertFalse(report['pass_'])
        self.assertEqual(report['threshold_rel'],.01)

    def test_advisory_does_not_allow_missing_clean_rounds(self):
        rows=[dict(kill=False) for _ in range(3)]+[dict(status='missing three clean rounds')]
        self.assertFalse(outcome(rows,advisory=True)['collection_allowed'])

if __name__=='__main__':unittest.main()
