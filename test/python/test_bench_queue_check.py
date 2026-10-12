import unittest

from tilemega.bench.queue_check import check


def step(name, after=(), after_any=()):
    return dict(name=name, after=list(after), after_any=list(after_any))


class QueueCheckTest(unittest.TestCase):
    def test_failed_old_run_is_not_a_valid_gate_for_a_recovery(self):
        existing = [step('old_build'), step('old_run', ['old_build']),
                    step('fixed_build'), step('fixed_run', ['fixed_build'])]
        state = dict(old_build=dict(status='failed'), old_run=dict(status='skipped'),
                     fixed_build=dict(status='done'), fixed_run=dict(status='pending'))
        bad = check(existing, [step('model', ['old_run'])], state)
        self.assertFalse(bad['pass_check'])
        self.assertEqual(bad['errors'][0]['path'], ['model', 'old_run'])
        good = check(existing, [step('model', ['fixed_run'])], state)
        self.assertTrue(good['pass_check'])
        self.assertEqual(state['old_run']['status'], 'skipped')

    def test_pending_ancestor_failure_is_detected(self):
        result = check([step('compile'), step('numeric', ['compile'])],
                       [step('model', ['numeric'])], dict(compile=dict(status='failed')))
        self.assertEqual(result['errors'][0]['path'], ['model', 'numeric', 'compile'])

    def test_after_any_failure_allows_diagnostics(self):
        result = check([step('numeric')], [step('diagnostic', after_any=['numeric'])],
                       dict(numeric=dict(status='failed')))
        self.assertTrue(result['pass_check'])

    def test_unknown_dependencies_and_cycles_are_rejected(self):
        self.assertFalse(check([], [step('a', ['missing'])], {})['pass_check'])
        result = check([], [step('a', ['b']), step('b', after_any=['a'])], {})
        self.assertTrue(any(e['reason'] == 'dependency cycle' for e in result['errors']))

    def test_duplicate_names_and_receipt_rewrites_are_rejected(self):
        self.assertFalse(check([step('a')], [step('a')], {})['pass_check'])
        result = check([], [step('a')], dict(a=dict(status='skipped')))
        self.assertIn('receipt already exists', result['errors'][0]['reason'])

    def test_completed_receipt_has_no_remaining_failed_ancestors(self):
        result = check([step('a', ['old'])], [step('b', ['a'])],
                       dict(a=dict(status='done'), old=dict(status='failed')))
        self.assertTrue(result['pass_check'])


if __name__ == '__main__':
    unittest.main()
