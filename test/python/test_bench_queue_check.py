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

    def test_recovered_numeric_gate_cannot_read_the_old_job(self):
        gate = step('gate_v2', ['gpu_v2'])
        gate['command'] = ['python3', '/frozen/validate_r16_numeric_gate.py', '--job', 'gpu_v1']
        result = check([step('gpu_v1'), step('gpu_v2')], [gate], {})
        self.assertFalse(result['pass_check'])
        self.assertEqual(result['errors'][0]['target'], 'gpu_v1')
        gate['command'][-1] = 'gpu_v2'
        self.assertTrue(check([step('gpu_v2')], [gate], {})['pass_check'])

    def test_dnn_numeric_gate_requires_its_declared_gpu_dependency(self):
        gate = step('dnn_gate', after_any=['gpu'])
        gate['command'] = ['python3', 'validate_r16_dnn_numeric_gate.py', '--job', 'gpu']
        self.assertFalse(check([step('gpu')], [gate], {})['pass_check'])
        gate['after'] = ['gpu']
        self.assertTrue(check([step('gpu')], [gate], {})['pass_check'])

    def test_shared_checkpoint_gate_cannot_point_at_an_unrelated_old_receipt(self):
        gate = step('shared_gate', ['gpu_v2'])
        gate['command'] = ['python3', '/frozen/validate_shared_numeric.py', '--job', 'gpu_v1']
        self.assertFalse(check([step('gpu_v1'), step('gpu_v2')], [gate], {})['pass_check'])
        gate['command'][-1] = 'gpu_v2'
        self.assertTrue(check([step('gpu_v2')], [gate], {})['pass_check'])

    def test_pci_domain_mismatch_is_rejected_before_a_gpu_job(self):
        gpu = step('gpu')
        gpu['command'] = ['python3', '/frozen/run_r16_with_device_identity.py',
                          '--expected-pci', '00000000:41:00.0']
        self.assertFalse(check([], [gpu], {})['pass_check'])
        gpu['command'][-1] = '0000:41:00.0'
        self.assertTrue(check([], [gpu], {})['pass_check'])


if __name__ == '__main__':
    unittest.main()
