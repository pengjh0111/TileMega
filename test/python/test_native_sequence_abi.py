"""Sequence plans dispatch graph replay with one shared iteration counter."""
from types import SimpleNamespace
import unittest
from tilemega.serving.plan import Plan


class SequenceAbiTests(unittest.TestCase):
    def plan(self, execution):
        self.calls = []
        def call(name):
            def launch(*arguments):
                self.calls.append((name, arguments))
                return 0
            return launch
        api = SimpleNamespace(tm_plan_launch=call('single'), tm_plan_launch_steps=call('steps'),
                              tm_plan_launch_sequence=call('sequence'), tm_plan_destroy=lambda _: None)
        library = SimpleNamespace(lib=api, execution=execution, info=SimpleNamespace(phase=1, modes=1))
        return Plan(library, 42)

    def test_sequence_single_and_batch_preserve_iteration(self):
        plan = self.plan('kernel_sequence')
        plan.launch_steps(0, 64, 1, 99)
        plan.launch(64, 1, 99)
        plan.launch_steps(65, 1022, 1, 99)
        self.assertEqual(self.calls, [('sequence', (42, 0, 64, 1, 0, 99)),
                                     ('sequence', (42, 64, 1, 1, 64, 99)),
                                     ('sequence', (42, 65, 1022, 1, 65, 99))])
        self.assertEqual(plan.iteration[1], 1087)

    def test_megakernel_abi_is_unchanged(self):
        plan = self.plan('megakernel')
        plan.launch(0, 1, 99); plan.launch_steps(1, 7, 1, 99)
        self.assertEqual(self.calls, [('single', (42, 0, 1, 0, 99)),
                                     ('steps', (42, 1, 7, 1, 1, 99))])

    def test_failure_does_not_advance_host_iteration(self):
        plan = self.plan('kernel_sequence')
        plan.library.lib.tm_plan_launch_sequence = lambda *args: -1
        with self.assertRaises(RuntimeError):
            plan.launch_steps(0, 4, 1, 99)
        self.assertEqual(plan.iteration[1], 0)


if __name__ == '__main__':
    unittest.main()
