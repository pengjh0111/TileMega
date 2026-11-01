"""CPU checks of optional TP ABI pointer arrays and iteration bookkeeping."""
import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tilemega.serving.plan import Plan, PlanLibrary


class PlanTests(unittest.TestCase):
    def library(self, world=2):
        lib = PlanLibrary.__new__(PlanLibrary)
        lib.lib = SimpleNamespace(tm_plan_tp_world=lambda: world)
        lib.info = SimpleNamespace(capacity=1088, modes=1)
        return lib

    def test_tp1_library_without_optional_symbols(self):
        lib = PlanLibrary.__new__(PlanLibrary)
        lib.lib = SimpleNamespace()
        self.assertEqual(lib.tp_world(), 1)

    def test_buffers_and_rank_ordered_pointer_arrays(self):
        lib = self.library()
        def buffers(handle, ar, exchange):
            C.cast(ar, C.POINTER(C.c_void_p)).contents.value = 0x1000
            C.cast(exchange, C.POINTER(C.c_void_p)).contents.value = 0x2000
            return 0
        lib.lib.tm_plan_tp_buffers = buffers
        lib.lib.tm_plan_bind_tp = Mock(return_value=0)
        plan = Plan(lib, 42)
        self.assertEqual(plan.tp_buffers(), (0x1000, 0x2000))
        plan.bind_tp(1, [0x1000, 0x3000], [0x2000, 0x4000])
        call = lib.lib.tm_plan_bind_tp.call_args.args
        self.assertEqual(call[:2], (42, 1))
        self.assertEqual(list(call[2]), [0x1000, 0x3000])
        self.assertEqual(list(call[3]), [0x2000, 0x4000])
        with self.assertRaises(ValueError):
            plan.bind_tp(2, [0x1000, 0x3000], [0x2000, 0x4000])

    def test_prompt_length_and_failed_launch_do_not_advance_counter(self):
        lib = self.library()
        lib.lib.tm_plan_set_prompt_len = Mock(return_value=0)
        lib.lib.tm_plan_launch_steps = Mock(return_value=-4)
        plan = Plan(lib, 42)
        plan.set_prompt_len(64)
        lib.lib.tm_plan_set_prompt_len.assert_called_once_with(42, 64)
        with self.assertRaisesRegex(RuntimeError, '-4'):
            plan.launch_steps(0, 64, 1, 7)
        self.assertEqual(plan.iteration[1], 0)
        lib.lib.tm_plan_launch_steps.return_value = 0
        plan.launch_steps(0, 64, 1, 7)
        self.assertEqual(plan.iteration[1], 64)
        with self.assertRaises(ValueError):
            plan.set_prompt_len(1089)


if __name__ == '__main__':
    unittest.main()
