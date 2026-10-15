# SPDX-License-Identifier: BSD-3-Clause
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_r16_forward_seed_candidates import KINDS, resource_gate


class ActualSeedResourcesTest(unittest.TestCase):
    def setUp(self):
        self.target = dict(resources=dict(num_sms=132), sm_major=9, sm_minor=0)
        self.actual = dict(num_sms=132, major=9, minor=0, kernels=[
            dict(name=kind, registers=128, resident=1, threads=128,
                 max_threads=128, dynamic_shared=70000) for kind in KINDS])

    def test_actual_final_executor_resources_are_required(self):
        self.assertEqual(resource_gate(self.actual, self.target, 70000), 1)
        for field, value in [('resident', 0), ('registers', 0), ('max_threads', 64),
                             ('dynamic_shared', 49152), ('threads', 160)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.actual)
                changed['kernels'][2][field] = value
                with self.assertRaisesRegex(ValueError, 'declared placement'):
                    resource_gate(changed, self.target, 70000)

    def test_device_and_complete_kernel_set_are_checked(self):
        changed = copy.deepcopy(self.actual)
        changed['major'] = 8
        with self.assertRaisesRegex(ValueError, 'bound target'):
            resource_gate(changed, self.target, 70000)
        for change in ('missing', 'duplicate'):
            changed = copy.deepcopy(self.actual)
            if change == 'missing':
                changed['kernels'].pop()
            else:
                changed['kernels'][-1]['name'] = KINDS[0]
            with self.assertRaisesRegex(ValueError, 'kernel set'):
                resource_gate(changed, self.target, 70000)


if __name__ == '__main__':
    unittest.main()
