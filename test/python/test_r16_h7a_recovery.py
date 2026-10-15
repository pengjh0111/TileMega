# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import run_r16_h7a_recovery as driver

class H7aAuditTest(unittest.TestCase):
    def test_cold_local_traffic_outside_smallest_compute_loop_is_retained(self):
        data=['Function : full_model_kernel','/*0000*/ LDL R4, [R1];','/*0010*/ MOV R5, 0;','/*0020*/ HGMMA.64X128 R12, R4, R6;','/*0030*/ IADD3 R5, R5, 1, RZ;','/*0040*/ @P0 BRA 0x20;','/*0050*/ STL [R1], R12;','/*0060*/ UTMALDG.4D R8, R4, R6;']
        a=driver.audit_sass(iter(data));self.assertEqual(a['hot_local_instructions'],0);self.assertEqual(a['functions'][0]['whole_function_local_instructions'],2);self.assertEqual(a['counts']['UTMALDG.4D'],1)
    def test_local_traffic_inside_compute_loop_is_rejected_by_gate_input(self):
        data=['Function : kernel','/*0020*/ HGMMA.64X128 R12, R4, R6;','/*0030*/ LDL R8, [R1+0x10];','/*0040*/ @P0 BRA 0x20;']
        a=driver.audit_sass(data);self.assertEqual(a['hot_local_instructions'],1)
    def test_each_function_and_straight_line_issue_region_are_audited(self):
        data=['Function : callee','/*0000*/ HGMMA.64X128 R12, R4, R6;','/*0010*/ STL [R1], R12;','/*0020*/ HGMMA.64X128 R12, R4, R6;','Function : caller','/*0000*/ HGMMA.64X128 R12, R4, R6;','/*0010*/ BRA 0x0;']
        a=driver.audit_sass(data);self.assertEqual(len(a['functions']),2);self.assertEqual(a['counts']['HGMMA'],3);self.assertEqual(a['hot_local_instructions'],1)

if __name__=='__main__':unittest.main()
