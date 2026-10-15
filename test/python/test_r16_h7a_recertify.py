# SPDX-License-Identifier: BSD-3-Clause
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_r16_h7a_recertify as driver

class ResourceRecertificationTest(unittest.TestCase):
    def setUp(self):
        self.old=dict(resident=2,l1=2,l2=2,registers_l1=255,registers_l2=255,dynamic_shared=49152,threads=128)
        self.target=dict(resources=dict(num_sms=132),sm_major=9,sm_minor=0)
        self.actual=dict(num_sms=132,major=9,minor=0,kernels=[dict(name=k,registers=255,resident=2,max_threads=128,threads=128,dynamic_shared=49152) for k in ('tilemega_l1_kernel','tilemega_l2_kernel','tilemega_l1_loop_kernel','tilemega_stage_kernel')])
    def test_final_artifact_is_checked_in_addition_to_original_query(self):
        self.assertEqual(driver.resource_gate(self.old,dict(self.old),self.actual,self.target,1),2)
        self.actual['kernels'][1]['resident']=0
        with self.assertRaisesRegex(ValueError,'actual kernel'):driver.resource_gate(self.old,dict(self.old),self.actual,self.target,1)
    def test_missing_kernel_wrong_gpu_and_changed_official_resources_rejected(self):
        for damage in ('missing','gpu','query'):
            with self.subTest(damage=damage):
                actual=copy.deepcopy(self.actual);query=dict(self.old)
                if damage=='missing':actual['kernels'].pop()
                elif damage=='gpu':actual['num_sms']=120
                else:query['registers_l1']=254
                with self.assertRaises(ValueError):driver.resource_gate(self.old,query,actual,self.target,1)
    def test_final_artifact_thread_or_smem_mismatch_rejected(self):
        for key in ('threads','max_threads','dynamic_shared','registers'):
            actual=copy.deepcopy(self.actual);actual['kernels'][0][key]=0
            with self.subTest(key=key),self.assertRaises(ValueError):driver.resource_gate(self.old,dict(self.old),actual,self.target,1)

if __name__=='__main__':unittest.main()
