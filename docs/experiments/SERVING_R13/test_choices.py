#!/usr/bin/env python3
import unittest
from builds_r13 import project
from choose_r13 import prefill,phase_c,final
class Choices(unittest.TestCase):
    def test_prefill_distinguishability(self):
        self.assertEqual(prefill({'llama_B1':{'B0-pfR12b':[10,10.1,9.9],'B0-pfR10':[9,9.1,8.9]}})['llama_B1'],'PF-R10-noWD')
        self.assertEqual(prefill({'llama_B1':{'B0-pfR12b':[10,10.1,9.9],'B0-pfR10':[9.9,10,9.8]}})['llama_B1'],'PF-R12b-noWD')
    def test_missing_data_never_triggers(self):
        self.assertFalse(phase_c({})['selected'])
    def test_preregistered_mb_thresholds(self):
        d={'mb':{'ceiling_gbps':1000,'loader1_gbps':940,'loader2_gbps':980,'gemv_best_instruction_rel':.03,'gemv_tile_layout_rel':.01}}
        self.assertEqual(phase_c(d)['selected'],['C-PG1','C-BW'])
    def test_projection_clamps_32k_stage(self):
        d={'gemms':[dict(tile_n=256,tile_k=64,stages=2,split_k=1)]}
        g=project(d,'paged')['gemms'][0]
        self.assertEqual(g['tile_n'],128);self.assertEqual(d['gemms'][0]['tile_n'],256)
        self.assertEqual(project(d,'paged',8192)['gemms'][0]['tile_n'],64)
if __name__=='__main__':unittest.main()
