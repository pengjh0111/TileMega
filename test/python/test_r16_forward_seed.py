# SPDX-License-Identifier: BSD-3-Clause
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from check_r16_forward_seed import check_matrix,model_counts

class FullForwardSeedContractTest(unittest.TestCase):
    def setUp(self):
        self.source='constexpr ModelSpec kModel = {kDims, ScalarType::kBF16, kBuffers, 89u, kGemms, 1u, kStages, 24u};\nconstexpr GemmRuntimeDesc kRuntimeGemms0[] = {\n{0u,1u,64u,128u,64u,2u},\n};'
        self.shape=dict(tile_m=64,tile_n=128,tile_k=64,stages=2,split_k=1)
        self.manifest=dict(gemms=[dict(index=0,impl='mma16',**self.shape)])
    def test_complete_descriptor_and_actual_dispatch_are_both_required(self):
        self.assertEqual(check_matrix(self.manifest,self.source,(89,1,24),self.shape),(89,1,24))
        with self.assertRaises(ValueError):check_matrix(self.manifest,self.source.replace('24u','0u'),(89,1,24),self.shape)
    def test_consistently_labelled_truncated_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'full-model'):check_matrix(self.manifest,self.source.replace('89u','2u').replace('24u','1u'),(89,1,24),self.shape)
    def test_manifest_cannot_certify_different_actual_geometry(self):
        changed=copy.deepcopy(self.manifest);changed['gemms'][0]['tile_m']=128
        with self.assertRaisesRegex(ValueError,'generated GEMM geometry'):check_matrix(changed,self.source,(89,1,24),self.shape)

if __name__=='__main__':unittest.main()
