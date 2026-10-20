"""Native serving receipts must certify the generated calls, not a label."""
import copy
import json
from pathlib import Path
import unittest

from tilemega.build.feature_contract import build_defines, implementations, validate_gemm_dispatch


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        c = json.loads((root / 'configs/serving/sm90r2/development/qwen3_0_6b_b1.tmexec.json').read_text())
        self.plan = dict(native_dense=c, native_origin='fixed', runtime_batch=1,
                         phase='decode', pg='native', mode='L1', loop=1, capacity=1088,
                         residency=1, gemms=[])
        self.source = ('#define TILEMEGA_STATIC_DENSE_NATIVE 1\n#define TILEMEGA_ARCH_ID 900\n'
                       '#define TILEMEGA_DN_VECTOR_SUMS 1\n#define TILEMEGA_SWIGLU_U 8\n'
                       'constexpr char kNativeCoordinateIdentity[]=' + json.dumps(json.dumps(c)) + ';\n')
        shapes = [('qkv', 4096, 1024, 'kStore'), ('o', 1024, 2048, 'kResidual'),
                  ('gate_up', 6144, 1024, 'kSwiGLU'), ('down', 1024, 3072, 'kResidual'),
                  ('lm_head', 151936, 1024, 'kArgmaxPartial')]
        for i, (family, n, k, op) in enumerate(shapes):
            g = c['gemms'][family]
            tn, tk = g['tile_n'], g['slot_bytes'] // (2*g['tile_n'])
            self.plan['gemms'].append(dict(index=i, family=family, impl='native_mma_sync',
                n=n, k=k, tile_m=16, tile_n=tn, tile_k=tk, stages=3, split_k=1,
                warps_n=g['warps_n'], warps_k=g['warps_k']))
            self.source += (f'using Stage_{i}=Gemm<1,{n},{k},{tn},{tk},3,'
                            f'{g["warps_n"]},{g["warps_k"]},backend::ServingEpilogueOp::{op},8,'
                            '0,1,kAbsent,2,kAbsent,3,4>;\n')
        self.source += 'using Stage_5=Attention<1,128,2,1,16,64,3,0,1088,0,1,2,3,4,5,6,7,8,9,10>;\n'
        self.defines = build_defines(self.source, [])

    def test_receipt_matches_typed_native_calls(self):
        rows = validate_gemm_dispatch(self.plan, self.defines, self.source)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['split_k'] == 1 for row in rows))
        info = implementations(self.plan, self.defines)
        self.assertEqual(info['attention']['partition'], 'actual_past')
        self.assertEqual(info['gemms'][0]['task_body'], 'Sm90DenseGemm')

    def test_manifest_cannot_claim_uncompiled_geometry_or_structure(self):
        for field, value in [('tile_n', 64), ('split_k', 2), ('impl', 'mma16'), ('warps_k', 2)]:
            plan = copy.deepcopy(self.plan)
            plan['gemms'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'GEMM call'):
                validate_gemm_dispatch(plan, self.defines, self.source)
        plan = copy.deepcopy(self.plan)
        plan['native_dense']['attention']['chunks'] = 8
        with self.assertRaisesRegex(ValueError, 'coordinate identity'):
            validate_gemm_dispatch(plan, self.defines, self.source)

    def test_missing_attention_or_legacy_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'attention call is missing'):
            validate_gemm_dispatch(self.plan, self.defines, self.source.split('using Stage_5')[0])
        with self.assertRaisesRegex(ValueError, 'attention call differs'):
            validate_gemm_dispatch(self.plan, self.defines, self.source.replace('1,128,2,1,16,64', '1,128,2,1,8,64'))
        with self.assertRaisesRegex(ValueError, 'legacy execution path'):
            validate_gemm_dispatch(self.plan, self.defines, self.source + '\nvoid tilemega_l1_kernel();')

    def test_register_prefetch_is_part_of_the_native_coordinate_identity(self):
        plan = copy.deepcopy(self.plan)
        for g in plan['gemms']:
            g['register_stages'] = 4
        for g in plan['native_dense']['gemms'].values():
            g['register_stages'] = 4
        source = self.source.replace('0,1,kAbsent,2,kAbsent,3,4>;',
                                     '0,1,kAbsent,2,kAbsent,3,4,4>;')
        start = source.index('constexpr char kNativeCoordinateIdentity[]=')
        end = source.index(';\n', start)
        source = source[:start] + 'constexpr char kNativeCoordinateIdentity[]=' + \
            json.dumps(json.dumps(plan['native_dense'])) + source[end:]
        rows = validate_gemm_dispatch(plan, build_defines(source, []), source)
        self.assertTrue(all(r['register_stages'] == 4 for r in rows))
        with self.assertRaisesRegex(ValueError, 'GEMM call differs'):
            validate_gemm_dispatch(plan, build_defines(source, []),
                                   source.replace('0,1,kAbsent,2,kAbsent,3,4,4>;',
                                                  '0,1,kAbsent,2,kAbsent,3,4,2>;', 1))


if __name__ == '__main__':
    unittest.main()
