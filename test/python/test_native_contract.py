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

    def test_historical_attention_is_explicitly_nonproduction(self):
        plan = copy.deepcopy(self.plan)
        plan['native_origin'] = 'historical_ablation'
        plan['historical_ablation'] = dict(production_eligible=False,attention_impl=3,attention_kv_block=128)
        source = self.source.split('using Stage_5')[0] + (
            '#define TILEMEGA_SM90R2_HISTORICAL_ABLATION 1\n#define TILEMEGA_ATTENTION_IMPL 3\n'
            'using Body_5=HistoricalAttention<1,128,2,1,128,1088,0,1,2,3,4,5,6,7,8,9>;\n'
            'using MergeBody_5=HistoricalMerge<1,128,2,128,1088,8,9,3>;\n')
        defines = build_defines(source, [])
        self.assertEqual(len(validate_gemm_dispatch(plan,defines,source)),5)
        self.assertFalse(implementations(plan,defines)['production_eligible'])
        for mutate in ('origin','eligible','marker','merge'):
            bad = copy.deepcopy(plan);text = source
            if mutate=='origin':bad['native_origin']='solver'
            if mutate=='eligible':bad['historical_ablation']['production_eligible']=True
            if mutate=='marker':text=text.replace('TILEMEGA_SM90R2_HISTORICAL_ABLATION 1','TILEMEGA_SM90R2_HISTORICAL_ABLATION 0')
            if mutate=='merge':text=text.replace('1088,8,9,3','1088,8,10,3')
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                validate_gemm_dispatch(bad,build_defines(text,[]),text)

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

    def test_arenas_and_preissue_must_be_bound_to_each_typed_stage(self):
        source = self.source
        for i in range(6):
            source = source.replace(f'using Stage_{i}=', f'using Body_{i}=')
            source += f'using Stage_{i}=Placement<Body_{i},0,0,0>;\n'
        self.assertEqual(len(validate_gemm_dispatch(self.plan, self.defines, source)), 5)
        for original, bad in [('Body_0,0,0,0', 'Body_0,1024,0,0'),
                              ('Body_0,0,0,0', 'Body_0,0,16,0'),
                              ('Body_0,0,0,0', 'Body_0,0,0,0,true'),
                              ('Body_5,0,0,0', 'Body_5,0,0,1024')]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'placement differs'):
                validate_gemm_dispatch(self.plan, self.defines, source.replace(original, bad))
        with self.assertRaisesRegex(ValueError, 'placement differs'):
            validate_gemm_dispatch(self.plan, self.defines,
                                   source.replace('using Stage_0=Placement<Body_0,0,0,0>;', ''))

    def test_late_issue_is_bound_to_the_generated_call(self):
        plan = copy.deepcopy(self.plan)
        plan['native_dense']['gemms']['qkv'].update(preissue_smem_bytes=16, preissue_late=True)
        source = self.source
        for i in range(6):
            source = source.replace(f'using Stage_{i}=', f'using Body_{i}=')
            values = '0,16,0,true' if i == 0 else '0,0,0'
            source += f'using Stage_{i}=Placement<Body_{i},{values}>;\n'
        source = source.replace(json.dumps(json.dumps(self.plan['native_dense'])),
                                json.dumps(json.dumps(plan['native_dense'])))
        self.assertEqual(len(validate_gemm_dispatch(plan, self.defines, source)), 5)
        with self.assertRaisesRegex(ValueError, 'placement differs'):
            validate_gemm_dispatch(plan, self.defines, source.replace('0,16,0,true', '0,16,0'))


if __name__ == '__main__':
    unittest.main()
