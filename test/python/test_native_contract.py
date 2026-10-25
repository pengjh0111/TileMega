"""Native serving receipts must certify the generated calls, not a label."""
import copy
import json
import unittest

from tilemega.build.feature_contract import build_defines, implementations, validate_gemm_dispatch

ARENA = 113 * 1024
SHAPES = [('qkv', 4096, 1024, 'kStore', 'true'), ('o', 1024, 2048, 'kResidual', 'false'),
          ('gate_up', 6144, 1024, 'kSwiGLU', 'true'), ('down', 1024, 3072, 'kResidual', 'false'),
          ('lm_head', 151936, 1024, 'kArgmaxPartial', 'true')]
# Typed stages of a two-layer program: 0 embedding, 1..10 the layer pair
# (qkv, attention, o, gate_up, down twice), 11 lm_head, 12 argmax.
TYPED = [None, 0, 'attention', 1, 2, 3, 0, 'attention', 1, 2, 3, 4, None]


def gemm(tn, wn, tk, slots):
    return dict(tile_n=tn, warps_n=wn, warps_k=4 // wn, slot_bytes=2 * tn * tk, slots=slots,
                mma='mma_sync', register_stages=1)


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        # Qwen3-0.6B B1 at schema 3: stage i takes arena i % 2 (prompt §6 D5).
        self.coordinates = dict(
            schema=3, residency=1, swiglu_u=8, weight_stream=False, layer_loop=False,
            gemms=dict(qkv=gemm(32, 4, 256, 4), o=gemm(8, 1, 512, 4), gate_up=gemm(48, 2, 256, 4),
                       down=gemm(8, 1, 512, 6), lm_head=gemm(128, 4, 64, 8)),
            attention=dict(chunks=16, block_keys=64, slots=2, new_row_work=0))
        self.plan = self.manifest(self.coordinates)
        self.stream = []
        for family, n, k, op, norm in SHAPES:
            g = self.coordinates['gemms'][family]
            tk = g['slot_bytes'] // (2 * g['tile_n'])
            self.stream.append(min(g['slots'], k // tk) * g['slot_bytes'])

    def manifest(self, c):
        plan = dict(native_dense=c, native_origin='fixed', runtime_batch=1, phase='decode',
                    pg='native', mode='L1', loop=1, capacity=1088, residency=1, gemms=[])
        for i, (family, n, k, op, norm) in enumerate([*SHAPES[:4], *SHAPES[:4], SHAPES[4]]):
            g = c['gemms'][family]
            plan['gemms'].append(dict(index=i, family=family, impl='native_mma_sync', n=n, k=k,
                tile_m=16, tile_n=g['tile_n'], tile_k=g['slot_bytes'] // (2 * g['tile_n']),
                stages=g['slots'], split_k=1, warps_n=g['warps_n'], warps_k=g['warps_k']))
        return plan

    def source(self, c, prefixes=(0,) * 13, offsets=None, loop=None):
        offsets = offsets or [(i % 2) * ARENA for i in range(13)]
        loop = c['layer_loop'] if loop is None else loop
        text = ('#define TILEMEGA_STATIC_DENSE_NATIVE 1\n#define TILEMEGA_ARCH_ID 900\n'
                '#define TILEMEGA_SWIGLU_U 8\n'
                'constexpr char kNativeCoordinateIdentity[]=' + json.dumps(json.dumps(c)) + ';\n'
                'using Body_0=Embedding<1,1024,151936,1088,0,1,2>;\n'
                'using Body_12=Argmax<1,1187,1088,3,4,5>;\n')
        a = c['attention']
        for number, family in enumerate(TYPED):
            if family == 'attention':
                text += (f'using Body_{number}=Attention<1,8,128,2,1,{a["chunks"]},{a["block_keys"]},'
                         f'{a["slots"]},{a["new_row_work"]},1088,0,kLayerSlot|1u,kLayerSlot|2u,3,4,5,'
                         f'kLayerSlot|3u,kLayerSlot|4u,8,9,10,1,2>;\n')
            elif family is not None:
                name, n, k, op, norm = SHAPES[family]
                g = c['gemms'][name]
                tn, tk = g['tile_n'], g['slot_bytes'] // (2 * g['tile_n'])
                text += (f'using Body_{number}=Gemm<1,{n},{k},{tn},{tk},{g["slots"]},{g["warps_n"]},'
                         f'{g["warps_k"]},backend::ServingEpilogueOp::{op},8,{norm},kLayerSlot|0u,'
                         f'kAbsent,2,kAbsent,{g["register_stages"]},0>;\n')
        for i in range(13):
            text += f'using Stage_{i}=Placement<Body_{i},{offsets[i]},{prefixes[i]}>;\n'
        pair = ','.join(f'Stage_{i}' for i in range(1, 11))
        return text + (f'using NativeProgram=Program<1,1088,1,12,3,{ARENA},2,5,{str(loop).lower()},'
                       f'Stage_0,LayerPair<{pair}>,Stage_11,Stage_12>;\n')

    def check(self, c, source):
        return validate_gemm_dispatch(self.manifest(c), build_defines(source, []), source)

    def test_receipt_matches_typed_native_calls(self):
        rows = self.check(self.coordinates, self.source(self.coordinates))
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row['split_k'] == 1 for row in rows))
        info = implementations(self.plan, {})
        self.assertEqual(info['attention']['partition'], 'actual_past')
        self.assertEqual(info['gemms'][0]['task_body'], 'Sm90DenseGemm')

    def test_manifest_cannot_claim_uncompiled_geometry_or_structure(self):
        source = self.source(self.coordinates)
        for field, value in [('tile_n', 64), ('split_k', 2), ('impl', 'mma16'), ('warps_k', 2)]:
            plan = copy.deepcopy(self.plan)
            plan['gemms'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'GEMM call'):
                validate_gemm_dispatch(plan, build_defines(source, []), source)
        plan = copy.deepcopy(self.plan)
        plan['native_dense']['attention']['chunks'] = 8
        with self.assertRaisesRegex(ValueError, 'coordinate identity'):
            validate_gemm_dispatch(plan, build_defines(source, []), source)

    def test_schema_and_legacy_paths_rejected(self):
        source = self.source(self.coordinates)
        with self.assertRaisesRegex(ValueError, 'attention call differs'):
            self.check(self.coordinates, source.replace('using Body_7=Attention', 'using Body_7=Other'))
        with self.assertRaisesRegex(ValueError, 'attention call differs'):
            self.check(self.coordinates, source.replace('1,8,128,2,1,16,64', '1,8,128,2,1,8,64'))
        with self.assertRaisesRegex(ValueError, 'legacy execution path'):
            self.check(self.coordinates, source + '\nvoid tilemega_l1_kernel();')
        old = dict(self.coordinates, schema=2)
        with self.assertRaisesRegex(ValueError, 'coordinate schema'):
            self.check(old, self.source(old))

    def test_arenas_follow_the_stage_index(self):
        for offsets in ([0] * 13, [ARENA * ((i + 1) % 2) for i in range(13)]):
            with self.subTest(offsets=offsets), self.assertRaisesRegex(ValueError, 'placement differs'):
                self.check(self.coordinates, self.source(self.coordinates, offsets=offsets))
        source = self.source(self.coordinates).replace('using Stage_0=Placement<Body_0,0,0>;\n', '')
        with self.assertRaisesRegex(ValueError, 'placement differs'):
            self.check(self.coordinates, source)

    def test_weight_stream_prefix_is_bound_to_the_coordinate(self):
        # The attention stages (D128 Q2 S16) fit one arena; their stream is
        # two K/V slots.
        kv = 2 * 4 * 64 * 128
        prefixes = [0] + [kv if f == 'attention' else self.stream[f] for f in TYPED[1:12]] + [0]
        with self.assertRaisesRegex(ValueError, 'placement differs'):
            self.check(self.coordinates, self.source(self.coordinates, prefixes))
        stream = dict(self.coordinates, weight_stream=True)
        self.assertEqual(len(self.check(stream, self.source(stream, prefixes))), 9)
        self.assertEqual(len(self.check(stream, self.source(stream))), 9)
        bad = list(prefixes)
        bad[3] -= 16
        with self.assertRaisesRegex(ValueError, 'placement differs'):
            self.check(stream, self.source(stream, bad))

    def test_layer_loop_is_bound_to_the_coordinate(self):
        loop = dict(self.coordinates, layer_loop=True)
        self.assertEqual(len(self.check(loop, self.source(loop))), 9)
        with self.assertRaisesRegex(ValueError, 'program structure'):
            self.check(loop, self.source(loop, loop=False))
        with self.assertRaisesRegex(ValueError, 'program structure'):
            self.check(self.coordinates, self.source(self.coordinates).replace(
                'Stage_11,Stage_12>', 'Stage_12,Stage_11>'))


if __name__ == '__main__':
    unittest.main()
