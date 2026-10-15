# SPDX-License-Identifier: BSD-3-Clause
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import r16_forward_sass_validation as proof


def fixture():
    identity = dict(plan=dict(model='resnet18', phase='forward', batch_lo=1,
        batch_hi=1, seq=1, pg='l2', gemms=[]), defines=dict(TILEMEGA_ARCH_ID='900',
        TILEMEGA_CONV_TMA_IM2COL='1', TILEMEGA_DM_WGMMA_INLINE='1', TILEMEGA_GEMM_IMPL_1='2'))
    gemms, runtime, convs, bindings = [], [], [], []
    for index in range(21):
        selected = 1 <= index <= 19
        variant = 1 if selected else 0
        identity['plan']['gemms'].append(dict(index=index, split_k=1,
            tile_m=64, tile_n=128, tile_k=64, stages=2,
            impl='wgmma' if selected else 'mma16'))
        runtime.append(f'  {{{variant}u, 1u, 64u, 128u, 64u, 2u}},')
        access = 0 if index == 20 else 1
        gemms.append('  {64, 576, 0u, 1u, 2u, 3u, 0.f, 0u, 4294967295u, '
            '4294967295u, 4294967295u, {static_cast<DmAAccess>('+str(access)+'u), '
            'static_cast<DmBAccess>(0u), 3136u, 0u, 0u, '+str(index)+'u, '
            '4294967295u, 4294967295u, 4294967295u, 0ull, '
            '{static_cast<DmWriteKind>(0u), 1u, 2u, 4294967295u}}, {}},')
        convs.append('  {1u, 56u, 56u, 64u, 64u, 3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u, 56u, 56u, 0u, 1u},')
        bindings.append(f'template<> struct DmGemmBinding<{index}, {variant}> {{ static constexpr unsigned roles = 1u; }};')
    generated = '\n'.join(bindings)+'\n'
    for kind, name, rows in [('GemmDesc', 'kGemms', gemms),
            ('ConvDesc', 'kConvolutions', convs), ('GemmRuntimeDesc', 'kRuntimeGemms0', runtime)]:
        generated += f'constexpr {kind} {name}[] = {{\n'+'\n'.join(rows)+'\n};\n'
    audit = dict(functions=[], counts=dict(HGMMA=656), hot_local_instructions=84)
    sass = []
    for name in ('tilemega_l1_kernel', 'tilemega_l1_loop_kernel',
                 'tilemega_l2_kernel', 'tilemega_stage_kernel'):
        sass.append('Function : '+name+'\n')
        regions = []
        for slot in range(41):
            begin, end = slot*64, slot*64+48
            conv = slot < 20
            regions.append(dict(begin=hex(begin), end=hex(end),
                local_instructions=[] if conv else [dict(instruction='LDL')]))
            sass += [f'/*{begin+i*8:04x}*/ HGMMA;\n' for i in range(4)]
            sass.append(f'/*{begin+32:04x}*/ '+('UTMALDG.4D' if conv else 'LDL')+';\n')
        audit['functions'].append(dict(function=name, regions=regions))
    return identity, generated, audit, sass


class SelectedProductionLoopsTest(unittest.TestCase):
    def check(self, identity, generated, audit, sass):
        with patch.object(proof, 'source_gate', return_value={'fixture': 'pinned'}):
            return proof.selected_loop_proof(identity, generated, audit, sass, '/unused')

    def test_retained_unreachable_dense_spills_are_reported_separately(self):
        result = self.check(*fixture())
        self.assertEqual(len(result['selected_runtime_routes']), 19)
        self.assertEqual(result['selected_K_loop_local_instructions'], 0)
        self.assertEqual(result['conservative_whole_generic_local_instructions'], 84)
        self.assertFalse(result['whole_binary_zero_spill_claim'])

    def test_real_dense_wg_classifier_is_never_waived(self):
        identity, generated, audit, sass = fixture()
        identity['plan']['gemms'][20]['impl'] = 'wgmma'
        generated = generated.replace('  {0u, 1u, 64u, 128u, 64u, 2u},\n};',
                                      '  {1u, 1u, 64u, 128u, 64u, 2u},\n};')
        with self.assertRaisesRegex(ValueError, 'real dense'):
            self.check(identity, generated, audit, sass)

    def test_split_and_scaled_paths_require_a_new_proof(self):
        for mode in ('split', 'scale'):
            identity, generated, audit, sass = fixture()
            if mode == 'split':
                identity['plan']['gemms'][1]['split_k'] = 2
                generated = generated.replace('{1u, 1u, 64u, 128u, 64u, 2u}',
                    '{1u, 2u, 64u, 128u, 64u, 2u}', 1)
            else:
                generated = generated.replace('4294967295u, 0ull,', '7u, 0ull,')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(identity, generated, audit, sass)

    def test_actual_im2col_hot_locals_are_rejected(self):
        identity, generated, audit, sass = fixture()
        audit['functions'][0]['regions'][0]['local_instructions'] = [dict(instruction='LDL')]
        with self.assertRaisesRegex(ValueError, 'selected im2col'):
            self.check(identity, generated, audit, sass)

    def test_missing_loop_coverage_or_wrong_model_is_rejected(self):
        for mode in ('instruction', 'model', 'kernel'):
            identity, generated, audit, sass = fixture()
            if mode == 'instruction':
                sass.append('/*ffff*/ HGMMA;\n')
            elif mode == 'model':
                identity['plan']['model'] = 'nafnet'
            else:
                audit['functions'].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(identity, generated, audit, sass)

    def test_unreviewed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runtime.h'
            path.write_text('changed runtime')
            with patch.object(proof, 'APPROVED_SOURCE_HASHES', {'runtime.h': '0'*64}):
                with self.assertRaisesRegex(ValueError, 'new proof required'):
                    proof.source_gate(directory)


if __name__ == '__main__':
    unittest.main()
