"""Implementation identity must agree for both ordinary and DM artifacts."""
import unittest

from tilemega.build.feature_contract import INTEGER_FEATURES, build_defines, implementations, validate


class FeatureContractTests(unittest.TestCase):
    def test_every_integer_feature_in_llm_and_dm_plans(self):
        for dm in (False, True):
            for feature, (macro, default) in INTEGER_FEATURES.items():
                for value in (default, 1 if default == 0 else 0):
                    with self.subTest(dm=dm, feature=feature, value=value):
                        plan = dict(pg='l2', **{feature: value})
                        source = ('#define TILEMEGA_DM_SUPPORT 1\n' if dm else '')
                        defines = build_defines(source, ['nvcc', f'-D{macro}={value}'])
                        validate(plan, defines)
                        plan[feature] = value + 1
                        with self.assertRaisesRegex(ValueError, 'manifest'):
                            validate(plan, defines)

    def test_attention_and_handoff_implementations(self):
        for dm in (False, True):
            for swap in (0, 1):
                for pg in ('l2', 'pages'):
                    plan = dict(pg=pg, attention_impl='pvswap' if swap else 'mma16',
                                nonpaged_la=pg != 'pages')
                    defines = build_defines('#define TILEMEGA_DM_SUPPORT '+str(int(dm))+'\n',
                        ['nvcc', f'-DTILEMEGA_ATTENTION_PVSWAP={swap}', '-DTILEMEGA_NONPAGED_LA=1'])
                    validate(plan, defines)
                    plan['attention_impl'] = 'mma16' if swap else 'pvswap'
                    with self.assertRaisesRegex(ValueError, 'attention_impl'):
                        validate(plan, defines)

    def test_attention_family_macro_matches_identity(self):
        for index,name in enumerate(('mma16','pvswap','pvswap2','simt')):
            for dm in (False,True):
                defines=build_defines('#define TILEMEGA_DM_SUPPORT '+str(int(dm))+'\n',
                                      ['nvcc',f'-DTILEMEGA_ATTENTION_IMPL={index}'])
                validate(dict(attention_impl=name),defines)
                with self.assertRaises(ValueError):
                    validate(dict(attention_impl='invalid'),defines)
        with self.assertRaises(ValueError):validate(dict(attention_impl='simt'),{'TILEMEGA_ATTENTION_IMPL':'4'})

    def test_defaults_and_inactive_nonpaged_handoff(self):
        validate(dict(pg='pages', nonpaged_la=False, attention_buffers=2,
                      attention_impl='mma16'), {})
        with self.assertRaisesRegex(ValueError, 'nonpaged_la'):
            validate(dict(pg='l2', nonpaged_la=True), {})

    def test_generated_source_cannot_silently_override_build_options(self):
        with self.assertRaisesRegex(ValueError, 'overrides compiler macro'):
            build_defines('#define TILEMEGA_ATTENTION_PVSWAP 0\n',
                          ['nvcc', '-DTILEMEGA_ATTENTION_PVSWAP=1'])
        self.assertEqual(build_defines('#define TILEMEGA_ATTENTION_PVSWAP 1\n',
            ['nvcc', '-DTILEMEGA_ATTENTION_PVSWAP=1'])['TILEMEGA_ATTENTION_PVSWAP'], '1')

    def test_conflicting_command_definitions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting compiler macro'):
            build_defines('', ['nvcc', '-DTILEMEGA_WATCHDOG=0', '-DTILEMEGA_WATCHDOG=1'])

    def test_generated_ifndef_defaults_remain_overridable(self):
        source = '#ifndef TILEMEGA_WATCHDOG\n#define TILEMEGA_WATCHDOG 1\n#endif\n'
        self.assertEqual(build_defines(source, ['nvcc', '-DTILEMEGA_WATCHDOG=0'])[
            'TILEMEGA_WATCHDOG'], '0')
        self.assertEqual(build_defines(source, ['nvcc'])['TILEMEGA_WATCHDOG'], '1')

    def test_source_checks_use_the_binary_identity_implementation_fields(self):
        manifest = dict(pg='l2', phase='decode', attention_impl='pvswap',
                        attention_noinline=True, gemms=[dict(index=0, impl='gemv')])
        defines = dict(TILEMEGA_MMA_REG_PIPE='1', TILEMEGA_EP_DIRECT='1',
                       TILEMEGA_ATTENTION_BUFFERS='1', TILEMEGA_ATTENTION_FRONTIER='1')
        fields = implementations(manifest, defines)
        self.assertEqual(fields['gemms'][0], dict(index=0, impl='gemv',
            task_body='ServingGemvTaskBody', mma_reg_pipe=1, ep_direct=1))
        self.assertEqual(fields['attention']['page_policy'], 'WarpPrivate')
        self.assertEqual(fields['attention']['impl'], 'pvswap')
        self.assertTrue(fields['attention']['noinline'])
        self.assertEqual(fields['attention']['buffers'], 1)
        self.assertTrue(fields['attention']['frontier'])

    def test_static_decode_v2_identity_has_actual_attention_and_geometry(self):
        manifest = dict(executor='mx_static', version=5, kv_layout='vt',
                        kv_storage='block_fragments', threads=512, classes={'qkv': {'tn': 16, 'wn': 1}},
                        b_min=4, attn_prefetch=2, decisions={'DT-P': 'P-B'})
        fields = implementations(manifest, {})
        self.assertEqual(fields['attention'], dict(impl='MxAttentionV2', task_body='DecodeProgramV2'))
        self.assertEqual(fields['threads'], 512)
        self.assertEqual(fields['classes'], manifest['classes'])
        self.assertEqual(fields['b_min'], 4)
        self.assertNotIn('attention_tmin', fields)
        legacy = dict(executor='mx_static', kv_layout='vt', classes={}, attention_tmin=128)
        self.assertEqual(implementations(legacy, {})['attention']['impl'], 'MxDecodeAttention')

    def test_fragment_prefill_layout_is_bound_to_build_and_identity(self):
        manifest = dict(phase='prefill', pg='off', gemms=[], kv_layout='vt', kv_storage='block_fragments')
        defines = dict(TILEMEGA_KV_LAYOUT_VT='1', TILEMEGA_MX_KV_FRAGMENTS_V2='1')
        validate(manifest, defines)
        self.assertEqual(implementations(manifest, defines)['attention']['kv_storage'], 'block_fragments')
        defines['TILEMEGA_MX_KV_FRAGMENTS_V2'] = '0'
        with self.assertRaisesRegex(ValueError, 'kv_storage'):
            validate(manifest, defines)

    def test_moe_v3_identity_requires_its_macro_and_512_threads(self):
        manifest=dict(executor='mx_static',program_kind='moe',version=6,program_version=3,
            kv_layout='vt',kv_storage='block_fragments',threads=512,classes={},b_min=1,
            attn_prefetch=0,split=1,route_table_offset=40000,decisions={})
        with self.assertRaisesRegex(ValueError,'compiler macro'):
            validate(manifest,{})
        validate(manifest,{'TILEMEGA_MX_STATIC_MOE_V3':'1'})
        self.assertEqual(implementations(manifest,{})['attention'],
                         dict(impl='MxAttentionV2',task_body='MoeProgramV3'))
        manifest['threads']=576
        with self.assertRaisesRegex(ValueError,'512 threads'):
            validate(manifest,{'TILEMEGA_MX_STATIC_MOE_V3':'1'})


if __name__ == '__main__':
    unittest.main()
