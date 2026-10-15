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

    def test_attention_storage_is_bound_to_the_arithmetic_family(self):
        validate(dict(attention_impl='wgmma',attention_page='uniform'),
                 {'TILEMEGA_ATTENTION_IMPL':'4'})
        validate(dict(attention_impl='mma16',attention_page='chunk16'),{})
        for impl,page,macro in [('wgmma','chunk16','4'),('mma16','uniform','0')]:
            with self.assertRaisesRegex(ValueError,'attention_page'):
                validate(dict(attention_impl=impl,attention_page=page),{'TILEMEGA_ATTENTION_IMPL':macro})
        with self.assertRaisesRegex(ValueError,'attention_page'):
            validate(dict(attention_impl='wgmma',attention_page='uniform'),
                     {'TILEMEGA_ATTENTION_IMPL':'4','TILEMEGA_ATTENTION_PAGE_UNIFORM':'0'})

    def test_encoder_identity_requires_the_hopper_macro(self):
        validate(dict(encoder_attention_impl='wgmma'),{'TILEMEGA_ENCODER_ATTENTION_WGMMA':'1'})
        validate(dict(encoder_attention_impl='mma16'),{})
        with self.assertRaisesRegex(ValueError,'encoder_attention_impl'):
            validate(dict(encoder_attention_impl='wgmma'),{})
        fields=implementations(dict(phase='forward',pg='l2',gemms=[],encoder_attention_impl='wgmma'),{'TILEMEGA_ENCODER_ATTENTION_WGMMA':'1'})
        self.assertEqual(fields['encoder_attention']['task_body'],'Sm90EncoderAttentionTaskBody')

    def test_prefill_hopper_identity_is_independent_of_decode(self):
        for name,index in [('mma16', '0'),('wgmma','1')]:
            defines={'TILEMEGA_PREFILL_ATTENTION_WGMMA':index}
            validate(dict(prefill_attention_impl=name),defines)
            with self.assertRaisesRegex(ValueError,'prefill_attention_impl'):
                validate(dict(prefill_attention_impl='wgmma' if name=='mma16' else 'mma16'),defines)
            fields=implementations(dict(phase='prefill',pg='off',gemms=[],prefill_attention_impl=name),defines)
            self.assertEqual(fields['attention']['impl'],name)
            self.assertEqual(fields['attention']['task_body'], 'Sm90PrefillAttentionTaskBody' if name=='wgmma' else 'FusedAttentionTaskBody')

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


    def test_convolution_loader_claim_is_bound_to_macro(self):
        validate({'conv_impl': 'cpasync'}, {})
        validate({'conv_impl': 'tma_im2col'}, {'TILEMEGA_CONV_TMA_IM2COL': '1'})
        with self.assertRaisesRegex(ValueError, 'conv_impl'):
            validate({'conv_impl': 'tma_im2col'}, {})
        with self.assertRaisesRegex(ValueError, 'conv_impl'):
            validate({'conv_impl': 'cpasync'}, {'TILEMEGA_CONV_TMA_IM2COL': '1'})


    def test_gemv_weight_cache_policy_is_bound_to_macro(self):
        validate({'gemv_load': 'ldg'}, {})
        validate({'gemv_load': 'nc'}, {'TILEMEGA_GEMV_LOAD_NC': '1'})
        with self.assertRaisesRegex(ValueError, 'gemv_load'):
            validate({'gemv_load': 'nc'}, {})
        with self.assertRaisesRegex(ValueError, 'gemv_load'):
            validate({'gemv_load': 'ldg'}, {'TILEMEGA_GEMV_LOAD_NC': '1'})


if __name__ == '__main__':
    unittest.main()
