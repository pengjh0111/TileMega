"""Implementation identity must agree for both ordinary and DM artifacts."""
import unittest

from tilemega.build.feature_contract import INTEGER_FEATURES, build_defines, validate


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


if __name__ == '__main__':
    unittest.main()
