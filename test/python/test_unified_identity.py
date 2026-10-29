"""One artifact schema for LLM, DNN and MoE; no CUDA context is needed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tilemega.bench.identity_join import checked_join
from tilemega.build.identity import bind_execution, digest, generate, sha, verify


class UnifiedIdentityTests(unittest.TestCase):
    def fixture(self, directory, dm=True):
        root = Path(directory)
        so = root / 'plan.so'
        so.write_bytes(b'binary')
        (root / 'runtime.h').write_text('runtime source')
        (root / 'compiler').write_bytes(b'compiler binary')
        plan = dict(pg='l2', phase='forward' if dm else 'decode', gemms=[dict(index=0)])
        if dm:
            plan.update(frontend='dnn', memory_arena_bytes=4096, dm_pool_la=True,
                        dm_reduction_mask=3, dm_reduction_proved_stages=[2, 4],
                        dnn_structure=dict(deferred_ln=True))
        Path(str(so) + '.plan.json').write_text(json.dumps(plan))
        Path(str(so) + '.cu').write_text(
            '#define TILEMEGA_WATCHDOG 0\n' + ('#define TILEMEGA_DM_SUPPORT 1\n' if dm else ''))
        Path(str(so) + '.build_command.txt').write_text(
            'cucc -DTILEMEGA_TRACE_STAGE=0 plan.so.cu -o plan.so')
        Path(str(so) + '.ptxas.log').write_text(""" maca info  : Function properties for  _ZN8tilemega7codegen18tilemega_l1_kernelEPKNS0_6ParamsE
  16 bytes stack frame
 maca info  : Used  128 MTregisters, 40 STregisters, 0 bytes shared mem
 maca info  : staticMaxWarps/PEU : 2
 maca info  : Function properties for  _ZN8tilemega7codegen18tilemega_l2_kernelEPKNS0_6ParamsE
  0 bytes stack frame
 maca info  : Used  120 MTregisters, 36 STregisters, 0 bytes shared mem
""")
        inputs = dict(root=str(root), head='fixture', source_files={'runtime.h': sha(root / 'runtime.h')},
                      compiler=str(root / 'compiler'), compiler_sha256=sha(root / 'compiler'))
        inputs['source_digest'] = digest(inputs['source_files'])
        source = root / 'source.json'
        source.write_text(json.dumps(inputs))
        return so, source, inputs

    def capture(self, options, out):
        out.mkdir()
        table = out / 'host.macros.txt'
        table.write_text('#define TILEMEGA_DM_SUPPORT 1\n#define TILEMEGA_TRACE_STAGE 0\n')
        return dict(compiler_command=options, translation_units=[
            dict(macros_file=str(table), macros_sha256=sha(table))])

    def build(self, so, source, capture=None):
        library = Mock()
        library.tm_plan_shared_bytes.return_value = 32768
        with patch('tilemega.build.identity.ctypes.CDLL', return_value=library), \
                patch('tilemega.build.identity.subprocess.check_output', return_value='nvcc 12.8'), \
                patch('tilemega.build.compiler_macros.capture', side_effect=capture or self.capture):
            return generate(so, so.parent, source)

    def test_common_schema_and_optional_dm_section(self):
        for dm in (False, True):
            with self.subTest(dm=dm), tempfile.TemporaryDirectory() as directory:
                so, source, inputs = self.fixture(directory, dm)
                actual = self.build(so, source)
                self.assertEqual(actual['schema'], 1)
                self.assertEqual(verify(so, expected_source=inputs), actual)
                self.assertEqual(actual['defines']['TILEMEGA_WATCHDOG'], '0')
                self.assertEqual(actual['arch'], 'xcore1000')
                self.assertEqual('dm' in actual, dm)
                self.assertEqual('complete_macros' in actual, dm)
                if dm:
                    self.assertEqual(actual['dm']['dm_reduction_proved_stages'], [2, 4])
                    self.assertEqual(actual['dm']['memory_arena_bytes'], 4096)
                    self.assertTrue(actual['dm']['dnn_structure']['deferred_ln'])
                self.assertTrue(bind_execution(actual, 'L1')['spill'])
                self.assertFalse(bind_execution(actual, 'L2')['spill'])

    def test_static_v2_identity_uses_the_same_source_snapshot_and_kernel_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            so, source, inputs = self.fixture(directory, dm=False)
            manifest = dict(executor='mx_static', version=5, phase='decode', pg='off',
                            kv_layout='vt', kv_storage='block_fragments', threads=576,
                            classes={'qkv': {'tn': 16, 'wn': 1, 'prefetch': 1}},
                            b_min=2, attn_prefetch=2, decisions={'DT-P': 'P-B'}, watchdog=0)
            Path(str(so) + '.plan.json').write_text(json.dumps(manifest))
            Path(str(so) + '.ptxas.log').write_text(
                'Function properties for _Zmx_decode_kernel_v2X\n'
                '0 bytes stack frame\nUsed 167 MTregisters, 105 STregisters, 0 bytes shared mem\n')
            actual = self.build(so, source)
            self.assertEqual(verify(so, expected_source=inputs), actual)
            fields = actual['implementations']
            self.assertEqual(fields['attention']['impl'], 'MxAttentionV2')
            self.assertEqual(fields['classes'], manifest['classes'])
            bound = bind_execution(actual, 'L1', loop=True)
            self.assertEqual(bound['kernel'], '_Zmx_decode_kernel_v2X')
            self.assertFalse(bound['spill'])
            with self.assertRaises(ValueError):
                bind_execution(actual, 'L2')

    def test_native_moe_prompt_and_decode_bind_the_actual_kernel(self):
        for prompt in (False, True):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as directory:
                so, source, inputs = self.fixture(directory, dm=False)
                manifest = dict(executor='mx_static_prompt' if prompt else 'mx_static',
                    program_kind='moe', version=5, phase='prefill' if prompt else 'decode', pg='off',
                    kv_layout='vt', kv_storage='block_fragments', threads=512,
                    classes={'router': {'tn': 32, 'wn': 1, 'prefetch': 1}},
                    b_min=4, attn_prefetch=0, decisions={'DT-P': 'P-B'}, watchdog=0)
                Path(str(so) + '.plan.json').write_text(json.dumps(manifest))
                Path(str(so) + '.ptxas.log').write_text(
                    'Function properties for _Zmx_moe_kernel_v2X\n'
                    '0 bytes stack frame\nUsed 243 MTregisters, 108 STregisters, 0 bytes shared mem\n')
                actual = self.build(so, source)
                self.assertEqual(verify(so, expected_source=inputs), actual)
                self.assertEqual(actual['implementations']['attention']['task_body'], 'MoeProgramV2')
                self.assertEqual(actual['implementations']['executor'], manifest['executor'])
                for loop in (False, True):
                    self.assertEqual(bind_execution(actual, 'L1', loop)['kernel'], '_Zmx_moe_kernel_v2X')
                with self.assertRaises(ValueError):
                    bind_execution(actual, 'L2')

    def test_resource_and_command_mutations_are_rejected(self):
        for suffix in ('.ptxas.log', '.build_command.txt', '.cu', '.plan.json', ''):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                so, source, _ = self.fixture(directory)
                self.build(so, source)
                Path(str(so) + suffix).write_bytes(b'mutated')
                with self.assertRaises(ValueError):
                    verify(so)

    def test_preprocessor_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            so, source, _ = self.fixture(directory)
            actual = self.build(so, source)
            Path(actual['complete_macros']['translation_units'][0]['macros_file']).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'macro table changed'):
                verify(so)

    def test_compiler_and_source_cannot_change_during_build(self):
        for name in ('runtime.h', 'compiler'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                so, source, _ = self.fixture(directory)
                def change(options, out):
                    result = self.capture(options, out)
                    (so.parent / name).write_bytes(b'changed during preprocessing')
                    return result
                with self.assertRaisesRegex(ValueError, 'source changed|compiler changed'):
                    self.build(so, source, change)
                self.assertFalse(Path(str(so) + '.identity.json').exists())

    def test_build_receipt_cannot_be_borrowed(self):
        with tempfile.TemporaryDirectory() as directory:
            so, source, inputs = self.fixture(directory)
            self.build(so, source)
            changed = dict(inputs, compiler_sha256='different compiler')
            with self.assertRaisesRegex(ValueError, 'requested build'):
                verify(so, expected_source=changed)

    def test_execution_and_trace_stay_in_join_key(self):
        with tempfile.TemporaryDirectory() as directory:
            so, source, _ = self.fixture(directory)
            actual = self.build(so, source)
            one = dict(execution_identity=bind_execution(actual, 'L1'))
            two = dict(execution_identity=bind_execution(actual, 'L2'))
            self.assertEqual(checked_join(one, copy.deepcopy(one))['left'], one)
            with self.assertRaisesRegex(ValueError, 'different artifact'):
                checked_join(one, two)
            actual['trace'] = True
            traced = dict(execution_identity=bind_execution(actual, 'L1'))
            with self.assertRaisesRegex(ValueError, 'trace timing'):
                checked_join(traced, traced)

    def test_missing_kernel_resources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            so, source, _ = self.fixture(directory, dm=False)
            Path(str(so) + '.ptxas.log').write_text('no resource information')
            with self.assertRaisesRegex(ValueError, 'missing ptxas'):
                self.build(so, source)


if __name__ == '__main__':
    unittest.main()
