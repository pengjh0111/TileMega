"""Check bench build CLI contracts without invoking cucc or a device."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[2] / 'experiments/mx5/build_bodies.py'
spec = importlib.util.spec_from_file_location('mx5_body_build', source)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BodyBuild(unittest.TestCase):
    def inputs(self, *arguments):
        args = build.parse_args(['--out', 'unused-cpu-test-directory', *arguments])
        source, keys = build.build_inputs(args)
        return args, source, keys

    def test_legacy_gemm_command_and_names(self):
        args, source, keys = self.inputs('--bench', '--tn', '16', '--depth', '1', '--threads', '512', '--pw', '0')
        self.assertEqual(str(source), 'experiments/mx5/stage_bench.cu')
        self.assertEqual(keys, [('tn16_d1', dict(MX5_TEST_TN=16, MX5_TEST_DEPTH=1), dict(tn=16, depth=1))])
        self.assertEqual(build.compiler_command(args, keys[0][1], source, Path('out/tn16_d1.so')), [
            '/opt/maca/tools/cu-bridge/bin/cucc', '-std=c++17', '-O3', '-shared',
            '-Xcompiler', '-fPIC,-resource-usage', '-fno-strict-aliasing',
            '-DTILEMEGA_SYNC_V3=1', '-DTILEMEGA_WATCHDOG=0', '-DMX5_TEST_THREADS=512',
            '-DMX5_TEST_TN=16', '-DMX5_TEST_DEPTH=1', '-DMX5_BENCH_PW=0', '-Iinclude',
            'experiments/mx5/stage_bench.cu', '-o', 'out/tn16_d1.so'])
        _, source, keys = self.inputs()
        self.assertEqual(str(source), 'test/unit/mx/decode_bodies_v2.cu')
        self.assertEqual(len(keys), 15)

    def test_legacy_head_unit_still_dispatches_depths_in_one_library(self):
        _, source, keys = self.inputs('--body', 'head', '--tn', '16', '128', '--depth', '1')
        self.assertEqual(str(source), 'test/unit/mx/decode_head_v2.cu')
        self.assertEqual(keys, [('tn16', dict(MX5_TEST_TN=16), dict(tn=16)),
                                ('tn128', dict(MX5_TEST_TN=128), dict(tn=128))])

    def test_legacy_attention_and_small_unit_bodies(self):
        _, source, keys = self.inputs('--body', 'attention', '--dim', '64', '--query', '3')
        self.assertEqual(str(source), 'test/unit/mx/decode_attention_v2.cu')
        self.assertEqual(keys, [('d64_q3', dict(MX5_TEST_DIM=64, MX5_TEST_Q=3), dict(dim=64, query=3))])
        for body in ['chain', 'transform']:
            _, source, keys = self.inputs('--body', body)
            self.assertEqual(str(source), f'test/unit/mx/decode_{body}_v2.cu')
            self.assertEqual(keys, [(body, {}, {})])

    def test_head_bench_legal_boundary_and_prefetch(self):
        args, source, keys = self.inputs('--bench', '--body', 'head', '--tn', '128',
                                         '--depth', '3', '4', '--pw', '4')
        self.assertEqual(str(source), 'experiments/mx5/head_stage_bench.cu')
        self.assertEqual(keys, [('tn128_d3', dict(MX5_TEST_TN=128, MX5_TEST_DEPTH=3), dict(tn=128, depth=3))])
        command = build.compiler_command(args, keys[0][1], source, Path('head.so'))
        self.assertEqual(command.count('-DMX5_BENCH_PW=4'), 1)
        self.assertIn('-DMX5_TEST_DEPTH=3', command)
        args, _, keys = self.inputs('--bench', '--body', 'head', '--pw', '0')
        self.assertEqual(len(keys), 15)
        self.assertEqual(args.pw, 0)

    def test_attention_bench_kinds_are_recorded_and_compiled(self):
        for requested in [None, 0, 1, 2]:
            with self.subTest(requested=requested):
                extra = [] if requested is None else ['--prefetch-kind', str(requested)]
                args, source, keys = self.inputs('--bench', '--body', 'attention', '--dim', '128',
                                                 '--query', '8', *extra)
                expected = 2 if requested is None else requested
                self.assertEqual(str(source), 'experiments/mx5/attention_stage_bench.cu')
                self.assertEqual(keys[0][2], dict(dim=128, query=8, prefetch_kind=expected))
                command = build.compiler_command(args, keys[0][1], source, Path('attention.so'))
                self.assertEqual(command.count(f'-DMX5_BENCH_PREFETCH_KIND={expected}'), 1)
                self.assertFalse(any(arg.startswith('-DMX5_BENCH_PW=') for arg in command))

    def test_invalid_new_bench_coordinates_fail_before_build(self):
        invalid = [
            ['--bench', '--null-idle'],
            ['--body', 'attention', '--null-idle'],
            ['--bench', '--body', 'head', '--tn', '128', '--depth', '4'],
            ['--bench', '--body', 'head', '--tn', '48'],
            ['--bench', '--body', 'head', '--depth', '5'],
            ['--bench', '--body', 'head', '--depth', '3', '--pw', '2'],
            ['--bench', '--body', 'head', '--pw', '-1'],
            ['--bench', '--body', 'attention', '--dim', '96'],
            ['--bench', '--body', 'attention', '--query', '1'],
            ['--bench', '--body', 'attention', '--pw', '1'],
            ['--bench', '--body', 'attention', '--prefetch-kind', '3'],
            ['--body', 'attention', '--prefetch-kind', '1'],
            ['--bench', '--body', 'head', '--prefetch-kind', '1'],
            ['--bench', '--body', 'chain'],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    self.inputs(*arguments)
                self.assertEqual(result.exception.code, 2)


if __name__ == '__main__': unittest.main()
