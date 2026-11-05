"""CPU-only target contract: add driver linking without changing CUDA math."""
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate as runner


class TargetTest(unittest.TestCase):
    def target(self):
        class CUDA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def compile_cmd(self, executable=False):
                return 'nvcc -o {target} {src}' if executable else 'nvcc -c -o {target} {src}'
            def compile_options(self):
                return '--other-shared-options'
            def _build_nvcc_compiler_options(self):
                return ['-gencode=arch=compute_90,code=[sm_90,compute_90]', '--keep-other-options']
        fake = ModuleType('aitemplate.backend.cuda.target_def')
        fake.CUDA = CUDA
        with patch.dict(sys.modules, {'aitemplate.backend.cuda.target_def': fake}):
            return runner.target()

    def test_only_profiler_executable_links_driver_after_sources(self):
        target = self.target()
        self.assertEqual(target.compile_cmd(True), 'nvcc -o {target} {src} -lcuda')
        self.assertEqual(target.compile_cmd(False), 'nvcc -c -o {target} {src}')

    def test_bf16_accumulation_and_original_target_options_stay_fixed(self):
        target = self.target()
        self.assertFalse(target.kwargs['use_fp16_acc'])
        self.assertTrue(target.kwargs['allow_cutlass_sm90'])
        self.assertFalse(target.kwargs['force_cutlass_sm90'])
        self.assertEqual(target._build_nvcc_compiler_options(),
            ['-gencode=arch=compute_90a,code=[sm_90a,compute_90a]', '--keep-other-options'])
        self.assertEqual(runner.RECIPE['precision'], 'bf16')
        self.assertTrue(runner.RECIPE['graph_mode'])

    def test_cli_recipe_identity_cannot_revert_to_pre_linkfix_schema(self):
        with patch.dict(runner.RECIPE):
            runner.configure_graph_recipe(False)
            self.assertEqual(runner.RECIPE['schema'], '5-no-graph-reference')
            runner.configure_graph_recipe(True)
            self.assertEqual(runner.RECIPE['schema'], 5)
            self.assertTrue(runner.RECIPE['graph_mode'])

    def test_shared_link_retains_driver_before_object_files(self):
        self.assertEqual(self.target().compile_options(),
            '--other-shared-options -Xlinker=--no-as-needed -lcuda -Xlinker=--as-needed')


if __name__ == '__main__':
    unittest.main()
