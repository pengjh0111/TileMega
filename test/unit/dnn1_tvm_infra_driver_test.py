"""Infrastructure rejection is never a BF16 capability result (CPU-only)."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('infra_driver', ROOT/'scripts/dnn1_tvm_infra_driver.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Infrastructure(unittest.TestCase):
    def test_resource_failures_cannot_create_fp16_reference(self):
        arm = dict(system='tvm', precision='bf16', arm='tvm-bf16')
        for code, error in [(75, 'guard'), (-15, 'signal'),
            (1, 'incomplete or failed TVM cache attempt; explicit diagnosis required before retry')]:
            result = dict(final=dict(returncode=code, failure=dict(failure_kind='compile', error=error)))
            with self.subTest(code=code), self.assertRaises(module.HardStop):
                module.fallback_arm(arm, result)
            self.assertEqual(result['final']['failure']['failure_kind'], 'infrastructure')
            self.assertEqual(result['final']['failure']['original_failure']['error'], error)

    def test_genuine_compile_or_numerical_failure_keeps_reference_policy(self):
        arm = dict(system='tvm', precision='bf16', arm='tvm-bf16')
        for kind in ('compile', 'correctness'):
            result = dict(final=dict(returncode=1, failure=dict(failure_kind=kind, error='genuine failure')))
            self.assertEqual(module.fallback_arm(arm, result)['precision'], 'fp16')
        torch = dict(system='torchcompile', fullgraph=True, arm='torchcompile')
        self.assertFalse(module.fallback_arm(torch, dict(final=dict(failure=dict(failure_kind='compile'))))['fullgraph'])

    def test_recovery_changes_only_explicit_cache_directory(self):
        command = ['python', 'baselines.py', '--tvm-work-dir', '/old', '--tvm-max-trials', '2000',
                   '--tvm-timeout-s', '1800', '--warmup', '50', '--iterations', '200']
        with patch.object(module, '_command', return_value=command.copy()):
            result = module.arm_command(dict(infrastructure_cache_work_dir='/new/attempt2'),
                dict(system='tvm', precision='bf16'), 'measurement', 0, Path('/unused'))
        expected = command.copy();expected[3] = '/new/attempt2/bf16'
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
