"""Known unsound causal-prefill binaries cannot enter performance reports."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('r16_validation_prefix', ROOT/'scripts/r16_measurement_validation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tile_m=16, mapping='kWindow', kind='kGemm', scale=3, count=3):
    return f'''constexpr StageDesc kStages[] = {{
  {{TaskKind::{kind}, 0u, 0u, 0u, 1u, {{0u}}}},
  {{TaskKind::kFusedAttention, 0u, 0u, 0u, 1u, {{8u}}}},
}};
constexpr GemmDesc kGemms[] = {{
  {{3072, 2048, 5u, 249u, 8u, 8u, 0.0f}}
}};
constexpr GemmRuntimeDesc kRuntimeGemms0[] = {{
  {{0u, 1u, {tile_m}u, 128u, 64u, 2u}}
}};
constexpr StageDependency kDependencies0[] = {{
  {{0u, 1u, StageDependency::Map::{mapping}, 1u, {scale}, 0, {count}u, 0u}}
}};
'''


class PrefixEligibility(unittest.TestCase):
    def test_current_tile_dependencies_rejected_for_gemm_and_combine(self):
        for kind in ('kGemm', 'kGemmCombine'):
            issues = module.prefill_prefix_dependency_issues({'plan': {'phase': 'prefill', 'seq': 64}}, source(kind=kind))
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0]['tile_m'], 16)

    def test_all_stage_and_whole_sample_tile_not_rejected(self):
        identity = {'plan': {'phase': 'prefill', 'seq': 64}}
        self.assertFalse(module.prefill_prefix_dependency_issues(identity, source(mapping='kAll')))
        self.assertFalse(module.prefill_prefix_dependency_issues(identity, source(tile_m=64)))

    def test_decode_and_encoder_forward_outside_causal_prefill_scope(self):
        for phase, seq in (('decode', 1), ('forward', 128)):
            self.assertFalse(module.prefill_prefix_dependency_issues({'plan': {'phase': phase, 'seq': seq}}, source()))

    def test_wide_window_not_confused_with_observed_current_tile_map(self):
        self.assertFalse(module.prefill_prefix_dependency_issues({'plan': {'phase': 'prefill', 'seq': 64}}, source(count=192)))

    def test_unrelated_gemm_dependency_is_not_qkv(self):
        text = source().replace('8u, 8u, 0.0f', '8u, 9u, 0.0f')
        self.assertFalse(module.prefill_prefix_dependency_issues({'plan': {'phase': 'prefill', 'seq': 64}}, text))


if __name__ == '__main__':
    unittest.main()
