"""Run actual export modules at asymmetric GQA and MoE shapes on meta tensors."""
import copy
import unittest

from tilemega.serving.structure_check import inspect
from tilemega.serving.workload import Workload


class ModelStructures(unittest.TestCase):
    def config(self):
        # Query width 48 differs from hidden 32, with QPerKV=3.
        return dict(architectures=['Qwen3ForCausalLM'], hidden_size=32,
                    intermediate_size=48, num_hidden_layers=2, vocab_size=64,
                    num_attention_heads=6, num_key_value_heads=2, head_dim=8,
                    rms_norm_eps=1e-6, tie_word_embeddings=False)

    def test_asymmetric_query_projection_and_untied_head_at_both_workloads(self):
        report = inspect(self.config(), (Workload(), Workload(128, 2048)))
        self.assertEqual((report['q_width'], report['hidden_size'], report['qperkv']),
                         (48, 32, 3))
        self.assertEqual(report['lm_head_fqn'], 'lm_head.weight')
        self.assertEqual(report['allocated_weight_bytes'], 0)
        self.assertFalse(report['model_gate_passed'])
        self.assertTrue(any(row['batch'] == 16 and row['past'] == 2174
                            and row['cache_shape'] == [16, 2, 2175, 8]
                            for row in report['cases']))

    def test_tied_head_counts_shared_parameters_once(self):
        config = self.config()
        untied = inspect(config, (Workload(3, 2, (1,)),))
        config['tie_word_embeddings'] = True
        tied = inspect(config, (Workload(3, 2, (1,)),))
        self.assertEqual(tied['lm_head_fqn'], 'model.embed_tokens.weight')
        self.assertEqual(untied['logical_bf16_parameter_bytes'] -
                         tied['logical_bf16_parameter_bytes'], 64 * 32 * 2)

    def test_moe_runs_real_expert_and_router_shape_operations(self):
        config = self.config()
        config.update(architectures=['Qwen3MoeForCausalLM'], num_experts=4,
                      num_experts_per_tok=2, moe_intermediate_size=16,
                      norm_topk_prob=True)
        report = inspect(config, (Workload(5, 3, (1, 16)),))
        self.assertEqual(report['layers'], 2)
        self.assertEqual(report['checkpoint_weight_reads'], 0)
        self.assertEqual({row['phase'] for row in report['cases']}, {'prefill', 'decode'})

    def test_osl_one_needs_only_prefill(self):
        report = inspect(self.config(), (Workload(5, 1, (1,)),))
        self.assertEqual([(row['phase'], row['past']) for row in report['cases']],
                         [('prefill', 0)])

    def test_nonintegral_gqa_fails_before_model_creation(self):
        config = copy.deepcopy(self.config())
        config['num_attention_heads'] = 5
        with self.assertRaisesRegex(ValueError, 'QPerKV'):
            inspect(config, (Workload(),))


if __name__ == '__main__':
    unittest.main()
