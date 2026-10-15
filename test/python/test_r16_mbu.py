# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mbu', Path(__file__).resolve().parents[2] / 'scripts/estimate_r16_mbu.py')
mbu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbu)


class PayloadTests(unittest.TestCase):
    def test_explicit_query_width_and_tied_lm_head(self):
        config = dict(hidden_size=8, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=1, head_dim=4, intermediate_size=12,
                      vocab_size=20, tie_word_embeddings=True)
        result = mbu.dense_payload(config, 3, isl=4, osl=8)
        # Independently enumerate Q, K, V, O, gate, up, down and lm_head.
        matrices = [(8, 16), (8, 4), (8, 4), (16, 8), (8, 12), (8, 12), (12, 8)]
        self.assertEqual(result['gemm_weight_bytes'], 2 * (2 * sum(m*n for m,n in matrices) + 8*20))
        self.assertEqual(result['average_kv_rows'], sum(range(5, 12)) / 7)
        self.assertEqual(result['kv_read_bytes'], 2 * 2 * 3 * 2 * 1 * 4 * 8)

    def test_unknown_moe_routing_is_not_dense_estimate(self):
        self.assertIsNone(mbu.dense_payload({'num_experts': 128}, 16))

    def test_bandwidth_rejects_failed_or_nonfinite_receipt(self):
        row = dict(name='bulk', unit='GB/s', passed=True, medians=[3000, 3100], gpus=['0', '2'])
        self.assertEqual(mbu.bandwidth(dict(passed=True, rows=[row])), (3050, '0;2'))
        with self.assertRaises(ValueError):
            mbu.bandwidth(dict(passed=False, rows=[row]))
        with self.assertRaises(ValueError):
            mbu.bandwidth(dict(passed=True, rows=[dict(row, medians=[float('nan'), 3100])]))


if __name__ == '__main__':
    unittest.main()
