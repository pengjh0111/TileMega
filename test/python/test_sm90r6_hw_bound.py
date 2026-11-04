"""Independent small-model arithmetic and incomplete-evidence rejection for H3."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import sm90r6_hw_bound as h


class HardwareBounds(unittest.TestCase):
    def test_dense_arithmetic(self):
        config = dict(hidden_size=4, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=2, intermediate_size=6, vocab_size=8)
        value = h.payload(config, 3)
        self.assertEqual(value['weight_bytes'], 544)
        self.assertEqual(value['kv_bytes'], 27648)
        self.assertEqual(value['flops_step'], 56928)
        with self.assertRaises(ValueError):
            h.payload(config, 3, tp=2)

    def test_moe_distinct_reads_and_active_compute(self):
        config = dict(hidden_size=4, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=2, intermediate_size=6,
                      vocab_size=8, num_experts=4, num_experts_per_tok=2)
        # Dense part: 2*(32+16+16)+32 = 160 parameters. Each expert has 72.
        value = h.payload(config, 3, [{'distinct_mean': 2}, {'distinct_mean': 3}])
        self.assertEqual(value['weight_bytes'], 2*160+2*72*5)
        self.assertEqual(value['flops_step'], 2*3*(160+2*2*72)+55296)
        with self.assertRaises(ValueError):
            h.payload(config, 3)

    def test_roofline_and_tp_latency(self):
        values = dict(bytes_step=1e9, flops_step=2e12, layers=2)
        self.assertEqual(h.bound(values, 1000, 100, 1, 608)['bound_ms'], 20)
        self.assertAlmostEqual(h.bound(values, 1000, 100, 2, 608)['bound_ms'], 1.00304)

    def test_h1_requires_every_supported_configuration(self):
        rows = [dict(gpu=g, item=item, config=str(c), processes=5, failed_processes=0,
                     median=100, throttle_reasons='0x0;0x1')
                for g in (0, 1, 2, 3, 4, 5, 7)
                for item, count in [('hbm_read', 2), ('hbm_read_bulk', 4),
                                    ('bf16_wgmma', 1), ('bf16_mma_sync', 1)]
                for c in range(count)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'peaks.csv'
            def save():
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader(); writer.writerows(rows)
            save(); self.assertEqual(h.peaks(path), (100, 100))
            rows.pop(); save()
            with self.assertRaises(ValueError):
                h.peaks(path)


if __name__ == '__main__':
    unittest.main()
