"""Noise supplements are limited to competitive keys in the same scope."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[2] / 'experiments/mx7/qualify_costs.py'
spec = importlib.util.spec_from_file_location('mx7_cost_qualification', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(stage, cost, spread, batch=1, length=None):
    return dict(key=dict(model='qwen3_1_7b_tp2', b=batch, stage=stage, length=length),
                median_ns=cost, spread=spread)


class CostNoiseTests(unittest.TestCase):
    def test_only_competitive_noisy_keys_are_supplemented(self):
        rows = [row('o', 100, .01), row('o', 101, .03), row('o', 103, .04),
                row('down', 500, .03), row('o', 900, .03, batch=16)]
        self.assertEqual(module.competitive_noise(rows), [rows[1], rows[3], rows[4]])

    def test_attention_lengths_have_separate_noise_comparisons(self):
        rows = [row('attention', 10, .01, length=65),
                row('attention', 100, .03, length=1088),
                row('attention', 104, .05, length=1088)]
        self.assertEqual(module.competitive_noise(rows), [rows[1]])


if __name__ == '__main__':
    unittest.main()
