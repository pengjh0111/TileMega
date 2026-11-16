"""Exercise conservation and Decode-window selection on CPU fixture routes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import moe_profile_instrumentation as hooks


class RoutingTest(unittest.TestCase):
    def test_decode_window_and_assignment_conservation(self):
        import torch
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 9, 10, 11, 12, 13, 14, 15]])
        hooks.arm()
        for _ in range(10):
            hooks.begin_layer(0, 2, 'prefill_or_unknown', {})
            hooks.record_ids(ids)
        self.assertEqual(hooks.STATE['routes'], [])
        for _ in range(60):
            hooks.begin_layer(0, 2, 'decode', {})
            hooks.record_ids(ids)
        with tempfile.TemporaryDirectory() as directory:
            summary = hooks.export(directory)
            rows = json.loads((Path(directory) / 'routing.json').read_text())['route_samples']
        self.assertEqual(summary['samples'], 17)
        self.assertEqual([row['decode_iteration'] for row in rows], list(range(32, 49)))
        for row in rows:
            self.assertEqual(sum(row['n_e']), 16)
            self.assertEqual(row['active_experts'], 15)
            self.assertEqual(row['empty_experts'], 113)
            self.assertEqual(row['max_n_e'], 2)


if __name__ == '__main__':
    unittest.main()
