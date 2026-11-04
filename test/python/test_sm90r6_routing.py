"""CPU checks for route validation and the exact prompt/decode launch boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('routing', ROOT / 'scripts/sm90r6_moe_routing.py')
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


class RoutingTest(unittest.TestCase):
    def test_unique_across_batch(self):
        self.assertEqual(routing.distinct_experts([[[1, 2], [2, 3]]], 2, 4, 2), [3])
        for indices in [[[0, 0]], [[-1, 2]], [[0, 4]], [[0]], []]:
            with self.assertRaises(ValueError):
                routing.distinct_experts([indices], 1, 4, 2)

    def test_exact_generation_boundary(self):
        tokens = torch.empty((1, 8), dtype=torch.int32)
        launched, captured = [], []
        def launch(position, mode, stream):
            launched.append(position)
            if position >= 2:
                tokens[0, position + 1] = position
        engine = SimpleNamespace(prefill=None, decode_loop=False, prompt_len=3,
            decode_mode=1, decode=SimpleNamespace(launch=launch),
            state=SimpleNamespace(tokens=tokens, kv_storage=torch.ones(4)))
        stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
        with patch.object(torch.cuda, 'current_stream', return_value=stream):
            result = routing.generation_routes(engine, torch.tensor([[7, 6, 5]]), 4, 10,
                lambda plan, step, position: captured.append((step, position)))
        self.assertEqual(launched, list(range(6)))
        self.assertEqual(captured, [(0, 2), (1, 3), (2, 4), (3, 5)])
        self.assertEqual(result, [[2, 3, 4, 5]])
        self.assertEqual(tokens[0, :3].tolist(), [7, 6, 5])
        self.assertTrue(torch.all(engine.state.kv_storage == 0))

    def test_model_expert_sizes(self):
        self.assertEqual(routing.expert_bytes(dict(hidden_size=8, intermediate_size=24,
                                                   moe_intermediate_size=4)), 192)
        self.assertEqual(routing.expert_bytes(dict(hidden_size=8, intermediate_size=4)), 192)


if __name__ == '__main__':
    unittest.main()
