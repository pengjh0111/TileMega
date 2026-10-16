"""User card authorization applies independently to functional and timed jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from tilemega.bench.gpu_guard import require_allowed_gpu, gpu_step_allowed

class AllowlistTests(unittest.TestCase):
    def test_functional_cards_cannot_run_timing(self):
        policy = dict(allowed_gpu_indices=[5, 6, 7], allowed_timing_gpu_indices=[7])
        for index in (5, 6):
            require_allowed_gpu(policy, index)
            self.assertTrue(gpu_step_allowed(policy, index, dict(gpu=True, timing=False)))
            self.assertFalse(gpu_step_allowed(policy, index, dict(gpu=True, timing=True)))
            with self.assertRaises(ValueError):
                require_allowed_gpu(policy, index, timing=True)
        require_allowed_gpu(policy, 7, timing=True)
        for index in range(5):
            with self.assertRaises(ValueError):
                require_allowed_gpu(policy, index)

    def test_live_file_tightens_cached_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'constraints.json'
            path.write_text(json.dumps(dict(allowed_gpu_indices=[5, 6, 7],
                                            allowed_timing_gpu_indices=[7])))
            policy = dict(allowed_gpu_indices=[5, 6, 7], gpu_allowlist_file=str(path))
            self.assertFalse(gpu_step_allowed(policy, 6, dict(gpu=True)))
            self.assertTrue(gpu_step_allowed(policy, 6, dict(gpu=True, timing=False)))
            path.write_text(json.dumps(dict(allowed_gpu_indices=[7])))
            self.assertFalse(gpu_step_allowed(policy, 6, dict(gpu=True, timing=False)))

    def test_timing_only_constraint_and_malformed_list(self):
        with self.assertRaises(ValueError):
            require_allowed_gpu(dict(allowed_timing_gpu_indices=[7]), 5, timing=True)
        with self.assertRaises(ValueError):
            require_allowed_gpu(dict(allowed_timing_gpu_indices=['7']), 7)
        with self.assertRaises(ValueError):
            require_allowed_gpu(dict(allowed_timing_gpu_indices=[]), 7, timing=True)

if __name__ == '__main__':
    unittest.main()
