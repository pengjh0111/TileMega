"""The MPK KV page budget must match MPK's own page allocator, not its demos.

persistent_kernel.cuh draws ceil((step + num_new_tokens)/block_size) pages per
request from one queue of MPK_MAX_NUM_PAGES, summed over the batch. The demos'
CLI default (16 pages of 4096) is sized for one long-context request and
over-allocates KV by 3.8x per page at this protocol's 64 + 1024 tokens.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "python/tilemega/serving/mpk_baseline.py"


def load():
    # mpk_baseline imports baseline_common as a top-level module, as the
    # wrapper does when run from its own directory.
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("independent_mpk_baseline", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MpkKvBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpk = load()

    def test_pages_cover_every_request(self):
        pages = self.mpk.kv_pages
        # One page holds the whole 1088-token sequence, so the pool is B pages.
        for batch in (1, 2, 4, 8, 16):
            self.assertEqual(pages(batch, 64, 1024, 4096), batch)
        # A page smaller than the sequence needs the ceiling, per request.
        self.assertEqual(pages(1, 64, 1024, 128), 9)
        self.assertEqual(pages(16, 64, 1024, 128), 144)
        self.assertEqual(pages(2, 64, 1024, 1088), 2)
        self.assertEqual(pages(2, 64, 1024, 1087), 4)

    def test_invalid_budgets_are_rejected(self):
        for args in ((0, 64, 1024, 4096), (1, 0, 1024, 4096),
                     (1, 64, 0, 4096), (1, 64, 1024, 0)):
            with self.assertRaises(ValueError):
                self.mpk.kv_pages(*args)

    def test_demo_default_over_allocates(self):
        # What the R4 runs paid: 16 pages x 4096 regardless of batch.
        self.assertEqual(self.mpk.kv_pages(1, 64, 1024, 4096) * 4096, 4096)
        self.assertLess(self.mpk.kv_pages(1, 64, 1024, 4096), 16)


if __name__ == "__main__":
    unittest.main()
