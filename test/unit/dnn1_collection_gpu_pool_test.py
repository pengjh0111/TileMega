"""CPU-only admission configuration checks; no scheduler is launched."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import dnn1_aitemplate_collect as ait
import dnn1_tvm_resume_collect as tvm


class CollectionPoolTest(unittest.TestCase):
    def test_explicit_all_authorized_cards_preserve_order(self):
        values = [2, 3, 4, 5, 7, 0, 1]
        self.assertEqual(tvm.gpu_pool(','.join(map(str, values))), values)
        self.assertEqual(ait.gpu_pool({'gpus': values}), values)

    def test_gpu6_is_rejected_even_in_an_otherwise_allowed_pool(self):
        with self.assertRaises(ValueError):
            tvm.gpu_pool('0,1,2,3,4,5,6,7')
        with self.assertRaises(ValueError):
            ait.gpu_pool({'gpus': [0, 1, 6]})

    def test_duplicate_or_empty_pools_cannot_dispatch_twice_per_card(self):
        for values in ([], [1, 1]):
            with self.assertRaises(ValueError):
                tvm.gpu_pool(','.join(map(str, values)))
            with self.assertRaises(ValueError):
                ait.gpu_pool({'gpus': values})

    def test_old_ait_epochs_keep_their_original_pool(self):
        self.assertEqual(ait.gpu_pool({}), [5, 7, 0, 1])


if __name__ == '__main__':
    unittest.main()
