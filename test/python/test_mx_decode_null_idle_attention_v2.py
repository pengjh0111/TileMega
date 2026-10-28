"""Exercise idle first-segment prefetch with null K/V and unchanged Run inputs."""
from pathlib import Path
import importlib.util
import os
import unittest

_path = Path(__file__).with_name('test_mx_decode_attention_v2.py')
_spec = importlib.util.spec_from_file_location('_mx5_attention_oracle', _path)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


class NullIdleAttention(_oracle.AttentionV2):
    def prefetch_kinds(self, dim, query, norm, b, length, bmin):
        self.assertEqual((dim, query, norm), (64, 4, False))
        return (2,)

    def test_matrix(self):
        directory = Path(os.environ['MX5_ATTN_DIR'])
        self.assertEqual({p.name for p in directory.glob('d*_q*.so')}, {'d64_q4.so'})
        super().test_matrix()


if __name__ == '__main__':
    unittest.main()
