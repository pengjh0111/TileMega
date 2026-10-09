"""Compare packed BF16 weights to independent host CuTe layout evaluation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from tilemega.serving.weights import _packed_gpu


class WeightPagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.temporary = tempfile.TemporaryDirectory()
        binary = Path(cls.temporary.name) / 'cute_layout'
        cuda = Path(os.environ.get('CUDA_PATH', '/usr/local/cuda'))
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                        '-I' + str(root / 'third_party/cutlass/include'),
                        '-I' + str(cuda / 'include'),
                        str(root / 'test/unit/dm_weight_page_oracle.cpp'),
                        '-o', str(binary)], check=True)
        words = np.frombuffer(subprocess.check_output([str(binary)]), dtype=np.uint32)
        cls.layouts = {}
        at = 0
        while at < words.size:
            n, k = map(int, words[at:at + 2])
            cls.layouts[n, k] = words[at + 2:at + 2 + n * k].copy()
            at += 2 + n * k
        if at != words.size or len(cls.layouts) != 24:
            raise AssertionError('incomplete CuTe oracle')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def pack(self, weight, tn, tk, nested=None):
        return _packed_gpu(dict(kind='tile_pages', tile_n=tn, tile_k=tk,
                                source=nested or dict(kind='alias', source='w')),
                           lambda name: weight)

    def expected(self, weight, tn, tk):
        n, k = weight.shape
        nt, kt = (n + tn - 1) // tn, (k + tk - 1) // tk
        result = torch.zeros(nt * kt * tn * tk, dtype=torch.bfloat16)
        offsets = self.layouts[tn, tk].reshape(tn, tk)
        for row in range(n):
            for col in range(k):
                tile = (row // tn) * kt + col // tk
                result[tile * tn * tk + int(offsets[row % tn, col % tk])] = weight[row, col]
        return result

    def test_cute_swizzle_bijections(self):
        for (n, k), offsets in self.layouts.items():
            with self.subTest(n=n, k=k):
                np.testing.assert_array_equal(np.sort(offsets), np.arange(n * k))

    def test_every_width_and_k_tail(self):
        for tn, tk in self.layouts:
            for n, k in [(3, 7), (tn - 1, tk - 1), (tn + 3, tk + 5)]:
                with self.subTest(tn=tn, tk=tk, n=n, k=k):
                    weight = (torch.arange(n * k).reshape(n, k) % 257).to(torch.bfloat16)
                    before = weight.clone()
                    torch.testing.assert_close(self.pack(weight, tn, tk),
                                               self.expected(weight, tn, tk), rtol=0, atol=0)
                    self.assertTrue(torch.equal(weight, before))

    def test_legacy_packing_stays_bitwise_equal(self):
        for tn in (8, 16, 32, 64, 128, 256):
            for tk in (64, 128):
                n, k = tn + 1, tk + 7
                weight = (torch.arange(n * k).reshape(n, k) % 257).to(torch.bfloat16)
                nt, kt = (n + tn - 1) // tn, (k + tk - 1) // tk
                padded = torch.zeros((nt * tn, kt * tk), dtype=torch.bfloat16)
                padded[:n, :k] = weight
                logical = padded.reshape(nt, tn, kt, tk).permute(0, 2, 1, 3)
                index = torch.arange(tn * tk)
                row, col = index // tk, index % tk
                perm = (512 * (row // 8 + (tn // 8) * (col // 64)) + 64 * (row % 8) +
                        8 * ((col % 64 // 8) ^ (row % 8)) + col % 8)
                prior = torch.empty((nt, kt, tn * tk), dtype=torch.bfloat16)
                prior[..., perm] = logical.reshape(nt, kt, tn * tk)
                self.assertTrue(torch.equal(self.pack(weight, tn, tk), prior.reshape(-1)))

    def test_nested_gate_pairs(self):
        gate = torch.arange(32 * 24).reshape(32, 24).to(torch.bfloat16)
        up = -gate
        nested = dict(kind='gate_up_interleave', sources=['gate', 'up'], u=16)
        values = dict(gate=gate, up=up)
        interleaved = torch.cat([gate[:16], up[:16], gate[16:], up[16:]])
        for tk in (16, 32):
            packed = _packed_gpu(dict(kind='tile_pages', tile_n=32, tile_k=tk,
                                      source=nested), values.__getitem__)
            torch.testing.assert_close(packed, self.expected(interleaved, 32, tk), rtol=0, atol=0)

    def test_invalid_geometry_rejects(self):
        for tn, tk in [(0, 16), (7, 16), (16, 0), (16, 8), (16, 24), (16, 48), (16, 96)]:
            with self.subTest(tn=tn, tk=tk), self.assertRaises(ValueError):
                self.pack(torch.zeros((3, 5), dtype=torch.bfloat16), tn, tk)


if __name__ == '__main__':
    unittest.main()
