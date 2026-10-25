"""Block-layout attention numerics, cache ownership and repeatability."""
import ctypes as C
import os
from pathlib import Path
import unittest

import numpy as np


class Ops(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['qkv', 'q_norm', 'k_norm', 'cosine', 'sine',
                                              'k_cache', 'v_cache', 'ctx', 'partial', 'tickets']] + [
        ('capacity', C.c_int), ('past', C.c_int), ('epsilon', C.c_float)]


class Args(C.Structure):
    _fields_ = [('ops', Ops), ('calls', C.c_void_p), ('b', C.c_int), ('hkv', C.c_int)]


class TransformArgs(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['input', 'weight', 'cosine', 'sine', 'output']] + [
        ('rows', C.c_int), ('epsilon', C.c_float)]


def layout_indices(dim, cap):
    token = np.arange(cap)[:, None]
    d = np.arange(dim)[None, :]
    t = token % 32
    lane = 4 * (t // 8) + t % 4 + 16 * ((d % 32) // 8)
    k = ((token // 32 * (dim // 32) * 2 + d // 32 * 2 + t % 8 // 4) * 512
         + lane * 8 + d % 8)
    v = (token // 32 * (dim // 16) + d // 16) * 512 + (d % 16 + 16 * (t // 8)) * 8 + t % 8
    return k, v


@unittest.skipUnless(os.environ.get('MX5_ATTN_DIR') and os.environ.get('MX5_TRANSFORM_SO') and
                     os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1', 'requires GPU guard and body libraries')
class AttentionV2(unittest.TestCase):
    def test_matrix(self):
        import torch as t
        transform = C.CDLL(os.environ['MX5_TRANSFORM_SO'])
        stream = C.c_void_p(t.cuda.current_stream().cuda_stream)
        count = 0
        for path in sorted(Path(os.environ['MX5_ATTN_DIR']).glob('d*_q*.so')):
            dim, query = map(int, path.stem.replace('d', '').replace('_q', ' ').split())
            lib = C.CDLL(str(path.resolve()))
            cap, hkv = 1088, 4
            ki, vi = [t.from_numpy(a.reshape(-1)).long().cuda() for a in layout_indices(dim, cap)]
            for norm in ([False, True] if dim == 128 else [False]):
                for b in [1, 16]:
                    for length in [1, 31, 32, 33, 65, 257, 1088]:
                        for bmin in [1, 8]:
                            for kind in [0, 1, 2]:
                                with self.subTest(d=dim, q=query, norm=norm, b=b, length=length,
                                                  bmin=bmin, kind=kind):
                                    heads = b * hkv
                                    g = t.Generator().manual_seed(dim + b + length)
                                    raw = (t.randn(heads, query + 2, dim, generator=g) * .2).bfloat16().cuda()
                                    key = (t.randn(heads, cap, dim, generator=g) * .2).bfloat16().cuda()
                                    value = (t.randn(heads, cap, dim, generator=g) * .2).bfloat16().cuda()
                                    weights = (1 + .1 * t.randn(2, dim, generator=g)).bfloat16().cuda()
                                    angle = t.linspace(-2., 2., dim // 2).repeat(2)
                                    cosine = angle.cos().repeat(cap, 1).bfloat16().cuda()
                                    sine = angle.sin().repeat(cap, 1).bfloat16().cuda()

                                    def row_transform(x, w):
                                        x = x.contiguous(); out = t.empty_like(x)
                                        a = TransformArgs(x.data_ptr(), w.data_ptr(),
                                                          cosine[length - 1].data_ptr(), sine[length - 1].data_ptr(),
                                                          out.data_ptr(), x.numel() // dim, 1e-6)
                                        self.assertEqual(transform.mx5_transform(C.byref(a), dim, int(norm), 0, stream), 0)
                                        return out

                                    q = row_transform(raw[:, :query], weights[0])
                                    new_key = row_transform(raw[:, query], weights[1])
                                    expected_key = key.clone(); expected_key[:, length - 1] = new_key
                                    expected_value = value.clone(); expected_value[:, length - 1] = raw[:, query + 1]
                                    score = t.einsum('hqd,htd->hqt', q.float(), expected_key[:, :length].float()) / dim**.5
                                    expected = t.einsum('hqt,htd->hqd', score.softmax(-1),
                                                        expected_value[:, :length].float()).bfloat16()
                                    results = []
                                    for repeat in range(2):
                                        packed_k = t.empty(heads, cap * dim, dtype=t.bfloat16, device='cuda')
                                        packed_v = t.empty_like(packed_k)
                                        packed_k[:, ki] = key.reshape(heads, -1)
                                        packed_v[:, vi] = value.reshape(heads, -1)
                                        output = t.full((heads, query, dim), float('nan'), dtype=t.bfloat16, device='cuda')
                                        partial = t.empty(heads * 34 * query * (dim + 2), device='cuda')
                                        tickets = t.zeros(heads, dtype=t.int32, device='cuda')
                                        calls = t.zeros(104 * 8, dtype=t.int32, device='cuda')
                                        ops = Ops(raw.data_ptr(), weights[0].data_ptr(), weights[1].data_ptr(),
                                                  cosine.data_ptr(), sine.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
                                                  output.data_ptr(), partial.data_ptr(), tickets.data_ptr(), cap, length - 1, 1e-6)
                                        a = Args(ops, calls.data_ptr(), b, hkv)
                                        self.assertEqual(lib.mx5_attention(C.byref(a), bmin, int(norm), kind, stream), 0)
                                        t.cuda.synchronize()
                                        self.assertTrue(t.all(calls == 1).item())
                                        self.assertEqual(tickets.abs().sum().item(), 0)
                                        self.assertTrue(t.equal(packed_k[:, ki].view(t.int16),
                                                                expected_key.reshape(heads, -1).view(t.int16)), 'K bits')
                                        self.assertTrue(t.equal(packed_v[:, vi].view(t.int16),
                                                                expected_value.reshape(heads, -1).view(t.int16)), 'V bits')
                                        error = (output.float() - expected.float()).abs()
                                        self.assertTrue(t.all(error <= .006 + .016 * expected.float().abs()).item(),
                                                        f'context max error {error.max().item()}')
                                        results.append(output)
                                    self.assertTrue(t.equal(results[0].view(t.int16), results[1].view(t.int16)))
                                    count += 1
        self.assertGreater(count, 0)
        print(f'MX5_ATTN_CASES executed={count}', flush=True)


if __name__ == '__main__': unittest.main()
