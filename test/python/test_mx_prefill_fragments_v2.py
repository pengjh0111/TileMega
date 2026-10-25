"""Bitwise 64-token prefill context and physical KV-fragment writer checks."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np


class Args(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in
                ('qkv', 'key_cache', 'value_cache', 'cosine', 'sine', 'q_norm', 'k_norm', 'context')] + [
        (name, C.c_int) for name in ('batch', 'heads_kv', 'capacity', 'past', 'block_extent')] + [
        ('epsilon', C.c_float)]


def fragment_indices(dim, capacity):
    token, d = np.arange(capacity)[:, None], np.arange(dim)[None, :]
    t = token % 32
    lane = 4 * (t // 8) + t % 4 + 16 * ((d % 32) // 8)
    key = ((token // 32 * (dim // 32) * 2 + d // 32 * 2 + t % 8 // 4) * 512
           + lane * 8 + d % 8)
    value = (token // 32 * (dim // 16) + d // 16) * 512 + (d % 16 + 16 * (t // 8)) * 8 + t % 8
    return key.reshape(-1), value.reshape(-1)


class PrefillFragmentsV2(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MX5_PREFILL_FRAGMENTS_CPU_PROOF'), 'requires native CPU proof executable')
    def test_cpu_ownership(self):
        subprocess.run([os.environ['MX5_PREFILL_FRAGMENTS_CPU_PROOF']], check=True)

    def test_fragment_index_bijections(self):
        for dim in (64, 128):
            for capacity in (64, 1088):
                for index in fragment_indices(dim, capacity):
                    np.testing.assert_array_equal(np.sort(index), np.arange(capacity * dim))
                    self.assertTrue(np.all(index.reshape(capacity, dim)[:64] < 64 * dim))
                    self.assertTrue(np.all(index.reshape(capacity, dim)[64:] >= 64 * dim))

    @unittest.skipUnless(os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1' and
                         os.environ.get('MX5_PREFILL_FRAGMENTS_DIR'), 'requires GPU guard and immutable libraries')
    def test_gpu_matrix(self):
        self.assertEqual(os.environ.get('TILEMEGA_GPU_LOCK_HELD'), '1')
        import torch as t

        directory = Path(os.environ['MX5_PREFILL_FRAGMENTS_DIR'])
        stream = C.c_void_p(t.cuda.current_stream().cuda_stream)
        report_path = os.environ.get('MX5_PREFILL_FRAGMENTS_REPORT')
        if report_path:
            self.assertFalse(Path(report_path).exists(), 'choose a fresh report path')
        results = []

        def bits_equal(a, b):
            return t.equal(a.contiguous().view(t.int16), b.contiguous().view(t.int16))

        for dim, norm in ((64, 0), (128, 0), (128, 1)):
            path = directory / f'd{dim}_norm{norm}.so'
            lib = C.CDLL(str(path.resolve()))
            lib.mx5_prefill_fragments_v2.argtypes = [C.POINTER(Args), C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
            lib.mx5_prefill_fragments_v2.restype = C.c_int
            self.assertEqual(lib.mx5_prefill_fragments_v2_args_size(), C.sizeof(Args))
            coordinates = (C.c_int * 4)()
            lib.mx5_prefill_fragments_v2_coordinates(coordinates)
            self.assertEqual(list(coordinates), [dim, norm, 64, 128])
            library_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            capacity = 1088
            ki, vi = [t.from_numpy(index).long().cuda() for index in fragment_indices(dim, capacity)]
            for query in (2, 3, 4, 8):
                for rows in (16, 32, 48, 64):
                    for tile in (32, 64):
                        for batch in (1, 16):
                            for hkv in (1, 4, 8):
                                case = dict(dim=dim, qk_norm=norm, q=query, query_rows=rows, kv_tile=tile,
                                            batch=batch, hkv=hkv, capacity=capacity, past=0, block_extent=capacity)
                                with self.subTest(**case):
                                    generator = t.Generator().manual_seed(501 + dim + 17 * query + rows + batch + hkv)
                                    heads = batch * hkv
                                    raw = (.2 * t.randn(batch, 64, hkv, query + 2, dim, generator=generator)).bfloat16().cuda()
                                    weights = (1 + .1 * t.randn(2, dim, generator=generator)).bfloat16().cuda()
                                    angle = t.randn(capacity, dim // 2, generator=generator).repeat(1, 2)
                                    cosine, sine = angle.cos().bfloat16().cuda(), angle.sin().bfloat16().cuda()
                                    key = (.2 * t.randn(heads, capacity, dim, generator=generator)).bfloat16().cuda()
                                    value = (.2 * t.randn(heads, capacity, dim, generator=generator)).bfloat16().cuda()
                                    old_key, old_value = key.clone(), value.transpose(1, 2).contiguous()
                                    old_context = t.full((batch, 64, hkv * query, dim), float('nan'), dtype=t.bfloat16, device='cuda')

                                    def launch(k, v, context, fragments):
                                        args = Args(raw.data_ptr(), k.data_ptr(), v.data_ptr(), cosine.data_ptr(), sine.data_ptr(),
                                                    weights[0].data_ptr(), weights[1].data_ptr(), context.data_ptr(),
                                                    batch, hkv, capacity, 0, capacity, 1e-6)
                                        self.assertEqual(lib.mx5_prefill_fragments_v2(C.byref(args), query, rows, tile, fragments, stream), 0)
                                        t.cuda.synchronize()

                                    launch(old_key, old_value, old_context, 0)
                                    self.assertTrue(t.isfinite(old_context).all().item(), 'legacy context incomplete')
                                    self.assertTrue(bits_equal(old_key[:, 64:], key[:, 64:]), 'legacy K tail changed')
                                    self.assertTrue(bits_equal(old_value[:, :, 64:], value[:, 64:].transpose(1, 2)), 'legacy V tail changed')
                                    initial_k = t.empty(heads, capacity * dim, dtype=t.bfloat16, device='cuda')
                                    initial_v = t.empty_like(initial_k)
                                    initial_k[:, ki] = key.reshape(heads, -1)
                                    initial_v[:, vi] = value.reshape(heads, -1)
                                    previous = None
                                    for repeat in range(2):
                                        packed_k, packed_v = initial_k.clone(), initial_v.clone()
                                        context = t.full_like(old_context, float('nan'))
                                        launch(packed_k, packed_v, context, 1)
                                        unpacked_k = packed_k[:, ki].reshape(heads, capacity, dim)
                                        unpacked_v = packed_v[:, vi].reshape(heads, capacity, dim)
                                        self.assertTrue(bits_equal(context, old_context), 'old/new context bits differ')
                                        self.assertTrue(bits_equal(unpacked_k[:, :64], old_key[:, :64]), 'K[0:64] bits differ')
                                        self.assertTrue(bits_equal(unpacked_v[:, :64], old_value.transpose(1, 2)[:, :64]), 'V[0:64] bits differ')
                                        self.assertTrue(bits_equal(unpacked_k[:, 64:], key[:, 64:]), 'fragment K tail changed')
                                        self.assertTrue(bits_equal(unpacked_v[:, 64:], value[:, 64:]), 'fragment V tail changed')
                                        if previous is not None:
                                            self.assertTrue(all(bits_equal(a, b) for a, b in zip(previous, (context, packed_k, packed_v))),
                                                            'new writer repeat differs')
                                        previous = context, packed_k, packed_v
                                    results.append(dict(**case, library_sha256=library_hash, status='pass', new_repeats=2))
        self.assertEqual(len(results), 576)
        if report_path:
            Path(report_path).write_text(json.dumps(dict(status='pass', cases=len(results), records=results), indent=2) + '\n')
        print(f'MX5_PREFILL_FRAGMENTS_CASES executed={len(results)} repeats=2', flush=True)


if __name__ == '__main__':
    unittest.main()
