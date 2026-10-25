"""MX-5 body numerics and determinism. Run under mx_gpu_job.sh."""
import ctypes as C
import os
from pathlib import Path
import unittest


class Args(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('a', 'w', 'residual', 'out', 'partial', 'tickets', 'next_calls')] + [
        (name, C.c_int) for name in ('n', 'k', 'b')] + [('epsilon', C.c_float)]


@unittest.skipUnless(os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1' and os.environ.get('MX5_BODIES_DIR'),
                     'requires GPU guard and MX5_BODIES_DIR')
class Gemm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tilemega.serving.weights import _packed_gpu
        cls.torch = torch
        cls.pack = staticmethod(_packed_gpu)
        cls.directory = Path(os.environ['MX5_BODIES_DIR'])

    def run_case(self, lib, tn, depth, wn, op, norm, part, b, pw=None):
        t = self.torch
        pw = depth if pw is None else pw
        n = tn * (40 if not part else 104 * wn + 37)
        k = 256 if pw <= depth else 2048
        if pw == 16: k = 8192
        generator = t.Generator().manual_seed(991 + tn + b + wn + depth)
        x = (t.randn(b, k, generator=generator) * .1).bfloat16().cuda()
        w = (t.randn(n, k, generator=generator) * .1).bfloat16().cuda()
        residual = (t.randn(b, n, generator=generator) * .1).bfloat16().cuda()
        packed = self.pack(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                                source=dict(kind='alias', source='w')), lambda _: w).contiguous()
        contributors = (104 + n // tn - 1) // (n // tn) + 1
        partial = t.empty(n // tn * contributors * b * (tn + 1), device='cuda')
        tickets = t.zeros(n // tn, dtype=t.int32, device='cuda')
        calls = t.zeros(104 * 8, dtype=t.int32, device='cuda')
        def ptr(v): return v.data_ptr()
        outputs = []
        for repeat in range(2):
            output = t.full((b, n // 2 if op == 2 else n), float('nan'), dtype=t.bfloat16, device='cuda')
            calls.zero_()
            args = Args(ptr(x), ptr(packed), ptr(residual), ptr(output), ptr(partial), ptr(tickets),
                        ptr(calls), n, k, b, 1e-6)
            rc = lib.mx5_gemm(C.byref(args), wn, op, int(norm), part, pw,
                              C.c_void_p(t.cuda.current_stream().cuda_stream))
            self.assertEqual(rc, 0, (tn, depth, wn, op, norm, part, b, pw))
            t.cuda.synchronize()
            self.assertEqual(int(tickets.abs().sum()), 0)
            self.assertTrue(bool((calls == 1).all()), 'next must execute once per compute wave')
            self.assertFalse(bool(output.isnan().any()))
            outputs.append(output)
        self.assertTrue(t.equal(outputs[0].view(t.int16), outputs[1].view(t.int16)),
                        'same input must be bitwise deterministic')
        acc = x.float() @ w.float().T
        if norm: acc *= t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        rounded = acc.bfloat16()
        if op == 1: want = (rounded.float() + residual.float()).bfloat16()
        elif op == 2:
            pair = rounded.reshape(b, -1, 2, 16)
            gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
            want = ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(b, -1)
        else: want = rounded
        error = (outputs[0].float() - want.float()).abs()
        self.assertTrue(bool((error <= want.float().abs() * 2**-6 + .02).all()),
                        f'{tn,depth,wn,op,norm,part,b,pw}: max error={float(error.max())}')

    def test_matrix(self):
        paths = sorted(self.directory.glob('tn*_d*.so'))
        self.assertTrue(paths, 'no body libraries')
        count = 0
        for path in paths:
            tn, depth = map(int, path.stem.replace('tn', '').replace('_d', ' ').split())
            lib = C.CDLL(str(path.resolve()))
            self.assertEqual(lib.mx5_gemm_args_size(), C.sizeof(Args))
            for wn in (1, 2, 4, 8):
                if os.environ.get('MX5_STREAM_ONLY') and wn != 1: continue
                for op in (0, 1, 2):
                    if op == 2 and tn < 32: continue
                    for norm in (False, True):
                        for part in ((0, 1) if wn == 1 else (1,)):
                            if os.environ.get('MX5_STREAM_ONLY') and part != 0: continue
                            for b in (1, 2, 4, 8, 16):
                                with self.subTest(tn=tn, depth=depth, wn=wn, op=op, norm=norm, part=part, b=b):
                                    self.run_case(lib, tn, depth, wn, op, norm, part, b)
                                    count += 1
        print(f'MX5_GEMM_CASES executed={count} libraries={len(paths)}', flush=True)

    def test_derived_widths(self):
        path = self.directory / 'tn16_d1.so'
        if not path.exists(): self.skipTest('requires TN16 D1 library')
        lib = C.CDLL(str(path.resolve()))
        for pw in (0, 8, 16):
            for op in (0, 1):
                for norm in (False, True):
                    for b in (1, 16):
                        with self.subTest(pw=pw, op=op, norm=norm, b=b):
                            self.run_case(lib, 16, 1, 1, op, norm, 1, b, pw)


if __name__ == '__main__':
    unittest.main()
