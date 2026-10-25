"""Compare the v2 fragment q/k transform bitwise with MX-4 RopeEpilogue."""
import ctypes as C
import os
import unittest


class Args(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('input', 'weight', 'cosine', 'sine', 'output')] + [
        ('rows', C.c_int), ('epsilon', C.c_float)]


@unittest.skipUnless(os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1' and os.environ.get('MX5_TRANSFORM_SO'),
                     'requires GPU guard and MX5_TRANSFORM_SO')
class Transform(unittest.TestCase):
    def test_bitwise(self):
        import torch as t
        lib = C.CDLL(os.environ['MX5_TRANSFORM_SO'])
        for dim, norm in [(64, False), (128, False), (128, True)]:
            for b in [1, 2, 4, 8, 16]:
                with self.subTest(dim=dim, norm=norm, b=b):
                    generator = t.Generator().manual_seed(dim + b)
                    x = t.randn(b, dim, generator=generator).bfloat16().cuda()
                    if b > 1: x[0].zero_()
                    w = (1 + .1 * t.randn(dim, generator=generator)).bfloat16().cuda()
                    angle = t.linspace(-3.14, 3.14, dim)
                    cosine, sine = angle.cos().bfloat16().cuda(), angle.sin().bfloat16().cuda()
                    actual = t.full_like(x, float('nan'))
                    reference = t.full((b, 3, dim), float('nan'), dtype=t.bfloat16, device='cuda')
                    for output, ref in [(actual, 0), (reference, 1)]:
                        args = Args(x.data_ptr(), w.data_ptr(), cosine.data_ptr(), sine.data_ptr(),
                                    output.data_ptr(), b, 1e-6)
                        self.assertEqual(lib.mx5_transform(C.byref(args), dim, int(norm), ref,
                                                           C.c_void_p(t.cuda.current_stream().cuda_stream)), 0)
                    t.cuda.synchronize()
                    want = reference[:, 0].contiguous()
                    self.assertTrue(t.equal(actual.view(t.int16), want.view(t.int16)),
                                    f'mismatched BF16 bits: {int((actual.view(t.int16) != want.view(t.int16)).sum())}')


if __name__ == '__main__': unittest.main()
