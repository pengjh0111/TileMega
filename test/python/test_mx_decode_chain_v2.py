import ctypes as C
import os
import unittest


class Args(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['x', 'w0', 'w1', 'w2', 'residual', 'y0', 'y1', 'y2',
                                              'partial', 'tickets', 'calls', 'counters']] + [
        ('batch', C.c_int), ('iterations', C.c_int)]


@unittest.skipUnless(os.environ.get('MX5_CHAIN_SO') and os.environ.get('TILEMEGA_GPU_LOCK_HELD') == '1',
                     'requires GPU guard and chain library')
class PrefetchChain(unittest.TestCase):
    def test_chain(self):
        import torch as t
        from tilemega.serving.weights import _packed_gpu
        lib = C.CDLL(os.environ['MX5_CHAIN_SO'])
        generator = t.Generator().manual_seed(9912)
        matrices = [(t.randn(n, k, generator=generator) * .02).bfloat16().cuda()
                    for n, k in [(1664, 256), (1280, 1664), (13312, 640)]]
        packed = [_packed_gpu(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn, tile_k=32,
                                   source=dict(kind='alias', source='w')), lambda _, w=w: w).contiguous()
                  for tn, w in zip([16, 32, 16], matrices)]
        for b in [1, 16]:
            x = t.randn(b, 256, generator=generator).bfloat16().cuda()
            residual = t.randn(b, 13312, generator=generator).bfloat16().cuda()
            output = [t.empty(b, n, dtype=t.bfloat16, device='cuda') for n in [1664, 640, 13312]]
            partial = t.empty(40 * 4 * b * 33, device='cuda')
            tickets = t.zeros(40, dtype=t.int32, device='cuda')
            calls = t.zeros(3, 104, 8, dtype=t.int32, device='cuda')
            counters = t.zeros(3, 8, dtype=t.int64, device='cuda')
            iterations = int(os.environ.get('MX5_CHAIN_ITERS', '2000'))
            args = Args(*[v.data_ptr() for v in [x, *packed, residual, *output, partial, tickets, calls, counters]],
                        b, iterations)
            self.assertEqual(lib.mx5_chain(C.byref(args), C.c_void_p(t.cuda.current_stream().cuda_stream)), 0)
            t.cuda.synchronize()
            self.assertTrue(t.all(calls == iterations).item())
            self.assertTrue(t.all(counters[:, 0] == iterations * 104).item())
            self.assertEqual(tickets.abs().sum().item(), 0)
            a = ((x.float() @ matrices[0].float().T) * t.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            z = ((a.float() @ matrices[1].float().T) * t.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
            pair = z.reshape(b, -1, 2, 16)
            gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
            h = ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(b, -1)
            y = ((h.float() @ matrices[2].float().T).bfloat16().float() + residual.float()).bfloat16()
            for actual, expected in zip(output, [a, h, y]):
                self.assertTrue(t.all((actual.float() - expected.float()).abs() <= .04 + .03 * expected.float().abs()).item())


if __name__ == '__main__': unittest.main()
