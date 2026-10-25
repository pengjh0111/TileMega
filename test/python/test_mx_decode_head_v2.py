import ctypes as C
import os
from pathlib import Path
import unittest

import torch

from tilemega.serving.weights import _packed_gpu


class Args(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ['a', 'w', 'partial', 'ticket', 'calls', 'tokens']] + [
        (name, C.c_int) for name in ['n', 'k', 'b', 'capacity', 'past']]


@unittest.skipUnless(os.environ.get('MX5_HEAD_DIR'), 'set MX5_HEAD_DIR under GPU guard')
class HeadV2(unittest.TestCase):
    def test_argmax(self):
        self.assertEqual(os.environ.get('TILEMEGA_GPU_LOCK_HELD'), '1')
        for path in sorted(Path(os.environ['MX5_HEAD_DIR']).glob('tn*.so')):
            tn = int(path.stem[2:])
            lib = C.CDLL(str(path))
            for wn in [4, 8]:
                for depth in [1, 2, 3, 4]:
                    if (tn // 16) * 4 + depth * (tn // 16 + 1) * 4 > 160: continue
                    for b in [1, 2, 4, 8, 16]:
                        for ties in [False, True]:
                            with self.subTest(tn=tn, wn=wn, d=depth, b=b, ties=ties):
                                n, k, cap, past = tn * (104 * wn + 37), 256, 128, 65
                                g = torch.Generator().manual_seed(1234 + b)
                                x = torch.randn(b, k, generator=g).bfloat16().cuda()
                                # A power-of-two gap isolates the rounding of the argmax
                                # from FP32 summation-order differences in the oracle.
                                w = torch.zeros(n, k, dtype=torch.bfloat16, device='cuda')
                                w[13] = x[0] * 8
                                if ties: w[29] = w[13]
                                packed = _packed_gpu(dict(kind='tile_pages', layout='xcore_mma', tile_n=tn,
                                                          tile_k=32, source=dict(kind='alias', source='w')),
                                                     lambda _: w).contiguous()
                                partial = torch.empty(b, 104, 2, dtype=torch.int32, device='cuda')
                                ticket = torch.zeros(1, dtype=torch.int32, device='cuda')
                                calls = torch.zeros(104, 8, dtype=torch.int32, device='cuda')
                                tokens = torch.full((b, cap), -1, dtype=torch.int32, device='cuda')
                                args = Args(x.data_ptr(), packed.data_ptr(), partial.data_ptr(), ticket.data_ptr(),
                                            calls.data_ptr(), tokens.data_ptr(), n, k, b, cap, past)
                                inv = torch.rsqrt(x.float().square().mean(-1) + 1e-6)
                                expected = ((x.float() @ w.float().T) * inv[:, None]).bfloat16().float().argmax(-1)
                                for repeat in range(2):
                                    calls.zero_()
                                    self.assertEqual(lib.mx5_head(C.byref(args), wn, depth,
                                                     C.c_void_p(torch.cuda.current_stream().cuda_stream)), 0)
                                    torch.cuda.synchronize()
                                    self.assertTrue(torch.equal(tokens[:, past + 1].long(), expected))
                                    self.assertTrue(torch.all(calls == 1).item())
                                    self.assertEqual(ticket.item(), 0)
                                self.assertTrue(torch.all(tokens[:, :past + 1] == -1).item())


if __name__ == '__main__': unittest.main()
