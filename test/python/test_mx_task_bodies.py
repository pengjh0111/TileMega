"""Numerical tests of the xcore1000 serving TaskBodies against FP32 torch.

Runs only under the GPU guard (TILEMEGA_GPU_LOCK_HELD=1) with the test
library built from test/unit/mx/serving_bodies.cu (TILEMEGA_MX_BODIES_SO).
"""
import ctypes
import os
import unittest

GUARDED = os.environ.get("TILEMEGA_GPU_LOCK_HELD") == "1" and os.environ.get("TILEMEGA_MX_BODIES_SO")


class GemmArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("a", "b", "residual", "output", "partial", "argmax_value", "argmax_index", "norm_ss", "ss_out")] + \
               [(name, ctypes.c_int) for name in
                ("m", "n", "k_total", "k_begin", "k_count", "output_stride", "partial_stride",
                 "a_row_stride", "b_row_stride", "norm_k")] + \
               [("norm_eps", ctypes.c_float), ("epilogue", ctypes.c_int)]


STORE, RESIDUAL, SWIGLU, ARGMAX, PARTIAL = range(5)
SWIGLU_U = 16


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_BODIES_SO")
class ServingGemmBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.lib = ctypes.CDLL(os.environ["TILEMEGA_MX_BODIES_SO"])
        assert cls.lib.tm_test_gemm_args_size() == ctypes.sizeof(GemmArgs)

    def operands(self, m, n, k, seed):
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        a = (torch.randn(m, k, generator=g) * 0.5).to(torch.bfloat16).cuda().contiguous()
        b = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda().contiguous()
        r = (torch.randn(m, n, generator=g)).to(torch.bfloat16).cuda().contiguous()
        return a, b, r

    def run_body(self, kind, tile_m, tile_n, a, b, op, k_begin=0, k_count=None, residual=None):
        torch = self.torch
        m, k = a.shape
        n = b.shape[0]
        k_count = k - k_begin if k_count is None else k_count
        out_n = n // 2 if op == SWIGLU else n
        tiles_n = (n + tile_n - 1) // tile_n
        output = torch.zeros(m, out_n, dtype=torch.bfloat16, device="cuda")
        partial = torch.full((m, n), float("nan"), dtype=torch.float32, device="cuda")
        argmax_value = torch.zeros(m, tiles_n, dtype=torch.float32, device="cuda")
        argmax_index = torch.full((m, tiles_n), -1, dtype=torch.int32, device="cuda")
        args = GemmArgs(a.data_ptr(), b.data_ptr(), residual.data_ptr() if residual is not None else None,
                        output.data_ptr(), partial.data_ptr(), argmax_value.data_ptr(), argmax_index.data_ptr(),
                        None, None, m, n, k, k_begin, k_count,
                        tiles_n if op == ARGMAX else out_n, n, k, k, 0, 0.0, op)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        if kind == "gemm":
            code = self.lib.tm_test_gemm(ctypes.byref(args), tile_m, tile_n, stream)
        else:
            code = self.lib.tm_test_gemv(ctypes.byref(args), tile_n, stream)
        self.assertEqual(code, 0)
        torch.cuda.synchronize()
        return output, partial, argmax_value, argmax_index

    def reference(self, a, b, k_begin=0, k_count=None):
        k = a.shape[1]
        k_count = k - k_begin if k_count is None else k_count
        sl = slice(k_begin, k_begin + k_count)
        return a[:, sl].float() @ b[:, sl].float().t()

    def assertBf16Close(self, got, want, what, scale=None):
        torch = self.torch
        got, want = got.float(), want.float()
        # Two BF16 ulps of the operand scale plus an absolute floor: the MMA
        # accumulates in a different order and is not bit-exact IEEE (P-9),
        # and a residual add rounds at the larger operand's ulp.
        tol = (want.abs() if scale is None else scale) * (2.0 ** -7) + 2e-3
        bad = ((got - want).abs() > tol) | torch.isnan(got)
        self.assertEqual(int(bad.sum()), 0,
                         f"{what}: {int(bad.sum())} of {bad.numel()} differ; max |d|={float((got-want).abs().max())}")

    def check_ops(self, kind, tile_m, tile_n, m, n, k):
        torch = self.torch
        a, b, r = self.operands(m, n, k, seed=m * 1000 + n + k)
        ref = self.reference(a, b)
        out, _, _, _ = self.run_body(kind, tile_m, tile_n, a, b, STORE)
        self.assertBf16Close(out, ref.to(torch.bfloat16), f"{kind} {tile_m}x{tile_n} store m={m} n={n} k={k}")
        out, _, _, _ = self.run_body(kind, tile_m, tile_n, a, b, RESIDUAL, residual=r)
        want = (ref.to(torch.bfloat16).float() + r.float()).to(torch.bfloat16)
        self.assertBf16Close(out, want, f"{kind} residual m={m} n={n} k={k}",
                             scale=ref.abs() + r.float().abs())
        _, partial, _, _ = self.run_body(kind, tile_m, tile_n, a, b, PARTIAL)
        tol = ref.abs() * 1e-4 + 1e-4
        self.assertTrue(bool(((partial - ref).abs() <= tol).all()), f"{kind} partial max |d|="
                        f"{float((partial-ref).abs().max())}")
        # split-K: two halves summed by the host equal the whole.
        half = (k // 2) // 8 * 8
        _, p0, _, _ = self.run_body(kind, tile_m, tile_n, a, b, PARTIAL, 0, half)
        _, p1, _, _ = self.run_body(kind, tile_m, tile_n, a, b, PARTIAL, half, k - half)
        self.assertTrue(bool(((p0 + p1 - ref).abs() <= tol * 2).all()), f"{kind} split-K")
        if n % (2 * SWIGLU_U) == 0 and tile_n % (2 * SWIGLU_U) == 0:
            out, _, _, _ = self.run_body(kind, tile_m, tile_n, a, b, SWIGLU)
            cols = torch.arange(n // 2, device="cuda")
            gate_cols = 2 * SWIGLU_U * (cols // SWIGLU_U) + cols % SWIGLU_U
            gate = ref[:, gate_cols].to(torch.bfloat16).float()
            up = ref[:, gate_cols + SWIGLU_U].to(torch.bfloat16).float()
            act = (gate / (1 + torch.exp(-gate))).to(torch.bfloat16).float()
            self.assertBf16Close(out, (act * up).to(torch.bfloat16), f"{kind} swiglu m={m} n={n}")
        _, _, value, index = self.run_body(kind, tile_m, tile_n, a, b, ARGMAX)
        tiles_n = (n + tile_n - 1) // tile_n
        rounded = ref.to(torch.bfloat16).float()
        for t in range(tiles_n):
            block = rounded[:, t * tile_n:min(n, (t + 1) * tile_n)]
            best = block.max(1).values
            got_index = index[:, t].long() - t * tile_n
            self.assertTrue(bool((got_index >= 0).all() and (got_index < block.shape[1]).all()), "argmax index range")
            picked = block.gather(1, got_index[:, None]).squeeze(1)
            # A chosen column must be (within accumulation noise) a maximum.
            self.assertTrue(bool(((best - picked) <= best.abs() * 2 ** -7 + 2e-3).all()), f"argmax tile {t}")

    def test_gemm_shapes(self):
        for tile_m, tile_n in ((16, 32), (16, 64), (16, 128), (32, 64), (64, 64), (64, 128)):
            for m, n, k in ((1, 96, 256), (16, 256, 1024), (7, 200, 520), (64, 128, 128), (100, 160, 336)):
                with self.subTest(tile=(tile_m, tile_n), shape=(m, n, k)):
                    self.check_ops("gemm", tile_m, tile_n, m, n, k)

    def test_gemv_shapes(self):
        for tile_n in (8, 16, 32, 64):
            for m, n, k in ((1, 96, 256), (4, 200, 1024), (3, 64, 520), (2, 1000, 3072)):
                with self.subTest(tile=tile_n, shape=(m, n, k)):
                    self.check_ops("gemv", 16, tile_n, m, n, k)


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_BODIES_SO")
class ServingTiledWeights(unittest.TestCase):
    """The loader's xcore_mma tile_pages packing and the tiled MMA mainloop agree."""

    def test_tiled_dense(self):
        import torch
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
        from tilemega.serving.weights import _packed_gpu
        lib = ctypes.CDLL(os.environ["TILEMEGA_MX_BODIES_SO"])
        g = torch.Generator().manual_seed(7)
        for tm, tn, tk in ((16, 32, 64), (16, 32, 128), (16, 128, 64), (32, 64, 128), (64, 128, 64)):
            for m, n, k, splits in ((1, 96, 512, 1), (16, 200, 1024, 2), (5, 64, 448, 1), (64, 128, 768, 3)):
                with self.subTest(tile=(tm, tn, tk), shape=(m, n, k), splits=splits):
                    a = (torch.randn(m, k, generator=g) * 0.5).to(torch.bfloat16).cuda().contiguous()
                    b = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda().contiguous()
                    packed = _packed_gpu(dict(kind="tile_pages", layout="xcore_mma", tile_n=tn, tile_k=tk,
                                              source=dict(kind="alias", source="w")), lambda name: b).contiguous()
                    kt = (k + tk - 1) // tk
                    # Split-K chunks start on tile boundaries; the last carries the remainder.
                    per = (kt + splits - 1) // splits * tk
                    tiles_m, tiles_n = (m + tm - 1) // tm, (n + tn - 1) // tn
                    total = torch.zeros(m, n, dtype=torch.float32, device="cuda")
                    for begin in range(0, k, per):
                        count = min(per, k - begin)
                        tiles = torch.full((tiles_m * tiles_n, tm * tn), float("nan"), device="cuda")
                        args = GemmArgs(a.data_ptr(), b.data_ptr(), None, None, None, None, None, None, None,
                                        m, n, k, begin, count, n, n, k, k, 0, 0.0, STORE)
                        code = lib.tm_test_gemm_tiled(ctypes.byref(args), ctypes.c_void_p(packed.data_ptr()), k,
                                                      tm, tn, tk, ctypes.c_void_p(tiles.data_ptr()),
                                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                        self.assertEqual(code, 0)
                        torch.cuda.synchronize()
                        full = tiles.view(tiles_m, tiles_n, tm, tn).permute(0, 2, 1, 3).reshape(tiles_m * tm, tiles_n * tn)
                        total += full[:m, :n]
                    ref = a.float() @ b.float().t()
                    err = (total - ref).abs()
                    self.assertTrue(bool((err <= ref.abs() * 1e-4 + 1e-3).all()), f"max |d|={float(err.max())}")


# MACA torch 2.10 gives a 4-D host-to-device copy channels-last-like strides
# (verified); every operand is made contiguous on the device.
class AttentionArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("qkv", "key_cache", "value_cache", "cosine", "sine", "q_norm", "k_norm", "context",
                 "partial", "lse")] + \
               [(name, ctypes.c_int) for name in ("batch", "heads_kv", "capacity", "past", "block_extent")] + \
               [("epsilon", ctypes.c_float)]


def rotate(torch, x, norm_weight, cos, sin, eps):
    """RotateRow's rounding sequence: bf16 after norm, weight and each RoPE product."""
    bf = torch.bfloat16
    d = x.shape[-1]
    a = x.float()
    partner = torch.cat([a[..., d // 2:], a[..., :d // 2]], -1)
    if norm_weight is not None:
        inv = torch.rsqrt((a * a).sum(-1, keepdim=True) / d + eps)
        w = norm_weight.float()
        pw = torch.cat([w[d // 2:], w[:d // 2]])
        a = (a * inv).to(bf).float() * w
        a = a.to(bf).float()
        partner = ((partner * inv).to(bf).float() * pw).to(bf).float()
    sign = torch.cat([-torch.ones(d // 2), torch.ones(d // 2)]).to(x.device)
    first = (a * cos.float()).to(bf).float()
    second = (sign * partner * sin.float()).to(bf).float()
    return (first + second).to(bf)


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_BODIES_SO")
class ServingAttentionBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.lib = ctypes.CDLL(os.environ["TILEMEGA_MX_BODIES_SO"])
        assert cls.lib.tm_test_attention_args_size() == ctypes.sizeof(AttentionArgs)

    def tables(self, capacity, d, seed):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        inv = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        angle = torch.arange(capacity).float()[:, None] * inv[None, :]
        angle = torch.cat([angle, angle], -1)
        cos, sin = angle.cos().to(torch.bfloat16).cuda().contiguous(), angle.sin().to(torch.bfloat16).cuda().contiguous()
        qn = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).cuda().contiguous()
        kn = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).cuda().contiguous()
        return cos, sin, qn, kn, g

    def test_decode(self):
        torch = self.torch
        for d, q, norm in ((128, 2, 1), (64, 4, 0), (128, 4, 0), (128, 3, 0), (128, 8, 1)):
            for batch, heads, capacity, past, block in ((1, 8, 1088, 64, 1088), (2, 4, 1088, 700, 256),
                                                        (3, 2, 512, 0, 512), (1, 8, 1088, 1086, 128)):
                with self.subTest(d=d, q=q, norm=norm, batch=batch, past=past, block=block):
                    cos, sin, qn, kn, g = self.tables(capacity, d, seed=d + q + past)
                    qkv = torch.randn(batch, heads, (q + 2) * d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    kc = torch.randn(batch, heads, capacity, d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    vc = torch.randn(batch, heads, capacity, d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    kc0, vc0 = kc.clone(), vc.clone()
                    blocks = (capacity + block - 1) // block
                    context = torch.zeros(batch, heads, q, d, dtype=torch.bfloat16, device="cuda")
                    partial = torch.zeros(batch, heads, blocks, q, d, dtype=torch.float32, device="cuda")
                    lse = torch.full((batch, heads, blocks, q), float("-inf"), dtype=torch.float32, device="cuda")
                    args = AttentionArgs(qkv.data_ptr(), kc.data_ptr(), vc.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                         qn.data_ptr(), kn.data_ptr(), context.data_ptr(), partial.data_ptr(),
                                         lse.data_ptr(), batch, heads, capacity, past, block, 1e-6)
                    code = self.lib.tm_test_decode_attention(ctypes.byref(args), d, q, norm,
                                                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                    self.assertEqual(code, 0)
                    torch.cuda.synchronize()
                    x = qkv.view(batch, heads, q + 2, d)
                    w_q, w_k = (qn, kn) if norm else (None, None)
                    queries = rotate(torch, x[:, :, :q], w_q, cos[past], sin[past], 1e-6)
                    key = rotate(torch, x[:, :, q], w_k, cos[past], sin[past], 1e-6)
                    keys, values = kc0.clone(), vc0.clone()
                    keys[:, :, past], values[:, :, past] = key, x[:, :, q + 1]
                    # RMS sums differ in association from torch: allow one bf16 ulp.
                    self.assertTrue(torch.allclose(kc[:, :, past].float(), key.float(), rtol=2 ** -7, atol=1e-3),
                                    f"new K row d={d} q={q} norm={norm} batch={batch} past={past} block={block}: max |d|={float((kc[:, :, past].float()-key.float()).abs().max())}")
                    self.assertTrue(torch.equal(vc[:, :, past], x[:, :, q + 1]), "new V row in cache")
                    scores = queries.float() @ keys[:, :, :past + 1].float().transpose(-1, -2) / d ** 0.5
                    want = torch.softmax(scores, -1) @ values[:, :, :past + 1].float()
                    if blocks == 1:
                        got = context.float()
                    else:
                        # Merge the per-block partials with their log2-sum-exp weights.
                        valid = torch.isfinite(lse)
                        m = torch.where(valid, lse, torch.full_like(lse, -1e30)).max(2, keepdim=True).values
                        wgt = torch.where(valid, torch.exp2(lse - m), torch.zeros_like(lse))
                        got = (partial * wgt[..., None]).sum(2) / wgt.sum(2)[..., None]
                    err = (got - want).abs()
                    self.assertLess(float(err.max()), 2e-2, f"max |d|={float(err.max())}")

    def test_prefill(self):
        torch = self.torch
        tokens = 64
        for d, q, norm, kv, rows in ((128, 2, 1, 32, 64), (128, 2, 1, 64, 16), (64, 4, 0, 64, 64), (128, 4, 0, 32, 32)):
            for batch, heads, capacity, past in ((1, 8, 1088, 0), (2, 4, 512, 100)):
                with self.subTest(d=d, q=q, kv=kv, rows=rows, batch=batch, past=past):
                    cos, sin, qn, kn, g = self.tables(capacity, d, seed=d * q + past)
                    qkv = torch.randn(batch, tokens, heads, (q + 2) * d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    kc = torch.randn(batch, heads, capacity, d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    vc = torch.randn(batch, heads, capacity, d, generator=g).to(torch.bfloat16).cuda().contiguous()
                    kc0, vc0 = kc.clone(), vc.clone()
                    context = torch.zeros(batch, tokens, heads, q, d, dtype=torch.bfloat16, device="cuda")
                    args = AttentionArgs(qkv.data_ptr(), kc.data_ptr(), vc.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                         qn.data_ptr(), kn.data_ptr(), context.data_ptr(), None, None,
                                         batch, heads, capacity, past, capacity, 1e-6)
                    code = self.lib.tm_test_prefill_attention(ctypes.byref(args), d, q, norm, kv, rows,
                                                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                    self.assertEqual(code, 0)
                    torch.cuda.synchronize()
                    x = qkv.view(batch, tokens, heads, q + 2, d)
                    pos = torch.arange(tokens, device="cuda") + past
                    w_q, w_k = (qn, kn) if norm else (None, None)
                    queries = rotate(torch, x[:, :, :, :q], w_q, cos[pos][None, :, None, None], sin[pos][None, :, None, None], 1e-6)
                    new_k = rotate(torch, x[:, :, :, q], w_k, cos[pos][None, :, None], sin[pos][None, :, None], 1e-6)
                    keys, values = kc0.clone(), vc0.clone()
                    keys[:, :, past:past + tokens] = new_k.transpose(1, 2)
                    values[:, :, past:past + tokens] = x[:, :, :, q + 1].transpose(1, 2)
                    dk = (kc[:, :, past:past + tokens].float() - keys[:, :, past:past + tokens].float()).abs()
                    self.assertTrue(torch.allclose(kc[:, :, past:past + tokens].float(),
                                                   keys[:, :, past:past + tokens].float(), rtol=2 ** -7, atol=1e-3),
                                    f"K cache max |d|={float(dk.max())} bad={int((dk > 1e-3 + 2 ** -7 * keys[:, :, past:past + tokens].float().abs()).sum())}")
                    self.assertTrue(torch.equal(vc[:, :, past:past + tokens], values[:, :, past:past + tokens]), "V cache")
                    total = past + tokens
                    qh = queries.permute(0, 2, 1, 3, 4).reshape(batch, heads, tokens * q, d).float()
                    scores = qh @ keys[:, :, :total].float().transpose(-1, -2) / d ** 0.5
                    row_token = torch.arange(tokens * q, device="cuda") // q
                    mask = torch.arange(total, device="cuda")[None, :] <= (past + row_token)[:, None]
                    scores = scores.masked_fill(~mask, float("-inf"))
                    want = (torch.softmax(scores, -1) @ values[:, :, :total].float())
                    want = want.view(batch, heads, tokens, q, d).permute(0, 2, 1, 3, 4)
                    err = (context.float() - want).abs()
                    self.assertLess(float(err.max()), 3e-2, f"max |d|={float(err.max())}")


if __name__ == "__main__":
    unittest.main()
