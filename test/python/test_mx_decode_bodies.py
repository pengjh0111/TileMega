"""Numerical tests of the MX-4 decode stage bodies (Backend/Mx/DecodeGemm.h,
DecodeAttention.h) against FP32 torch, and bit for bit against the RotateRow
path for the Qkv epilogue (MX4 prompt §8).

Runs only under the GPU guard (TILEMEGA_GPU_LOCK_HELD=1) with the library
built from test/unit/mx/decode_bodies.cu (TILEMEGA_MX_DECODE_BODIES_SO).
"""
import ctypes
import os
import sys
import unittest

GUARDED = os.environ.get("TILEMEGA_GPU_LOCK_HELD") == "1" and os.environ.get("TILEMEGA_MX_DECODE_BODIES_SO")

RESIDUAL, SWIGLU, ARGMAX = 1, 2, 3
STREAM_K, TILE_RANGE = 0, 1
EPS = 1e-6


class DecodeGemmArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("a", "w", "out", "partial", "tickets", "q_norm", "k_norm", "cosine", "sine", "k_cache",
                 "vt_cache", "head_value", "head_index", "head_ticket", "tokens")] + \
               [("capacity", ctypes.c_int), ("past", ctypes.c_int), ("epsilon", ctypes.c_float),
                ("n", ctypes.c_int), ("k", ctypes.c_int), ("b", ctypes.c_int)]


class DecodeAttentionArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("qkv", "k_cache", "vt_cache", "ctx", "partial", "tickets")] + \
               [(name, ctypes.c_int) for name in ("capacity", "past", "b", "hkv")]


def depth_fits(tn, d):
    """§3.5 rule 6 (DecodeGemm.h MxDepthFits)."""
    return (tn // 16) * 4 + d * (tn // 16 + 1) * 4 <= 160


def ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else None


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
        from tilemega.serving.weights import _packed_gpu
        cls.torch = torch
        cls.pack_gpu = staticmethod(_packed_gpu)
        cls.lib = ctypes.CDLL(os.environ["TILEMEGA_MX_DECODE_BODIES_SO"])
        assert cls.lib.tm_decode_gemm_args_size() == ctypes.sizeof(DecodeGemmArgs)
        assert cls.lib.tm_decode_attention_args_size() == ctypes.sizeof(DecodeAttentionArgs)
        cls.grid = cls.lib.tm_decode_grid()

    def stream(self):
        return ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)

    def pack(self, w, tn):
        """The loader's tile_pages(xcore_mma, TN, 32) packing."""
        return self.pack_gpu(dict(kind="tile_pages", layout="xcore_mma", tile_n=tn, tile_k=32,
                                  source=dict(kind="alias", source="w")), lambda name: w).contiguous()

    def contributors(self, n, k, tn):
        """n(t) per tile under the stream-K split (§3.5)."""
        tiles, kc = n // tn, k // 32
        units, grid = tiles * kc, self.grid
        owner = lambda u: ((u + 1) * grid - 1) // units
        return [owner(t * kc + kc - 1) - owner(t * kc) + 1 for t in range(tiles)]


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_DECODE_BODIES_SO")
class DecodeGemm(Base):
    SHAPES = ((4096, 1024), (1024, 2048), (2048, 3072))

    def run_gemm(self, op, tn, d, norm, part, x, w, residual=None, capacity=256, past=70):
        torch = self.torch
        b, k = x.shape
        n = w.shape[0]
        tiles = n // tn
        contributors = self.lib.tm_decode_max_contributors(tiles, self.grid)
        packed = self.pack(w, tn)
        partial = torch.full((tiles * contributors * b * (tn + 1),), float("nan"), device="cuda")
        tickets = torch.zeros(tiles, dtype=torch.int32, device="cuda")
        out = head_value = head_index = head_ticket = tokens = None
        if op == RESIDUAL:
            out = residual.clone()
        elif op == SWIGLU:
            out = torch.zeros(b, n // 2, dtype=torch.bfloat16, device="cuda")
        else:
            head_value = torch.zeros(b, self.grid, device="cuda")
            head_index = torch.zeros(b, self.grid, dtype=torch.int32, device="cuda")
            head_ticket = torch.zeros(1, dtype=torch.int32, device="cuda")
            tokens = torch.full((b, capacity), -1, dtype=torch.int32, device="cuda")
        args = DecodeGemmArgs(ptr(x), ptr(packed), ptr(out), ptr(partial), ptr(tickets), None, None, None, None,
                              None, None, ptr(head_value), ptr(head_index), ptr(head_ticket), ptr(tokens),
                              capacity, past, EPS, n, k, b)
        code = self.lib.tm_decode_gemm(ctypes.byref(args), tn, d, op, int(norm), part, self.stream())
        torch.cuda.synchronize()
        self.assertEqual(code, 0)
        self.assertEqual(int(tickets.abs().sum()), 0, "tickets return to 0")
        if op == ARGMAX:
            self.assertEqual(int(head_ticket.item()), 0, "head ticket returns to 0")
            return tokens[:, past + 1].clone()
        return out

    def reference(self, x, w, norm):
        acc = x.float() @ w.float().t()
        if norm:
            inv = self.torch.rsqrt((x.float() ** 2).sum(1, keepdim=True) / x.shape[1] + EPS)
            acc = acc * inv
        return acc

    def close(self, got, want, scale=None):
        # The serving GEMM body tests' tolerance (test_mx_task_bodies.py):
        # value * 2^-7 plus an absolute floor, for the MMA's accumulation order.
        tol = (want.abs() if scale is None else scale) * (2.0 ** -7) + 2e-3
        return ((got - want).abs() <= tol) & ~self.torch.isnan(got)

    def assertBf16Close(self, got, ok, what):
        self.assertEqual(int((~ok).sum()), 0, f"{what}: {int((~ok).sum())} of {ok.numel()} out of tolerance")

    def roundings(self, acc):
        """bf16(acc) and, where acc sits within 2^-14 (relative) of a BF16
        rounding tie, its other neighbour. The MMA sums in another order
        (P-9), so an accumulator that close to a tie may round either way at
        the materialization point; with ~10^7 outputs checked this happened
        (a 1.7e-7 distance turned into 2 ulps after the residual add)."""
        torch = self.torch
        delta = acc.abs() * 2.0 ** -14 + 1e-6
        return (acc - delta).to(torch.bfloat16).float(), (acc + delta).to(torch.bfloat16).float()

    def operands(self, b, n, k, seed):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(b, k, generator=g) * 0.5).to(torch.bfloat16).cuda().contiguous()
        w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).cuda().contiguous()
        r = torch.randn(b, n, generator=g).to(torch.bfloat16).cuda().contiguous()
        return x, w, r

    def check(self, op, tn, d, norm, part, b, n, k):
        torch = self.torch
        x, w, r = self.operands(b, n, k, seed=n + k + 7 * b + tn + d)
        ref = self.reference(x, w, norm)
        what = f"op={op} tn={tn} d={d} norm={norm} part={part} b={b} n={n} k={k}"
        if op == RESIDUAL:
            out = self.run_gemm(op, tn, d, norm, part, x, w, residual=r)
            ok = torch.zeros_like(out, dtype=torch.bool)
            for acc in self.roundings(ref):
                want = (acc + r.float()).to(torch.bfloat16).float()
                ok |= self.close(out.float(), want, scale=ref.abs() + r.float().abs())
            self.assertBf16Close(out, ok, what)
            again = self.run_gemm(op, tn, d, norm, part, x, w, residual=r)
            self.assertTrue(bool(torch.equal(out.view(torch.int16), again.view(torch.int16))), what + " repeat")
        elif op == SWIGLU:
            out = self.run_gemm(op, tn, d, norm, part, x, w)
            cols = torch.arange(n // 2, device="cuda")
            gate_cols = 32 * (cols // 16) + cols % 16
            ok = torch.zeros_like(out, dtype=torch.bool)
            for gate in self.roundings(ref[:, gate_cols]):
                for up in self.roundings(ref[:, gate_cols + 16]):
                    act = (gate / (1 + torch.exp(-gate))).to(torch.bfloat16).float()
                    want = (act * up).to(torch.bfloat16).float()
                    ok |= self.close(out.float(), want)
            self.assertBf16Close(out, ok, what)
            again = self.run_gemm(op, tn, d, norm, part, x, w)
            self.assertTrue(bool(torch.equal(out.view(torch.int16), again.view(torch.int16))), what + " repeat")
        else:
            token = self.run_gemm(op, tn, d, norm, part, x, w)
            rounded = ref.to(torch.bfloat16).float()
            best = rounded.max(1).values
            self.assertTrue(bool(((token >= 0) & (token < n)).all()), what + " index range")
            picked = rounded.gather(1, token.long()[:, None]).squeeze(1)
            self.assertTrue(bool(((best - picked) <= best.abs() * 2 ** -7 + 2e-3).all()), what)
            again = self.run_gemm(op, tn, d, norm, part, x, w)
            self.assertTrue(bool(torch.equal(token, again)), what + " repeat")

    def matrix(self, op, parts):
        covered = set()
        i = 0
        for tn in (16, 32, 64, 128):
            if op == SWIGLU and tn < 32:
                continue
            for d in (1, 2, 3, 4):
                if not depth_fits(tn, d):
                    continue
                for norm in (False, True):
                    for part in parts:
                        for b in (1, 2, 4, 8, 16):
                            n, k = self.SHAPES[i % len(self.SHAPES)]
                            i += 1
                            if part == STREAM_K:
                                covered.update(c for c in self.contributors(n, k, tn) if c > 1)
                            with self.subTest(tn=tn, d=d, norm=norm, part=part, b=b, n=n, k=k):
                                self.check(op, tn, d, norm, part, b, n, k)
        return covered

    def test_residual(self):
        covered = self.matrix(RESIDUAL, (STREAM_K, TILE_RANGE))
        self.assertTrue({2, 3, 4} <= covered, f"stream-K contributors covered: {sorted(covered)}")

    def test_swiglu(self):
        self.matrix(SWIGLU, (STREAM_K, TILE_RANGE))

    def test_argmax(self):
        self.matrix(ARGMAX, (TILE_RANGE,))

    def test_argmax_ties(self):
        """Equal maxima in different tiles and CTAs: the smallest index wins."""
        torch = self.torch
        b, n, k = 4, 32768, 1024
        x = torch.zeros(b, k, dtype=torch.bfloat16, device="cuda")
        w = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
        for row, cols in enumerate(((5, 900, 30000), (31000, 17), (128, 129), (32767, 0))):
            x[row, row] = 1
            for c in cols:
                w[c, row] = 2
        for d in (1, 2):
            token = self.run_gemm(ARGMAX, 128, d, True, TILE_RANGE, x, w)
            self.assertEqual(token.tolist(), [5, 17, 128, 0])


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_DECODE_BODIES_SO")
class DecodeQkv(Base):
    """kQkvRope against the un-fused path bit for bit. The MMA's FP32
    accumulation is not exact even for small integers (P-9; a true zero came
    back as ~1e-6), so each row of x has one nonzero: every output is a single
    exact product and the comparison isolates the norm and RoPE epilogue."""
    CONFIGS = ((0, 128, 2, True), (1, 64, 4, False), (2, 128, 4, False))

    def test_qkv_rope(self):
        torch = self.torch
        hkv, k, capacity, past = 8, 1024, 256, 70
        for config, dim, q, qk_norm in self.CONFIGS:
            n = hkv * (q + 2) * dim
            g = torch.Generator().manual_seed(config + 11)
            w = torch.randn(n, k, generator=g).to(torch.bfloat16).cuda().contiguous()
            angle = torch.arange(capacity).float()[:, None] * (1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
            angle = torch.cat([angle, angle], -1)
            cos = angle.cos().to(torch.bfloat16).cuda().contiguous()
            sin = angle.sin().to(torch.bfloat16).cuda().contiguous()
            qn = (1 + 0.1 * torch.randn(dim, generator=g)).to(torch.bfloat16).cuda().contiguous()
            kn = (1 + 0.1 * torch.randn(dim, generator=g)).to(torch.bfloat16).cuda().contiguous()
            for tn in (dim, 2 * dim):
                if tn > 128:
                    continue
                for d in (1, 2, 3, 4):
                    if not depth_fits(tn, d):
                        continue
                    for part in (STREAM_K, TILE_RANGE):
                        for b in (1, 2, 4, 8, 16):
                            with self.subTest(config=config, tn=tn, d=d, part=part, b=b):
                                x = torch.zeros(b, k)
                                x[torch.arange(b), torch.randint(0, k, (b,), generator=g)] = \
                                    torch.tensor([1.0, -1.0, 2.0, -0.5])[torch.randint(0, 4, (b,), generator=g)]
                                x = x.to(torch.bfloat16).cuda().contiguous()
                                got = self.run_qkv(config, tn, d, part, x, w, None, qn, kn, cos, sin, capacity, past, hkv,
                                               q, dim)
                                want = self.run_qkv(config, tn, d, part, x, w, "reference", qn, kn, cos, sin, capacity,
                                                past, hkv, q, dim)
                                for name, a, e in zip(("qkv", "k", "vt"), got, want):
                                    self.assertTrue(bool(torch.equal(a.view(torch.int16), e.view(torch.int16))),
                                                    f"{name} differs at {int((a.view(torch.int16) != e.view(torch.int16)).sum())} places")

    def run_qkv(self, config, tn, d, part, x, w, mode, qn, kn, cos, sin, capacity, past, hkv, q, dim):
        torch = self.torch
        b, k = x.shape
        n = w.shape[0]
        qkv = torch.zeros(b, hkv, q + 2, dim, dtype=torch.bfloat16, device="cuda")
        kc = torch.zeros(b, hkv, capacity, dim, dtype=torch.bfloat16, device="cuda")
        vt = torch.zeros(b, hkv, dim, capacity, dtype=torch.bfloat16, device="cuda")
        tiles = n // tn
        partial = torch.full((tiles * self.lib.tm_decode_max_contributors(tiles, self.grid) * b * (tn + 1),),
                             float("nan"), device="cuda")
        tickets = torch.zeros(tiles, dtype=torch.int32, device="cuda")
        packed = self.pack(w, tn) if mode is None else None
        args = DecodeGemmArgs(ptr(x), ptr(packed), ptr(qkv), ptr(partial), ptr(tickets), ptr(qn), ptr(kn), ptr(cos),
                              ptr(sin), ptr(kc), ptr(vt), None, None, None, None, capacity, past, EPS, n, k, b)
        if mode is None:
            code = self.lib.tm_decode_qkv(ctypes.byref(args), tn, d, config, part, self.stream())
        else:
            scratch = torch.zeros(b, n, dtype=torch.bfloat16, device="cuda")
            code = self.lib.tm_decode_qkv_reference(ctypes.byref(args), ptr(w), config, ptr(scratch), self.stream())
        torch.cuda.synchronize()
        self.assertEqual(code, 0)
        self.assertEqual(int(tickets.abs().sum()), 0)
        return qkv, kc, vt


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_DECODE_BODIES_SO")
class DecodeAttention(Base):
    def run_attention(self, dim, q, tmin, b, hkv, length, capacity, seed):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        qkv = torch.randn(b, hkv, q + 2, dim, generator=g).to(torch.bfloat16).cuda().contiguous()
        kc = torch.randn(b, hkv, capacity, dim, generator=g).to(torch.bfloat16).cuda().contiguous()
        vt = torch.randn(b, hkv, dim, capacity, generator=g).to(torch.bfloat16).cuda().contiguous()
        # The cache past L is uninitialized in serving: NaN must not leak.
        kc[:, :, length:] = float("nan")
        vt[:, :, :, length:] = float("nan")
        ctx = torch.full((b, hkv * q, dim), float("nan"), dtype=torch.bfloat16, device="cuda")
        partial = torch.full((b * hkv * ((capacity + 31) // 32) * q * (dim + 2),), float("nan"), device="cuda")
        tickets = torch.zeros(b * hkv, dtype=torch.int32, device="cuda")
        args = DecodeAttentionArgs(ptr(qkv), ptr(kc), ptr(vt), ptr(ctx), ptr(partial), ptr(tickets), capacity,
                                   length - 1, b, hkv)
        code = self.lib.tm_decode_attention(ctypes.byref(args), dim, q, tmin, self.stream())
        torch.cuda.synchronize()
        self.assertEqual(code, 0)
        self.assertEqual(int(tickets.abs().sum()), 0, "tickets return to 0")
        query = qkv[:, :, :q].float()                              # [b][g][q][d]
        keys = kc[:, :, :length].float()                           # [b][g][t][d]
        values = vt[:, :, :, :length].float().transpose(2, 3)      # [b][g][t][d]
        score = query @ keys.transpose(2, 3) / dim ** 0.5
        want = torch.softmax(score, -1) @ values                   # [b][g][q][d]
        return ctx.view(b, hkv, q, dim).float(), want

    def chunks(self, b, hkv, length, tmin):
        slots = self.grid * 8
        tokens = 256
        for t in range((tmin + 31) // 32 * 32, 256, 32):
            if b * hkv * -(-length // t) <= slots:
                tokens = t
                break
        return -(-length // tokens)

    def test_attention(self):
        torch = self.torch
        capacity, hkv = 1152, 8
        seen = set()
        for dim in (64, 128):
            for q in (2, 4):
                for b in (1, 16):
                    for length in (1, 31, 32, 33, 65, 257, 1088):
                        for tmin in ((32, 64, 128) if length in (257, 1088) else (64,)):
                            seen.add(self.chunks(b, hkv, length, tmin) > 1)
                            with self.subTest(dim=dim, q=q, b=b, length=length, tmin=tmin):
                                got, want = self.run_attention(dim, q, tmin, b, hkv, length, capacity,
                                                               seed=dim + q + b + length)
                                err = (got - want).abs()
                                tol = want.abs() * 2 ** -6 + 1e-2
                                self.assertFalse(bool(torch.isnan(got).any()), "NaN")
                                self.assertTrue(bool((err <= tol).all()), f"max |d|={float(err.max())}")
        self.assertEqual(seen, {False, True}, "both C = 1 and C > 1 covered")


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_DECODE_BODIES_SO")
class DecodeEmbed(Base):
    def test_embed(self):
        torch = self.torch
        vocab, hidden, capacity, past = 1000, 1024, 256, 63
        for b in (1, 16):
            embed = torch.randn(vocab, hidden).to(torch.bfloat16).cuda().contiguous()
            tokens = torch.randint(0, vocab, (b, capacity), dtype=torch.int32).cuda().contiguous()
            x = torch.zeros(b, hidden, dtype=torch.bfloat16, device="cuda")
            code = self.lib.tm_decode_embed(ptr(embed), ptr(tokens), ptr(x), b, hidden, capacity, past, self.stream())
            torch.cuda.synchronize()
            self.assertEqual(code, 0)
            self.assertTrue(bool(torch.equal(x, embed[tokens[:, past].long()])))


if __name__ == "__main__":
    unittest.main()


class PrefillArgs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ("qkv", "k_cache", "vt_cache", "cosine", "sine", "q_norm", "k_norm", "context")] + \
               [(name, ctypes.c_int) for name in ("batch", "heads_kv", "capacity")] + [("epsilon", ctypes.c_float)]


@unittest.skipUnless(GUARDED, "needs the GPU guard and TILEMEGA_MX_DECODE_BODIES_SO")
class PrefillToDecode(Base):
    """§8: the prefill attention body writes K rows and V^T columns [0, 64)
    (kv_layout vt); MX-4 decode attention reads them at past = 64. Both
    against one HuggingFace attention layer (Qwen3 RMSNorm and RoPE helpers,
    eager causal softmax in FP32)."""

    def test_prefill_then_decode(self):
        torch = self.torch
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb
        assert self.lib.tm_decode_prefill_args_size() == ctypes.sizeof(PrefillArgs)
        capacity, hkv, tokens, eps = 1088, 8, 64, EPS
        for dim, q, qk_norm in ((128, 2, True), (64, 4, False), (128, 4, False)):
            for b in (1, 4):
                with self.subTest(dim=dim, q=q, b=b):
                    g = torch.Generator().manual_seed(dim + q + b)
                    raw = torch.randn(b, tokens + 1, hkv, q + 2, dim, generator=g).to(torch.bfloat16).cuda()
                    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
                    angle = torch.arange(capacity).float()[:, None] * inv[None, :]
                    angle = torch.cat([angle, angle], -1)
                    cos = angle.cos().to(torch.bfloat16).cuda().contiguous()
                    sin = angle.sin().to(torch.bfloat16).cuda().contiguous()
                    qn = (1 + 0.1 * torch.randn(dim, generator=g)).to(torch.bfloat16).cuda().contiguous()
                    kn = (1 + 0.1 * torch.randn(dim, generator=g)).to(torch.bfloat16).cuda().contiguous()
                    kc = torch.full((b, hkv, capacity, dim), float("nan"), dtype=torch.bfloat16, device="cuda")
                    vt = torch.full((b, hkv, dim, capacity), float("nan"), dtype=torch.bfloat16, device="cuda")
                    ctx = torch.zeros(b, tokens, hkv * q, dim, dtype=torch.bfloat16, device="cuda")
                    prompt = raw[:, :tokens].contiguous()
                    args = PrefillArgs(ptr(prompt), ptr(kc), ptr(vt), ptr(cos), ptr(sin), ptr(qn), ptr(kn), ptr(ctx),
                                       b, hkv, capacity, eps)
                    self.assertEqual(self.lib.tm_decode_prefill_vt(ctypes.byref(args), dim, q, int(qk_norm),
                                                                   self.stream()), 0)
                    torch.cuda.synchronize()

                    # HF: q [B,H,T,D], k/v [B,hkv,T,D] for positions 0..64.
                    qh = raw[:, :, :, :q].reshape(b, tokens + 1, hkv * q, dim).transpose(1, 2)
                    kh = raw[:, :, :, q].transpose(1, 2)
                    vh = raw[:, :, :, q + 1].transpose(1, 2)
                    if qk_norm:
                        qnorm, knorm = Qwen3RMSNorm(dim, eps).cuda().to(torch.bfloat16), \
                            Qwen3RMSNorm(dim, eps).cuda().to(torch.bfloat16)
                        qnorm.weight.data.copy_(qn)
                        knorm.weight.data.copy_(kn)
                        with torch.no_grad():
                            qh, kh = qnorm(qh), knorm(kh)
                    pos = torch.arange(tokens + 1, device="cuda")
                    qh, kh = apply_rotary_pos_emb(qh, kh, cos[pos][None], sin[pos][None])
                    keys = kh.float().repeat_interleave(q, 1)
                    values = vh.float().repeat_interleave(q, 1)
                    score = qh.float() @ keys.transpose(2, 3) / dim ** 0.5
                    score = score.masked_fill(pos[None, None, None, :] > pos[None, None, :, None], float("-inf"))
                    want = torch.softmax(score, -1) @ values          # [B][H][T+1][D]

                    got = ctx.float().transpose(1, 2)                 # [B][H][64][D]
                    err = (got - want[:, :, :tokens]).abs()
                    self.assertTrue(bool((err <= want[:, :, :tokens].abs() * 2 ** -6 + 2e-2).all()),
                                    f"prefill max |d|={float(err.max())}")
                    # The prefill wrote K and V^T [0, 64) and nothing beyond.
                    self.assertFalse(bool(torch.isnan(kc[:, :, :tokens]).any() or torch.isnan(vt[:, :, :, :tokens]).any()))
                    self.assertTrue(bool(torch.isnan(kc[:, :, tokens:]).all() and torch.isnan(vt[:, :, :, tokens:]).all()))

                    # Decode at past = 64: the Qkv epilogue's outputs, written as it does.
                    kc[:, :, tokens] = kh[:, :, tokens].to(torch.bfloat16)
                    vt[:, :, :, tokens] = vh[:, :, tokens].to(torch.bfloat16)
                    qkv = torch.zeros(b, hkv, q + 2, dim, dtype=torch.bfloat16, device="cuda")
                    qkv[:, :, :q] = qh[:, :, tokens].reshape(b, hkv, q, dim).to(torch.bfloat16)
                    out = torch.full((b, hkv * q, dim), float("nan"), dtype=torch.bfloat16, device="cuda")
                    partial = torch.full((b * hkv * ((capacity + 31) // 32) * q * (dim + 2),), float("nan"),
                                         device="cuda")
                    tickets = torch.zeros(b * hkv, dtype=torch.int32, device="cuda")
                    dargs = DecodeAttentionArgs(ptr(qkv), ptr(kc), ptr(vt), ptr(out), ptr(partial), ptr(tickets),
                                                capacity, tokens, b, hkv)
                    self.assertEqual(self.lib.tm_decode_attention(ctypes.byref(dargs), dim, q, 64, self.stream()), 0)
                    torch.cuda.synchronize()
                    last = want[:, :, tokens]                         # [B][H][D]
                    err = (out.float() - last).abs()
                    self.assertTrue(bool((err <= last.abs() * 2 ** -6 + 2e-2).all()),
                                    f"decode max |d|={float(err.max())}")
