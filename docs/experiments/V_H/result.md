# V-H — `torch.export` coverage

Evidence labels: ✅ executed/observed; ⚠️ interface or portability caveat;
❌ conjecture.

## Export result

✅ PyTorch `2.13.0+cpu` produced a real strict `ExportedProgram` for a
randomly initialized two-layer Llama decoder (hidden 512, four query heads,
two KV heads, RoPE, GQA, SwiGLU, and explicit KV cache). No model weights or
CUDA PyTorch package were downloaded. Three independent strict exports had
the same FX-graph SHA-256, and the saved program ran at `(seq,past)` values
`(1,8)`, `(8,1)`, and `(2,5)`.

The complete graph has 179 `call_function` task candidates and 222 direct
tensor dependencies. `raw/exported_program.pt2`, `exported_program.txt`,
`node_shapes.tsv`, `aten_ops.tsv`, and `cg.json` are executable/raw evidence.

No `strict=False` workaround was needed. Practical pitfalls were:

- A `Dim` reused for four cache inputs still became `s14`, `s61`, and `s65`
  plus equality guards; symbol spelling is not semantic identity.
- The executable guards are exposed by the private
  `ExportedProgram._guards_code` field in this release. A frontend must use a
  supported guard API when one is available, or version-pin this adapter.
- `ExportedProgram.module()` is executable, but calling `.eval()` on that
  module raises `NotImplementedError` in this release.
- Export inserts metadata checks, symbolic-size operations, Python
  `operator.add/getitem`, and layout-only ATen operations in addition to the
  mathematical decoder operators.
- The first two official CPU-wheel transfers failed (timeout and SSL EOF),
  and the local proxy died during a retry. Direct resumable range requests
  completed the exact 191,774,032-byte wheel. See `raw/install_attempts.txt`.

## KV cache

✅ KV state is four explicit user inputs and four explicit user outputs. It is
neither a mutated buffer nor hidden outside the graph. A deliberately unequal
K/V cache length was rejected with:

```text
Guard failed: hidden.size()[1] + past_v0.size()[2]
              == hidden.size()[1] + past_k0.size()[2]
```

For this dense cache, Tier 1 layout cancellation can describe old positions
and the contiguous append interval: logical position is enough to cancel the
physical dense stride, and `cat` adds the affine boundary `past + seq`.
⚠️ This result does not cover paged KV cache block-table lookup; that remains
Tier 1 only if the same indirection is proven to cancel on both accesses.

## Symbolic shapes and guards

✅ Range constraints were:

```text
s11, s14, s61, s65 in [1, 8]
```

Representative propagated shapes include `(1,s11,512)`,
`(1,2,s11+s14,128)`, and attention scores
`(1,4,s11,s11+s14)`. All 160 tensor-valued `call_function` nodes retained a
symbolic extent; the 20 static tensor nodes are parameter/buffer placeholders.
Batch, hidden size, intermediate size, head counts, and head dimension were
intentionally specialized constants. The exact inventory is in
`raw/node_shapes.tsv`.

Expressions such as `s11 + s14` map directly to `ClosedForm`. Bounds and the
four observed equality guards map to a Presburger/ISL parameter domain.
`ClosedForm` alone is not a guard language: comparisons and possible future
modulo guards belong in a separate constraint set. A guard such as
`s0 % 128 == 0` would be representable in ISL/quasi-affine constraints, not by
the current minimal arithmetic AST alone.

## Complete operator whitelist

All 30 distinct targets are listed below; counts by category are pointwise 12,
matmul 3, reduction 2, broadcast 2, concat 1, slice 2, transpose 1, other 7.

| Target | Category |
|---|---|
| `<built-in function add>` | pointwise |
| `<built-in function getitem>` | slice |
| `aten._assert_tensor_metadata.default` | other |
| `aten.add.Tensor` | pointwise |
| `aten.arange.default` | other |
| `aten.arange.start` | other |
| `aten.cat.default` | concat |
| `aten.chunk.default` | slice |
| `aten.contiguous.default` | other |
| `aten.cos.default` | pointwise |
| `aten.div.Tensor` | pointwise |
| `aten.gt.Tensor` | pointwise |
| `aten.linear.default` | matmul |
| `aten.masked_fill.Scalar` | pointwise |
| `aten.matmul.default` | matmul |
| `aten.mean.dim` | reduction |
| `aten.mul.Tensor` | pointwise |
| `aten.neg.default` | pointwise |
| `aten.outer.default` | matmul |
| `aten.pow.Tensor_Scalar` | pointwise |
| `aten.repeat_interleave.self_int` | broadcast |
| `aten.rsqrt.default` | pointwise |
| `aten.silu.default` | pointwise |
| `aten.sin.default` | pointwise |
| `aten.softmax.int` | reduction |
| `aten.sym_size.int` | other |
| `aten.to.dtype` | other |
| `aten.transpose.int` | transpose |
| `aten.unsqueeze.default` | broadcast |
| `aten.view.default` | other |

The seven `other` targets are not new mathematical TaskBody families:
metadata assertion and `sym_size` are frontend/guard operations; `view`,
`contiguous`, and `to` are layout/type operations; `arange` constructs RoPE
positions. They still require explicit Phase 1 lowering rules and must not be
silently dropped.

## Reproduction

```bash
docs/experiments/V_H/run.sh
cat docs/experiments/V_H/raw/report.json
```
