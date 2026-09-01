# E2E — L0 to L0.5 to L1

> **Handwritten reference baseline.** This file and `e2e.cu` are retained for
> differential comparison. The product path is the CG-driven generator in
> [`../E2E_GEN/`](../E2E_GEN/result.md).

Evidence labels: ✅ executed/observed; ⚠️ limited scope; ❌ conjecture.

## Result

✅ The entry point is V-H's saved, real two-layer Llama `ExportedProgram`,
not a handwritten coupling graph. `prepare_e2e.py` turns its 179
`call_function` nodes into task spaces, its 222 tensor-use dependencies into
couplings, copies the symbolic parameter domain, rejects operators absent from
the V-H whitelist, and emits `fixture/e2e_graph.json`. The fixed granularity
`g` is recorded there; no Solver claim is made.

The numerical ladder passed:

| Comparison | Mismatches | Maximum absolute error | Result |
|---|---:|---:|---|
| L0.5 versus PyTorch L0 | 0 | `1.5497208e-6` | ✅ |
| L1 versus L0.5 | 0 | `0` | ✅ bitwise identical |

The tolerance was `atol=rtol=3e-5`. L1 passed 50/50 fresh processes with no
mismatch, error, or timeout.

## Lowering exercised

- L0 is the loaded `ExportedProgram.module()` on CPU with seed 20260901.
- L0.5 launches one CUDA kernel per one of 24 semantic stages; stream order is
  the only inter-stage synchronization.
- L1 executes the same 24 compile-time-bounded stages in one kernel and uses a
  global event barrier after every stage. Every producer thread fences before
  CTA convergence, one thread publishes, one thread polls with backoff, and
  the CTA reconverges after acquire.
- All 14 linear projections call CUTLASS's collective directly from the custom
  kernels. The FP32 SIMT configuration selects `MainloopSm80CpAsync<3>`—the
  same direct SM80 cp.async adapter family validated in V-B—without
  `GemmUniversal`. Residual additions use the collective epilogue's `beta=1`
  path.
- RMSNorm, RoPE/cache append, attention, softmax, and SwiGLU are intentionally
  naive TaskBody implementations; performance is not claimed for them.

## L1 resources and launch

| Quantity | Observed value |
|---|---:|
| Block size | 256 threads |
| Registers | 168/thread |
| Explicit shared-storage union | 49,536 bytes |
| Spill stores/loads | 0 / 0 |
| Active CTAs/SM | 1 |
| SMs from `TargetSpec::Probe()` | 128 |
| Grid | 128 CTAs |

The launch formula is
`grid = num_sms * ActiveBlocksPerSM(l1_kernel, block, sizeof(TaskSmem))`.
Unlike F-4/F-9's streaming wait graphs, a full-grid barrier after every stage
requires every launched CTA to be resident, so L1 deliberately uses the
resident capacity itself (§8.7), not an arbitrary larger grid.

The union has five task-storage members: RMSNorm, attention, pointwise, GEMM,
and RoPE. Its 49,536-byte size is the maximum collective storage, not a sum.
No member pointer escapes the dispatch lifetime, so the F-8 degeneration is
not triggered.

## Frontend issues exposed

- Reusing one `Dim` did not force one symbol name; KV cache equality is in
  executable guards and must be imported as constraints.
- KV cache is explicit input/output state. Dense `cat` is representable, but
  the frontend must preserve the old/new interval boundary.
- `view`, `contiguous`, transpose, unsqueeze, chunk/getitem, and
  `repeat_interleave` are graph structure, not disposable noise. They affect
  access/layout relations even when they do not require separate compute.
- `_assert_tensor_metadata`, `sym_size`, Python scalar add, and dtype conversion
  require frontend-only lowering rules.
- This graph contains no data-dependent control flow. Such control flow was
  therefore not validated by this experiment.

All 30 observed targets were present in the V-H whitelist; the frontend
reported no outside-whitelist operator. These observations drove the C++
importer now validated in E2E_GEN; the old `prepare_e2e.py` semantic adapter is
no longer the product path and is used only to reproduce binary fixtures.

## Performance baseline

Across the retained 50 fresh-process logs, the medians were:

| Path | Kernel time |
|---|---:|
| L0.5 | 1.091584 ms |
| L1 | 1.078272 ms |
| L1 / L0.5 | 0.987283 |

Thus L1 was about 1.27% faster in this correctness-sized workload. This is only
a baseline; most non-GEMM bodies are naive and launch/context noise is large.

## Recorded corrections

The first correct prototype used scalar FP32 GEMMs and passed 50/50, but it did
not satisfy the requested V-B reuse and was replaced. The first CUTLASS attempt
used the wrong B layout tag and produced 6,143 mismatches; CUTLASS presents B
to the collective in logical `(N,K)` order, so `ColumnMajor` is the tag that
matches PyTorch's contiguous `[N,K]` weights here. Finally, passing the large
parameter object by value produced a 2,592-byte thread stack frame and a
5.76-ms L0.5 time. Passing a device-side parameter pointer reduced the frame
to 32 bytes and restored the reported baseline. These failed/intermediate
numbers are not counted as validation passes.

## Reproduction

```bash
docs/experiments/V_H/run.sh
docs/experiments/E2E/run.sh
cat docs/experiments/E2E/raw/fresh_process_summary.txt
cat docs/experiments/E2E/raw/timing_median.txt
```
