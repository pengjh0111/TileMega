# BE-9 — operator coverage on the two anchored models

Every operator the anchored Llama-3.2-1B and Qwen3-1.7B graphs execute, the
body that runs it, the path that body takes, and whether any cross-thread
reduction is still done by one lane. The stage kinds are read off the
generated sources (`E2E_REAL/llama/auto.cu`: 1 `kEmbedding`, 33 `kRMSNorm`,
32 `kRoPE`, 32 `kKVAppend`, 16 `kAttention`, 16 `kElementwise`, 115 `kGemm`;
the Qwen3 cut adds `kQKNorm`).

| operator | TaskBody | implementation path | arch branch | serial reduction left |
|---|---|---|---|---|
| GEMM | `GemmStageTaskBody` | CUTLASS `CollectiveMma`, `MainloopSm80CpAsync`, `SM80_16x8x16_F32BF16BF16F32_TN` atom, FP32 accumulator | `caps.kBf16CollectiveBuilder` selects the TMA warp-specialized `CollectiveBuilder` on sm_90/sm_100; sm_80/sm_89/sm_120 take the cp.async multistage collective | no |
| split-K combine | `GemmCombineTaskBody` | per-output-element accumulation in FP32, one thread per element over the peers | none needed | no — the loop is per thread over its own element, parallel across elements |
| attention | `AttentionChunkTaskBody` | running max and running exponential sum over the key sequence, both CTA reductions on `__shfl_xor_sync` (`WarpReduce.cuh`), FP32 internals | `caps.tma` would select a TMA K/V load; not taken on sm_89 | **no — was three serial scans on lane 0 before R8** |
| attention, chunked | `AttentionPhasedTaskBody` | same reductions in the `kNormalize` phase | as above | **no — was `if (threadIdx.x != 0) return;` plus three scans** |
| attention combine | `AttentionCombineTaskBody` | per-element accumulation over chunks, parallel across the head dimension | none needed | no |
| RMSNorm | `RMSNormTaskBody` | sum of squares as a CTA shuffle reduction, FP32, `rsqrtf` | none needed | **no — was a shared-memory tree costing log2(threads) barriers** |
| per-head Q/K norm | `QKNormTaskBody` | delegates to `RMSNormTaskBody::RunRow` over one head | none needed | no |
| RoPE | `RoPETaskBody` | elementwise over half-dimension pairs, FP32 angle | none needed | no reduction to remove |
| KV append | `KVAppendTaskBody` | elementwise copy into the cache | none needed | no reduction to remove |
| residual add | `AddTaskBody` | elementwise | none needed | no reduction to remove |
| elementwise / SwiGLU | `ElementwiseTaskBody` | elementwise, FP32 internals | none needed | no reduction to remove |
| token embedding | `EmbeddingTaskBody` | one row gather per token, parallel across the row | none needed | no reduction to remove |
| fused RoPE+KV | `FusedRoPEKVTaskBody` | the two elementwise bodies in one task | none needed | no reduction to remove |
| fused GEMM | `FusedGemmTaskBody` / `FusedGemmStageTaskBody` | the same CUTLASS candidate as `GemmStageTaskBody`, with the fused epilogue | as GEMM | no |

✅ **Verified: no naive implementation is left on either anchored model.**
Every body either goes through a CUTLASS collective, or performs its
cross-thread reduction with `__shfl_xor_sync`, or has no cross-thread
reduction to perform. The four bodies whose reductions were rewritten this
round are marked in bold above.

## The two bodies that are not on this list

`GemmTaskBody` and `GemmSplitKTaskBody` are counted by §1 as GEMM-family files
with zero CUTLASS references. They are **not reachable from either model**:
`ModelHarness.cuh` instantiates twelve bodies and neither is among them.
`GemmTaskBody::Run` writes `context.iteration` into the output — it is the
"minimal executable ABI" placeholder from an early phase, and
`GemmSplitKTaskBody` is `: GemmTaskBody<Arch, Stages>` with nothing added.
They are left alone rather than converted: converting a placeholder no model
runs would produce coverage on paper and nothing on the device.

`MoERouterTaskBody` is the same case — a stub with a single `threadIdx.x == 0`
store, not dispatched by the harness, and no anchored model has an MoE layer.

## Vectorization

BE-4 asks for vectorized access "по alignment". The elementwise bodies already
stride by `blockDim.x` over contiguous rows, which coalesces into full
transactions at `ModelElement` width; none of them was changed for this round,
and no wider vector type was introduced. Recorded as it stands rather than
claimed: the reductions were the measured defect (`AttentionChunkTaskBody`'s
127 idle threads), and widening loads is a separate change with its own
alignment preconditions.
