# Numerical contracts

verified: 1024 scalar cases and sixteen tile cases pass the original
`|err| <= 1.6e-2 + 1.6e-2*|ref|` criterion against PyTorch-generated fixtures.
The scalar and tile variants compile for sm_80/89/90/100/120. Only sm_89 runs.
Tile checks pass in 50/50 fresh processes; this validates tile-local shared
storage and barriers, not any inter-stage path listed in §8.A.

Each generated step specializes its operation, activation, pairing unit,
residual map and input/output rounding. Materialization to BF16 uses
round-to-nearest-even. FP32 steps retain their intermediate values until the
next declared rounding point; final store rounding is separately declared.
The old five serving epilogues retain their original code path.

| Operation | Arithmetic and storage contract |
|---|---|
| bias / scale | FP32 addition / product after declared input rounding |
| ReLU / ReLU6 | FP32 clamp, then declared output rounding |
| GELU | erf or tanh expression selected at compile time |
| tanh / SiLU | FP32 transcendental expression, then declared output rounding |
| residual | current value times optional FP32 channel scale, then residual addition |
| SimpleGate | paired values rounded as declared, product, then output rounding |
| SwiGLU | paired BF16 gate/up; SiLU gate rounded to BF16 before product |
| deferred RMSNorm | FP32 accumulator times inverse RMS |
| deferred LayerNorm | FP32 `rstd*(acc-mu*u)+v` |
| residual LN | FP32 LN reconstruction, BF16 materialization, then residual addition |

verified: deferred LN's value oracle is explicit PyTorch LayerNorm followed
by Linear. The tile fixture supplies FP32 linear accumulations and statistics
of stored BF16 inputs. This tests epilogue algebra; BF16 folded-weight packing,
Tensor Core accumulation and model-level accuracy remain separate gates.

Row statistics and channel partial sums consume values rounded as the final
store contract requires. Row statistics retain FP32 sum and squared sum for
each column tile. Channel sums are segmented at image boundaries, including
tiles that cover more than one image. Top-k and argmax use stored logits;
ties select smaller global indices. Split-K partial side stores retain FP32.

verified: independent PyTorch permute/pixel-shuffle/indexing oracles cover
NCHW, padded NHWC with halo, pixel shuffle with padded channels, and row
scatter. Untouched halo and padding entries must equal their poison sentinel
exactly. No numerical tolerance is applied to those entries or top-k indices.

Gate results stay in their original shared-memory gate slots. Up slots remain
read-only while other lanes compute products; later steps and stores use the
same compact logical channel map. verified: this tile path passes 50/50 fresh
processes, with no spills in any of the five compile targets. sm_89 tile
variants use 40–64 registers and 4224 B shared memory. Per-kernel resource
rows and binary/input hashes are in `results/CI2_epilogue_tiles.json`.

verified: four independent PyTorch linear references exercise BF16 GEMM
accumulation through the generated-chain dispatch, dense row stride/offset,
bias/ReLU, SwiGLU, pixel-shuffle residual and scale/ReLU6. Each runs unsplit
and with two/three FP32 split-K chunks. M/N tails, poisoned halo and padding,
and per-tile row statistics retain their original assertions. Dense and page
paths with the same geometry must be bitwise equal. Five targets compile;
sm_89 passes 50/50 fresh processes with no spills. Kernel rows are in
`results/CI2_gemm_chains.json` (sm_89: 64/80/116 registers). This is local
GEMM/combiner and page-ring evidence, not a full paged plan or §8.A gate.
