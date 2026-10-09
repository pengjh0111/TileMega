# Numerical contracts

verified: `results/MO_topk_dispatch_combine_cuda.json` seals 77 routing and
19 combine cases, three epochs, 50/50 fresh processes, five architectures,
zero spills/stack and zero-error memcheck/racecheck. Router partial logits
are already BF16-rounded. Top-k orders by decreasing logit, then increasing
expert index; selected FP32 softmax rounds its normalized weights once to
BF16. Stable expert grouping preserves token/rank order, including invalid
binding tails. Combine rounds each BF16 weight times BF16 expert output to
BF16, accumulates rank order in FP32, rounds once, then adds the residual
with a BF16 store. Row statistics consume that final stored value. Synthetic
histogram/dispatch and weighted-arrival callbacks are included; generated
MoE model integration and real-weight gates remain pending.

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


verified: explicit LayerNorm uses centered FP32 variance and applies BF16 gamma
and beta before one BF16 output store. Embedding sum preserves the two ordered
BF16 stores in `(word + token_type) + position`. Optional output statistics
consume those rounded outputs. Layout conversion copies external NCHW input
into the padded NHWC interior and never writes spatial halo. Standalone tests
are sealed in `results/DN_primitive_norm_embedding_layout_cuda.json`.

verified: native forward fixtures compare convolution against an independent
FP32 dot product, round its stored result to BF16, then compare LayerNorm
against centered CPU FP64 statistics. They check the intermediate values, final
outputs, statistics and L1/L2 bit equality with the original tolerance. Missing
tile-owned combiner flags caused the original split-five failure; expectations
are unchanged. Native paged sanitizer completion remains a separate gate.

verified: DNN checkpoint recipes pass 12 CPU checks, including independent
module expressions and the existing host CuTe page-layout oracle
(`results/DN_weight_recipes_host.json`). BN statistics retain their checkpoint
precision; the folded convolution weight is rounded to BF16, and its separate
bias remains FP32. Linear and packed QKV bias remain FP32. QKV packing is
head-major `[Q_head, K_head, V_head]`; SimpleGate interleaves channel chunks
without changing their values.

verified: deferred LN stores `W' = bf16(W*gamma)`, uses FP32
`u = sum(float(W'))` and `v = W.float() @ beta.float() + bias.float()`.
Summing the stored weight makes the mean correction cancel constant input
rows. Weight folding still changes the rounding relative to explicit LN;
model-level G-DNN remains required. Nested page packing preserves these
recipes, and the loader binds the dtype declared by each buffer.

verified: encoder attention uses noncausal MHA with D=64, FP32 QK/online
softmax and FP32 running output. Probabilities round to BF16 at the PV MMA
input; output rounds once to BF16. The independent FP64 QK/softmax/PV oracle
covers S=128/384/512, query tiles 64/128 and key-padding cases (including only
one valid key). All-true masking is bitwise equal to the unmasked body.
`results/DN_encoder_attention_cuda.json` seals 24 cases, three epochs, 50/50
fresh processes, five architectures, no spills/stack and zero-error sanitizers.
verified: the CG-generated masked encoder region also passes 50/50 fresh
processes with repeated L1/L2 bit equality against PyTorch FP32 SDPA, five
architecture builds and zero-error sanitizers. Its shared-library artifacts
have no spills and 5/5 stack frames (`results/DN_generated_encoder_cuda.json`).
The generated global-pool region passes the same checks against FP64 segmented
means (`results/DN_generated_global_cuda.json`). Full model gates remain open.

verified: fused dw→pw computes depthwise taps in FP32, applies its finite
bias/scale/activation chain, then materializes BF16 in shared memory before
the pointwise MMA. Each pointwise column tile recomputes the same complete
depthwise tile. Optional depthwise global stores have one designated column
owner. Dense/tiled/paged pointwise paths agree bitwise in the local fixture;
48 cases, three epochs, 50/50 fresh processes and zero-error sanitizers are
sealed in `results/DN_dwpw_fused_cuda.json`. Full-model fusion remains pending.

verified: CG-generated depthwise and SimpleGate regions preserve all halo and
channel-padding canaries. Their per-image channel sums use stored BF16 outputs.
The generated depthwise→global-pool region checks segmented partial sums and
means against FP64 reduction, with 50/50 fresh processes, five architectures
and zero-error sanitizers. Receipts are `results/DN_generated_depthwise*_cuda.json`;
all artifacts have zero spills and 5/5 stack frames.

verified: finite GEMM chains use the same mapped residual reads and output
addresses in direct and split-K builds. Side statistics and channel sums are
written only by the final split combiner and describe the stored BF16 values.
Geometry-dependent partition axes are materialized after selecting TM/TN;
global-pool consumers read exactly the tiles overlapping each image. Dense,
NCHW and PixelShuffle generated fixtures each pass 50/50 L1/L2 processes
(`results/DN_generated_epilogue_*_cuda.json`).
