# DM-1 working report

Status: **incomplete**. Shared infrastructure and DNN primitive integration are in progress.
Export coverage and primitive checks do not imply model correctness.

## Scope and provenance

- stated: the user removed final performance testing on 2026-10-09. Implement the full
  extensions and verify execution/correctness; no latency matrices or speed claims.
  `scope_update.json` records the override. Shared GPU lock and numerical gates remain.
- verified: prompt `/root/Prompt/TileMega_DM1_prompt.md`, SHA256
  `6ecaa5be8d5157937a959baeabac6b5497397c2abd56b25cf5a70e4b6f56e828`.
- verified: companion `/root/Prompt/DNN_MoE_plan.md`, SHA256
  `ce288acf9c9557cf98956eb201ffc10a7c7a5bef9b12787c12e41f36275d01ce`.
- verified: startup `origin/tilemega` and reference HEAD
  `76beaea5e2d66e3311b36d020f470c4f016406d0`.
- verified: isolated branch/worktree `dnn-moe`, `/root/dm1_work/development`;
  reference `/root/dm1_work/reference`; scratch `/root/dm1_work/integration-stage`.
- Final HEAD and commit count: pending completion. M0: origin unchanged, no merge;
  M1/M2 have not been reached. Unpublished R14 changes have not been merged.
- verified: GPU lock `/root/r14_work/gpu.lock`; build `build-dm`;
  cache `/root/.cache/tilemega-dm`. No performance measurements have been taken.

## Implementation ledger

| Item | State and evidence |
|---|---|
| Phase 0 | verified: R13 framework, preregistration, reference bank, five model exports and fixtures committed |
| CI-1 | verified: typed arguments, CPU constants/shape bindings, legacy bridge compatibility (`bridge.md`) |
| CI-2 | verified: DM descriptors, finite epilogue chains, dense/page/split-K dispatch, narrow tile families (`descriptors.md`, `numerics.md`) |
| CI-3 | verified: forward ABI, native CG CLI plans and five-target builds; exported DNN/MoE model entry pending (`forward.md`) |
| CI-4 | verified: exact window/table/WAR-WAW foundations, virtual capacity/counting, native thresholds; memory-planner and body integration pending (`analysis.md`) |
| CI-5 | verified: synthetic waits/binding/LA and native forward/prefill PageStream; real model dispatch remains incomplete (`synchronization.md`) |
| CI-6 | verified: fitting/profile schema and streaming collection foundations; checkpoint downloaded and real routing collection in progress; body fits pending |
| CI-7 | verified: class/candidate, exact projection, issued convolution K, conditional/live-row and histogram pricing foundations; profile consumers/structural search pending |
| DN-1 | verified: six real-weight upstream exports, before/Core fixtures, source FQN/dtype preservation and weight recipes. These are not model gates |
| DN-2/3 | verified: automatic primitive DNN L-sem and exact ownership/halo oracles; exported-model planning and remaining operators pending |
| DN-4 | verified: im2col operands and dense/tiled/page TaskBodies, small channels, tails, stride/dilation, issued split-K. Native ABI primitives verified; complete models pending |
| DN-5/6 | verified: staged depthwise/SimpleGate and window/global pooling bodies; generated pool pipeline. dw→pw body verified; fused planning, SCA and complete model integration pending |
| DN-7 | verified: explicit LayerNorm and ordered BF16 embedding sum primitives; native/generated LN stages. Deferred LN/model integration pending |
| DN-8/9/10/11 | verified: encoder attention body and layout conversion; generated encoder integration in progress. Full chain/model numerics, memory reuse and model CLI pending |
| MO-1 through MO-9 | verified: checkpoint transformations and expert streaming recipes; real L12/E16 checkpoints generated. Model/region integration and end-to-end gates pending |

## Tables and gates

| Table | Current evidence |
|---|---|
| T1 | Eight LLM plans preserve CUDA bytes, ptxas resources, SASS and 64-step tokens; full G-REG remains false |
| T2 | Five models plus masked BERT export inventories verified; execution coverage pending |
| T3 | Exact access, virtual/counting and convolution issued-K proofs verified at unit/CG level |
| T4 | Finite epilogue, GEMM, convolution and normalization/embedding/layout primitive numerics verified; complete model geometries/chains pending |
| T5 | G-DNN has not passed for any complete model |
| T6 | Omitted under the user's no-performance-testing override |
| T7 | Timing attribution omitted; full mechanism/model correctness remains pending |
| T8 | G-MOE and real-weight end-to-end paths (a)(b)(c) have not passed |
| T9/T10 | Performance measurements omitted under the user override |
| T11 | Measured compile-scaling matrix omitted under the user override |
| T12 | Per-artifact resources retained in identity receipts; calibration/model resource coverage remains incomplete |

### T1: regression

verified: latest `results/T1_CI7_conv_counted.json` matches all eight generated
CUDA/resource/SASS/token comparisons. Candidate native suite was 149/151;
reference was 94/97. Candidate CG diagnostic failure was repaired and rebuilt
(`results/CI7_cg_diagnostic_prefix_host.json`, `CI2_conv_runtime_host.json`).
The independent-attention mismatch also exists on the unchanged reference.
Reference target schema/missing norm binary failures remain recorded.
**G-REG is false** until the full required suite passes; queue completion is not gate success.
Prior checkpoint history is retained in T1 receipts and `analysis.md`.

### T2: upstream inventories and assets

| Model | Before calls | Core ATen calls | Appendix C differences |
|---|---:|---:|---|
| ResNet18 | 69 | 92 | none among stated counts |
| MobileNetV1 | 84 | 113 | none among stated counts |
| MobileNetV2 | 153 | 206 | none among stated counts |
| NAFNet | 1605 | 1622 | none among stated counts |
| BERT | 296 | 970 | addmm 61/73; bmm 36/24; view 276/264 |
| BERT masked | 306 | 987 | extra correctness variant |

verified: twelve BERT output-dense linears decompose to bmm+bias, accounting
for the observed difference; the stated expectations are retained.
All twelve archives reload with matching non-getitem inventories and BF16 weights;
extra tuple-field getitems are separately recorded (`results/export_roundtrip.json`).
BERT's shape-only mask evaluates all true at B=1/8/32/64, including the
2^20-element boundary (`results/bert_mask_provenance.json`).
verified: official SIDD 1280 RGB blocks pass LMDB/CRC/key checks; ImageNetV2
and WikiText-103 identities are recorded (`results/data_integrity.json`).
NAFNet, ResNet18 and MobileNetV2 pretrained states load strictly. BERT and
MobileNetV1 weight identities are recorded; complete model correctness is pending.

### T3/T4/T12: new primitive evidence

verified: `results/CI2_conv_operand_matrix_cuda.json` records 13 tile configurations,
650/650 fresh numerical processes and 65 final architecture artifacts. A unique
sm_100 recovery build replaces an interrupted overwritten artifact; its old failed
queue and mismatched identity are retained and are not credited.
All 65 final operand artifacts have zero spills; two have nonzero stack frames.

verified: `results/CI7_dense_pipeline_cuda.json` records dense/tiled canonical
100/100 processes, 12 neighboring configurations, five architectures and four
zero-error sanitizers. The prior version passed numerically but produced 144
racecheck errors per layout. A compute-warp join before scratch reuse repairs
that race; the failed evidence is retained.

verified: `results/CI2_conv_task_cuda.json` records 50 architecture/configuration
artifacts, canonical row-major/tiled 50/50 each, eight neighbor checks, four
zero-error sanitizers, and independent FP32 references for dense/page/split-K
entries. **12/50 artifacts spill and 50/50 have stack frames**; every resource
line is retained. This is standalone TaskBody coverage, not the complete DNN gate.

verified: `results/DN_primitive_norm_embedding_layout_cuda.json` records 216
primitive cases per process and 50/50 fresh processes, five architectures,
zero spills/stack and zero-error memcheck/racecheck. It covers centered-variance
LayerNorm, ordered BF16 embedding additions/statistics, NCHW conversion,
channel padding, immutable 0/-inf halo and end canaries.

verified: `results/CI2_conv_runtime_host.json` records 6/6 native host/CG checks
of issued K, descriptor legality and ownership. Original illegal C24/TK fixtures
and a stale CG diagnostic binary are retained; expectations were not loosened.
verified: `results/DN_primitive_stage_host.json` records 6/6 native host checks,
including int64 operand validation, LayerNorm epsilon and finite stage dispatch.
verified: `results/DN_primitive_semantic_host.json` records 8/8 host checks,
automatic layout/conv/LN lifting, two-way enumeration/containment, issued split-K
and embedding binding request counts. Native ABI integration is being checked
in immutable `runs/dm1-dnn-runtime-*`.
The first runtime queue did not execute numerics because of a misplaced descriptor
field; its failure is retained. A later paged build exposed an unreachable legacy
SwiGLU instantiation at TN16; the DM branch is separated and rechecked.
verified: split-five fixtures omitted tile-owned combiner flags and left output
columns unwritten. The corrected fixture passes all six native memchecks and
numerical paths; malformed DM split ownership is rejected. Prior paged racechecks
fail on page generation and cross-task scratch/page reuse. The frozen v5 repair passes 300/300 fresh processes across six configurations,
30 architecture builds and 12 zero-error sanitizers (`results/DN_native_forward_cuda.json`). The full
model synchronization gate remains pending.

verified: `results/CI7_moe_histogram_pricing_host.json` records 5/5 native checks,
650 histogram cases (472 empty contributions) and an equal-mean contrast that
rejects mean-only costing. Real profile attachment, cache identities and the
MoE DRAM floor remain pending. Synthetic coefficients are not body calibration.

verified: real-weight exports reload for all six variants, retaining original source
FQNs/dtypes (`results/DN_real_exports.json`). DNN packing passes 12 independent
recipe checks (`results/DN_weight_recipes_host.json`).

verified: actual CG-generated layout→conv→LN and layout→conv→pool→LN shared
libraries each pass 50/50 fresh processes, five architecture builds, repeated
L1/L2 bit equality and zero-error sanitizers (`results/DN_generated_primitive_cuda.json`,
`results/DN_generated_pool_cuda.json`). Each has zero spills and 5/5 stack frames.
These small pipelines are not the five upstream model gates.

verified: staged depthwise covers 58 geometries/SimpleGate chains and encoder
attention covers 24 noncausal/masked geometries, each with three epochs, 50/50
fresh processes, five architectures, zero spills/stack and zero-error sanitizers
(`results/DN_depthwise_cuda.json`, `results/DN_encoder_attention_cuda.json`).
Eight of the ten new body kinds have local numerical evidence;
MoE top-k/dispatch and MoE combine remain. Pool/global L-sem and primitive
solver traits pass 8/8 host checks (`results/DN_pool_primitive_pricing_host.json`).

verified: CG-generated encoder attention and global pooling each pass 50/50
fresh L1/L2 processes, five architecture builds and zero-error sanitizers
(`results/DN_generated_encoder_cuda.json`, `results/DN_generated_global_cuda.json`).
Both have zero spills and 5/5 stack frames. Exact ownership, task-count
projection and forward batch binding pass 9/9 host checks
(`results/DN_encoder_global_host.json`). The pre-launch batch-binding failure
is retained; numerical criteria are unchanged.

verified: fused dw→pw passes 48 cases covering three repeated epochs and dense,
tiled and paged weights in 50/50 fresh processes, five architecture builds and
zero-error sanitizers (`results/DN_dwpw_fused_cuda.json`). All five artifacts
have zero spills and stack frames. Generated depthwise/per-image pool host
integration passes 15/15 checks and a separate ownership-padding oracle
(`results/DN_depthwise_host.json`). Generated depthwise, SimpleGate and
depthwise→per-image pool libraries each pass 50/50 fresh L1/L2 processes,
five architectures and zero-error sanitizers (`results/DN_generated_depthwise_cuda.json`,
`results/DN_generated_depthwise_gated_cuda.json`,
`results/DN_generated_depthwise_pool_cuda.json`). Each has zero spills and
5/5 stack frames. Pre-launch Python import failures are retained.

### Synchronization coverage

verified: mapped GEMM epilogues preserve dense, NCHW and PixelShuffle outputs
across split-K finalization. Each generated library passes 50/50 fresh L1/L2
processes, five architectures and zero-error sanitizers, with zero spills and
5/5 stack frames (`results/DN_generated_epilogue_{dense,nchw,shuffle}_cuda.json`).
Tile side-storage geometry and exact segmented reads pass 9/9 host checks;
the 24-case epilogue proof and side-store CG emissions also pass
(`results/DN_tile_storage_host.json`, `DN_epilogue_semantics_host.json`,
`DN_gemm_sides_host.json`). The generated side-store queue remains unsealed.

verified: original-weight MoE checkpoint tools pass four host tests, including
unchanged HF loading; expert streaming/deduplication and prior weight regressions
pass 16 tests (`results/MO_checkpoint_tools_host.json`, `MO_expert_weights_host.json`).
The complete 18,867-tensor checkpoint has 61,064,245,248 BF16 bytes. L12/E128
and L48/E16 transformations pass index checks (`results/MO_real_checkpoints_host.json`).
These are asset/packing checks; neither generated-model end-to-end gate has passed.

verified: shared primitives and synthetic table/counted/threshold stages have
50-process receipts; native nonpaged split-K LA and attention LA each pass
100/100, binding PageStream synthetic dispatch 100/100, native paged
forward/prefill 50/50 each (`synchronization.md` and CI5 receipts).
These do not discharge the corresponding real DNN/MoE body/model paths in §8.A.
**The full §8.A checklist is incomplete.**

## Q1–Q7 and deviations

- Q1 — verified: upstream export and the tested primitives work; five complete
  exported-model pipelines and G-DNN remain unverified.
- Q2/Q3 — stated: latency comparisons and performance attribution are omitted
  under the user override. Implementation/correctness requirements remain.
- Q4/Q5 — stated: MoE performance comparisons are omitted; template/binding,
  placement and model correctness implementation remains required and incomplete.
- Q6 — verified: checkpoint index/schema foundations exist; real-weight (a)(b)(c)
  and the complete-model readiness script have not passed.
- Q7 — stated: compile-scaling measurement is omitted under the user override.
- verified: earlier failed builds/numerics/sanitizers are preserved. No numerical
  thresholds were changed to accommodate implementation errors.
- verified: nvcc 12.8.93 miscompiles an unbraced `else for` in a capturing
  `if constexpr` lambda. The reduced counterexample and braced repair are sealed
  in `results/CI2_nvcc_constexpr_scope.json`; legacy LLM code is preserved.

## Remaining work

Complete native DNN stage integration, DNN planning/L-sem and all remaining bodies,
then model correctness gates. Complete CI7 profile/search integration and check G-REG.
Proceed through MO-1..MO-3, M1, MO-4..MO-9 and correctness gates in the prescribed
merge order. M2/freeze, final documentation and branch push remain pending.
No performance matrices will be executed.
