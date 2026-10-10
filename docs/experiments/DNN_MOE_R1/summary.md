# DM-1 working report

Status: **incomplete**. Generated DNN/MoE paths execute; full-graph analysis and joint solver integration remain.
Export coverage and primitive checks do not imply model correctness.

## Scope and provenance

- stated: the user removed final performance testing on 2026-10-09. Implement the full
  extensions and verify execution/correctness; no latency matrices or speed claims.
  `scope_update.json` records both overrides. On 2026-10-10 the user also removed
  real-weight/dataset gates and repeated-process matrices; one fixed synthetic
  numerical smoke per execution path remains. The shared GPU lock remains.
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
| CI-3 | verified: forward ABI and native CG CLI; DNN exported-model entry implemented; MoE forward region CLI entry/codegen implemented; complete DNN gates and full decoder checks pending |
| CI-4 | verified: exact windows/tables, WAR/WAW, virtual capacities/counting and native thresholds; full fused/reused MobileNetV1 and full B16 MoE prefill now generate; bounded symbolic interval proofs retain exact fallback (`analysis.md`, `memory.md`) |
| CI-5 | verified: native forward/prefill pages and table/counted/binding primitives; generated MoE dynamic L2 control now passes fixed-input checks; generic pool/dispatch/counted-combine LA elision passes generated L1/L2/pages fixtures (`dm_last_arriver.md`) |
| CI-6 | verified: 48-layer routing profile/captures and separate rational expected-read cardinalities; production binding/unique-expert floor consumers and public profile identity are implemented; conditional occupancy flow and final-plan attribution pass host checks (`moe_structure_search.md`); empirical body fits omitted under the no-measurement scope |
| CI-7 | verified: narrow tile/class candidates, private DW compute pricing, virtual/histogram work foundations; shared packed-layout constraints now cover fixed/solver plans and capacity-limited CLI deployments; DNN per-pair fusion, per-edge LN, reuse and C′ selection now pass host search/codegen; MoE slot/group BM16/32/64/128 host search now passes; GEMV/LA structural pricing remains incomplete |
| DN-1 | verified: six upstream real-weight exports, before/Core fixtures and source-FQN recipes; assets are retained, no further real-weight/dataset checks are required |
| DN-2/3 | verified: five DNN before/Core plans and semantics, including masked BERT and NAFNet SG/SCA/shuffle; prior model discrepancies remain recorded |
| DN-4 | verified: im2col, small channels, tails, stride/dilation, split-K and paged B primitive execution (`conv.md`, T4 receipts) |
| DN-5 | verified: DW/PW fusion integrated into frontend, exact semantics, resource pricing, split-K and executors; nonpaged/paged fixtures and full MobileNetV1 fused fixed-input L1/L2 execution pass (`dwpw_fusion.md`) |
| DN-6 | verified: window/global pooling and SCA bodies/plans; cross-image pool LA executes after split-K; structural solver selection remains pending |
| DN-7 | verified: explicit LN, embedding sum and per-consumer deferred LN rewrite; generated synthetic execution passes; per-edge joint structural selection passes a two-edge host fixture; one combined C4/two-fusion/two-deferred-edge native B2 smoke passes (`DN_dnn_structure_native.json`) |
| DN-8/9 | verified: Tensor Core encoder attention, layouts and finite chains; independent primitive receipts exist; prior full-model numerical failures remain explicit |
| DN-10 | verified: memory planner, arena aliases, exact hazards and retained-allocation budget accounting; generated small reuse case passes; full fused MobileNetV1 reuse now generates and passes fixed-input L1/L2 execution (18 arena bindings, 13559552 B) (`memory.md`) |
| DN-11 | verified: public DNN export/build/run/check entry, architecture-only construction and separate fixed synthetic smoke receipts; baseline/timing work is omitted under user scope (`dnn_cli.md`) |
| MO-1/2/3 | verified: original-FQN exports/checkpoints, streaming expert packs, decoder/region plans, virtual/counting semantics; canonical shared-layout constraints and capacity-limited serving/full-entry integration have host evidence |
| MO-4/5/6/7 | verified: routing, gathered/indirect expert GEMMs, combine and QPerKV=8 execute in generated region/decoder fixtures; bound dense/paged GEMV and generic LA execute; GEMV pricing/search remains pending |
| MO-8 | verified: slot/group binding and dynamic L2 control execute; opaque control executes with full-stage/prefetch boundaries; joint slot/group and four BM families pass host search/codegen (`CI7_moe_binding_profile_host.json`) |
| MO-9 | verified: synthetic two-layer decoder paths execute; full B1/B16 decode/prefill generation, header/packing checks and allocation reports pass (`MO_full_four_plan_host.json`); full native execution remains unverified |

## Tables and gates

| Table | Current evidence |
|---|---|
| T1 | Eight LLM plans preserve CUDA bytes, ptxas resources, SASS and 64-step tokens; full G-REG remains false |
| T2 | Five models plus masked BERT export inventories verified; execution coverage pending |
| T3 | Exact access, virtual/counting and convolution issued-K proofs verified at unit/CG level |
| T4 | Finite epilogue, GEMM, convolution and normalization/embedding/layout primitive numerics verified; complete model geometries/chains pending |
| T5 | MobileNetV1/ResNet18 B=2 pass ImageNetV2; NAFNet B=2 passes SIDD. Other configurations and MBV2/BERT remain incomplete |
| T6 | Omitted under the user's no-performance-testing override |
| T7 | Timing attribution omitted; full mechanism/model correctness remains pending |
| T8 | G-MOE and real-weight end-to-end paths (a)(b) have not passed; full48 B1 dry-build/index checks pass |
| T9/T10 | Performance measurements omitted under the user override |
| T11 | Measured compile-scaling matrix omitted under the user override |
| T12 | Per-artifact resources retained in identity receipts; calibration/model resource coverage remains incomplete |
verified: complete synthetic two-layer QPerKV=8 decoder kernels execute in
nonpaged/paged decode and prefill. The paged-prefill fixed case passes independent
HF attention and MoE references on each native layer input, its final output head,
and same-binary L1/L2 bit comparison. Its independent HF whole-model token differs;
the original whole-decoder elementwise/token failures are retained. This is a
component numerical smoke under the user override, **not G-MOE**.
`python/tilemega/moe/check_decoder.py` records both component and whole-model results.

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
verified: complete upstream MobileNetV1 graph execution at B=2 passes 50/50
fresh processes, repeated L1/L2 bit equality and zero-error memcheck/racecheck.
Five architecture builds pass; all five artifacts spill and have stack frames.
Minimum cosine with the exported BF16 checkpoint promoted to FP32 is
0.9997538328 (`results/DN_mbv1_model_smoke_cuda.json`). Random-input smoke
does not replace the 1000-image G-DNN gate. The actual CLI replay passes;
its earlier timeout remains in `runs/dm1-dnn-cli-host-v1/events/mbv1.log`.
verified: BERT before/Core, masked/unmasked planning and lifting pass four
real-graph checks plus negative mask/attention/position contracts; plans have
87 stages and 49 packed GEMMs (`results/DN_encoder_model_plan_host.json`).
Native BERT smoke passes with minimum hidden/pooler cosine 0.99990898/0.99998534,
repeated L1/L2 equality and zero-error sanitizers; replay passes 50/50 fresh processes (`results/DN_bert_native_smoke.json`).
The original-FP32 WikiText gate fails: minimum token/pooler cosine
0.93862915/0.99785495. BF16 reference diagnostics also fail but do not relax
the gate (`results/DN_bert_dataset_failure.json`). No BERT G-DNN gate is claimed.
verified: complete NAFNet before/Core planning and lifting pass, with 335 stages,
190 GEMMs, 226 convolutions, 72 LN2d, 36 SCA and four PixelShuffle mappings.
Native CUDA is generated with target-bounded 99,840-byte workspace (sm_89);
execution reached the image-output check. Corrected PSNR smoke fails on the second image (13.6362 versus BF16 21.3903 dB);
the original thresholds remain fixed (`results/DN_naf_smoke_failure.json`). Official SIDD B=2 passes all 1280 blocks: PSNR(TM,FP32)=66.42515 dB; PSNR(TM,GT)=39.92491 dB (`results/DN_naf_sidd_dataset_cuda.json`). B=1 remains pending.
verified: MobileNetV2 fails the original 1000-image gate: mean cosine 0.99840518
versus 0.999; TM/BF16 top-1 agreement 0.969/0.971. The failed identified artifact
and unchanged thresholds are retained (`results/DN_mbv2_dataset_failure.json`).
The new FP32 epilogue BN factor improves MobileNetV2 to cosine 0.99867522,
still below 0.999; top-1 0.971 now matches the BF16 reference
(`results/DN_mbv2_bn_dataset.json`). The failure remains open.
verified: MobileNetV1's BN-factor artifact passes the original 1000-image
numerical gate at B=2: mean cosine 0.99997887 and TM/BF16 top-1 0.963/0.960
(`results/DN_mbv1_dataset.json`). This is one identified configuration;
other batches, paged paths and feature integrations remain unverified.
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
All ten new body kinds have local numerical evidence; model integration
remains incomplete. Pool/global L-sem and primitive
solver traits pass 8/8 host checks (`results/DN_pool_primitive_pricing_host.json`).

verified: MoE top-k/dispatch and combine pass 77 routing and 19 combine cases,
three epochs and 50/50 fresh processes, including deterministic grouping,
histogram/dispatch LA and weighted combine LA. Five architectures compile;
zero spills/stack and zero-error sanitizers are recorded in
`results/MO_topk_dispatch_combine_cuda.json`. Three original pre-kernel CUDA
allocation failures are retained beside the successful 50-process replay.

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

verified: ResNet18 B=2 passes 1000 ImageNetV2 images: cosine 0.99990601,
TM/BF16 top-1 0.992/0.983 (`results/DN_resnet_dataset.json`). Native execution
passes 50/50 processes, five architectures and zero-error sanitizers; all five
artifacts spill and have stack frames (`results/DN_resnet_native_smoke.json`).
The BN-factor MobileNetV1/V2 artifacts also pass 50/50 native processes and
five architectures; this does not turn MobileNetV2's dataset failure into a pass.

verified: SCA dense/im2col operands and six expert dense/tiled/page/split variants
each pass 50/50 processes, five architectures and zero-error sanitizers.
The wide dense SCA case spills on three architectures; expert variants do not.
Routing/combine wrappers and bound counted-row publication independently pass
50/50 processes each (`results/MO_stage_wrappers_cuda.json`,
`MO_counted_publication_cuda.json`). Generated MoE regions produce CUDA;
plan-count and flattened top-k layout failures are retained. Real layer-0 T=1
passes HF output/routing and sanitizers; slot T=17 and grouped T=513 paged
regions also pass initial numerics/sanitizers with nontrivial synthetic experts
(`results/MO_region_initial_native.json`, `MO_pages_initial_native.json`).
Full-depth semantics and four compact decoder CUDA emissions pass; execution remains
pending (`results/MO_decoder_semantic_host.json`). Full decoder recognition
and 48-fragment metadata composition pass; neither proves full decoder execution.

### Synchronization coverage

verified: mapped dense/NCHW/PixelShuffle epilogues retain their earlier 50/50,
five-architecture and sanitizer receipts in `results/DN_generated_epilogue_*`.
Tile side-storage and epilogue proof receipts remain in `results/DN_tile_storage_host.json`,
`DN_epilogue_semantics_host.json` and `DN_gemm_sides_host.json`.

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
- Q6 — verified: full B1/B16 decode/prefill generation and header/packing checks pass;
  the correctness-only script exists. Full native execution and real-weight (a)(b) remain unverified.
- Q7 — stated: compile-scaling measurement is omitted under the user override.
- verified: earlier failed builds/numerics/sanitizers are preserved. No numerical
  thresholds were changed to accommodate implementation errors.
- stated: BN uses original BF16 convolution weights plus FP32 scale/bias in the
  same TaskBody epilogue, avoiding a second weight quantization. This deviates
  from storing scaled BF16 weights; seven locked recipe checks pass, but the
  repaired MobileNetV2 model gate still fails the unchanged mean-cosine threshold.
- verified: an earlier CPU recipe check and one CUDA compilation ran without
  the GPU lock. The seven recipe checks were repeated under the shared lock;
  the unlocked CUDA compilation is not credited as final architecture evidence.
- verified: nvcc 12.8.93 miscompiles an unbraced `else for` in a capturing
  `if constexpr` lambda. The reduced counterexample and braced repair are sealed
  in `results/CI2_nvcc_constexpr_scope.json`; legacy LLM code is preserved.

## Remaining work

Finish GEMV/LA solver integration and full-model generated execution checks.
Combined DNN structural selection and fixed-input native execution now pass.
Complete full-model native build/check coverage. Preserve prior model discrepancies.
Default host CUDA invariance now passes 8/8 (`T1_default_cuda_host.json`); complete documentation and
merge checkpoints, then push the completed branch. No latency matrices,
real-weight/dataset gates or repeated-process matrices will be run.
