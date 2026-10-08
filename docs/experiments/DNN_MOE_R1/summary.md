# DM-1 working report

Status: **incomplete, Phase 1 in progress**. This file is an evidence ledger, not a claim
that DNN or MoE execution is implemented.

## Provenance

- verified: prompt `/root/Prompt/TileMega_DM1_prompt.md`, SHA256
  `6ecaa5be8d5157937a959baeabac6b5497397c2abd56b25cf5a70e4b6f56e828`.
- verified: companion `/root/Prompt/DNN_MoE_plan.md`, SHA256
  `ce288acf9c9557cf98956eb201ffc10a7c7a5bef9b12787c12e41f36275d01ce`.
- verified: startup `origin/tilemega` and reference HEAD
  `76beaea5e2d66e3311b36d020f470c4f016406d0`.
- verified: branch `dnn-moe`, isolated worktree
  `/root/dm1_work/development`; detached reference `/root/dm1_work/reference`.
- Final HEAD and commit count: pending completion. M0: checked, origin unchanged,
  no merge (`results/M0_probe.json`); M1/M2: not reached.
- verified: both production compilers build in their own `build-dm`.
- verified: GPU lock `/root/r14_work/gpu.lock`; R13 guard policy copied
  unchanged, SHA256 in `start.json`. No performance measurements taken.

## Implementation ledger

| Item | State and evidence |
|---|---|
| Phase 0 framework | verified: copied R13 scripts, 12 framework/selection/identity/recipe tests pass |
| Preregistration | verified: §3 retained verbatim in predictions; solver winners sealed before control-arm audit |
| Reference bank | verified: eight baseline plans and 64-step smokes sealed; candidate CUDA/resources/SASS/tokens match all eight plans |
| DN-1 inventories | verified: five upstream models and masked BERT exported, before/Core fixtures committed |
| DN-1 NAFNet source | verified: unchanged architecture/dependencies match upstream file SHA256; license preserved |
| DN-1 weights/data | preparation only: NAFNet/ResNet18/MobileNetV2 strictly load; SIDD CRC passes; BERT/MobileNetV1 safetensors identities recorded |
| CI-1 | implemented transcription/constant/binding API; verification in `bridge.md`; forward build integration follows CI-3/DN-11 |
| CI-2 | verified descriptors, finite epilogue and GEMM/combiner dispatch (`descriptors.md`, `numerics.md`); new operand policies follow DN/MO bodies |
| CI-3 | verified core ABI/dimensions, five targets, 50/50 fresh processes per seq (1/128/4096); CLI/model integration pending (`forward.md`) |
| CI-4 | verified static access/metrics/table/WAR-WAW, exact bound encoding and virtual capacity/provenance; counted-write and memory-planner integration pending (`analysis.md`) |
| CI-5 | verified primitives (50/50), synthetic table/counted stages (150/150), paged forward/prefill cores (150/150), binding PageStream synthetic dispatch (100/100), nonpaged split-K LA (100/100); attention LA and new bodies pending (`synchronization.md`) |
| CI-6 | verified CPU fitting and streaming-routing foundations (5/5 each); native samplers, actual fits and real routing profiles pending (`calibration.md`) |
| CI-7 | verified table projection, geometry rebinding, GEMM access/conv classes and routing-cost validation; tile families, profile consumers, structural search and second-level selection pending |
| DN-2 through DN-11 | not implemented |
| MO-1 through MO-9 | not implemented |

## Tables and gates

T1: `results/T1_phase0.json` records the initial verified failure in both toolchains:
`norm_prologue_gemm_test.cu` lacks the RMSNorm header and calls a removed
`PagedGemmTaskBody::Run` overload. This is reproduced in the unchanged
reference. It prevented native-suite completion at that checkpoint. Later
repairs are recorded below; a production-plan sealing queue alone does not
count as passing G-REG.
verified: Phase 0 reference bank is sealed with SHA256
`0d7c170870af4c677daf14afdc9d7e6606f1192d014124fe9ab9f4853ce0641a`.
All eight reference artifact identities require the actual baseline HEAD and
an empty worktree diff. This completes reference preparation; it does not
resolve the native ctest failure.
verified: `results/T1_phase0_seal.json` records equality of CUDA bytes,
every ptxas resource context, SASS and 64-step tokens for all eight plans.
Its complete G-REG result remains false because ctest was not run after
the native test build failed.
verified: donor manifests name paged reduction handoffs `last_arriver`,
whereas baseline CLI requires policy `off` and unconditionally lowers those
reductions to last-arriver. Invalid direct-name and `auto` attempts failed
before compilation; their logs are retained. Corrected retry jobs preserve
the donor's fixed handoff implementation, not a disabled mechanism.

T2: `results/phase0_exports.json`, `results/T2_inventory_comparison.json`,
and `test/fixtures/dnn/*_ops.json` record **export coverage only**.

| Model | Before calls | Core ATen calls | Appendix C differences |
|---|---:|---:|---|
| ResNet18 | 69 | 92 | none among stated counts |
| MobileNetV1 | 84 | 113 | none among stated counts |
| MobileNetV2 | 153 | 206 | none among stated counts |
| NAFNet | 1605 | 1622 | none among stated counts |
| BERT | 296 | 970 | addmm 61/73; bmm 36/24; view 276/264 |
| BERT masked | 306 | 987 | extra correctness variant |

verified: BERT's twelve `attention.output.dense` linears decompose to bmm
and bias in these exports, accounting for the addmm/bmm difference. The
stated counts remain unchanged in the comparison. Forward plans must handle
both spellings and these observed targets.

verified: all twelve saved archives reload with identical non-getitem
operator inventories and BF16 floating weights. Core ATen deserialization
adds unused tuple-field getitems (BN, LayerNorm and maxpool); original counts
are retained and reloaded counts are separately recorded in
`results/export_roundtrip.json`. DNN archive loading registers HF's original
ModelOutput pytree definitions before loading BERT.
verified: all twelve unmasked BERT SDPA calls share the same shape-only
mask fragment, whose sole input dependency is `sym_size(token_type_ids,0)`.
Isolated CPU FX Interpreter evaluations at B=1/8/32/64 are all true, including
the specification's 2²⁰-element boundary (`results/bert_mask_provenance.json`).

T1 checkpoint CI2: verified CUDA bytes, ptxas resources, SASS and 64-step
tokens equal for all eight plans (`results/T1_CI2.json`). Full G-REG remains
false pending the complete native suite. Three initial smokes failed from
OOM while loading weights; their fresh-process retries pass without changing
the plans or token criteria.

T4/T12 subcomponent evidence: 1024 scalar cases and sixteen epilogue tile
cases pass against PyTorch, with five-target compilation and no spills.
Tile-local checks pass in 50/50 fresh processes; these do not count as §8.A
inter-stage synchronization evidence (`results/CI2_epilogue_tiles.json`).

T4/T12: verified finite-chain GEMM/combiner dispatch for four independent
linear references, dense/page paths, split-K=1/2/3, M/N tails and mapped
writes. Five targets compile with no spills; sm_89 passes 50/50 fresh
processes (`results/CI2_gemm_chains.json`). This does not count as §8.A.
verified: refreshed extended descriptor device interpretation and unchanged
legacy ABI (`results/CI2_descriptor_abi.json`). Native interface failures
were missing ignored register fixtures; both now pass with original inputs.
The independent-attention numerical failure also reproduces on the unchanged
baseline (`results/native_test_repairs.json`); full G-REG remains false.

T3: verified 11/11 host analysis checks: window/stride/dilation/halo,
flattened row/image tiles, pixel shuffle, exact interval tables, split-K,
symbolic dimensions/origins, I2 row gather and WAR/WAW. Relations match
independent enumeration; box envelopes are explicitly `over`
(`results/CI4_exact_analysis.json`). Primitive and synthetic stage waits are verified below; real-body waits and reuse poison checks remain pending.

T3: verified bound window/table proofs, closed-form group capacities, and virtual
ownership/provenance (`results/CI4_bound_encoding.json`,
`results/CI4_capacity_form.json`, `results/CI4_virtual_binding.json`).
T4/T12 subcomponent evidence: sparse table, counted publication and weighted
last-arriver primitives pass 50/50 fresh processes; full synthetic stage
materialization passes 150/150 (50 each at kappa=1/4/16), with bitwise L1/L2
outputs and separate counter banks. Five architectures compile without spills
(`results/CI5_dependency_primitives.json`, `results/CI5_stage_dependencies.json`).
These synthetic bodies do not establish DNN/MoE TaskBody or model gates.

T3: verified exact table projection and geometry rebinding: 9/9 host checks,
plus 2/2 empty-row/corruption checks (`results/CI7_table_flow_host.json`,
`results/CI7_table_projection_empty.json`). B=1/2/8, workers=1/3/8 and
kappa=0/1/4/16 match enumerated dependency/event sets; TM=16/32/64 refits
the seed table from L-sem. These checks do not complete CI-7 or G-REG.

T4/T12: verified page-core forward/prefill dense and prefill attention checks
pass 150/150 fresh processes, five architectures and TM=16/32/64/128
(`results/CI5_paged_phases.json`, `results/CI5_paged_attention.json`). Native
L1/L2 outputs match bitwise and independent numerical oracles pass. Spills
remain in some larger-tile and retained loop kernels; every resource line is
recorded. These are synthetic core plans, not DNN/MoE model gates.

T4/T12: verified binding/page primitives pass 50/50 fresh processes and
five architectures with zero spills (`results/CI5_binding_page_primitives.json`);
registered host capacity/address checks pass 3/3 (`results/CI5_binding_host.json`).
Production PageStream integration and expert TaskBody/model gates remain pending.

T2/T3: verified binding capacity/provenance transport and legacy codec form
pass 6/6 host checks (`results/CI5_binding_metadata_host.json`).

T4/T12: verified monotonic weighted/unweighted LA primitives pass 50/50 fresh
processes and five architectures with zero spills
(`results/CI5_epoch_last_arriver.json`). Full nonpaged runtime and §8.A.11
checks remain pending.

T2/T3: verified nonpaged handoff selection and codegen transport pass 11/11
host checks (`results/CI5_nonpaged_handoff_host.json`); native core evidence follows.

T4/T12: verified nonpaged native split-K LA passes 100/100 fresh processes,
five architectures and kappa=0/1/4/16 (`results/CI5_nonpaged_handoff_native.json`).
L1/L2 and independent-stage/LA outputs match bitwise; poisoned intermediates,
monotonic mode-separated tickets, elided barriers and a four-step L1 loop pass.
Sixteen builds have zero spills. Attention LA and new-body/model gates are pending.

T12 foundation: verified 5/5 CPU NNLS and observation metadata checks
(`results/CI6_fit_host.json`). No measured fit or body timing is reported.

T12 foundation: verified 5/5 CPU streaming-weight, routing histogram, source
token and queue checks (`results/CI6_routing_host.json`). CUDA grouped_mm
collection matches the unchanged tiny HF model bitwise for two layers and all
13 T coordinates (`results/CI6_routing_cuda.json`). Real routing is pending.

T4/T12: verified binding-aware production PageStream passes 100/100 fresh
processes and five architectures (`results/CI5_binding_pagestream.json`).
Resources retain StreamProbe spills; real router/rowgather/model gates pending.

T4/T12: verified elided L1 binding producers resolve the owner's barrier in
100/100 fresh processes (`results/CI5_elided_binding_owner.json`). Five targets,
32 owner-barrier epochs and repeated production page cases pass; the skipped
producer's events remain zero. Thirteen builds retain StreamProbe spills.
Default non-DM PageStream fields/signatures are guarded; full G-REG is pending.

T2/T3/T12 foundations: verified access/layout/conv/epilogue-aware GEMM classes
and routing-profile capacity/traffic checks pass 5/5 host ctests; check-policy
passes (`results/CI7_classes_routing_host.json`). Legacy class keys remain
unchanged for default descriptors. Real-profile consumers and search are pending.

T12 foundation: verified small-tile resource/legality checks and neighboring
host regressions pass 6/6 (`results/CI7_gemm_traits_host.json`); check-policy
passes. The DM domain has 76 legal geometries, 46 additions; legacy serving
legality is unchanged. CUDA families and candidate integration are pending.

Remaining T3–T12 entries: pending implementation and required gates. G-DNN and G-MOE:
not run. §8.A: primitive and synthetic table/counted/weighted-LA paths have fresh-process evidence; real-body and remaining path gates are pending. No new TaskBody
has entered a performance matrix. No performance or calibration result is
reported.

## Scope decisions and deviations

- verified: one compiler rebuild also compiled two CUDA calibration objects
  without `flock`, contrary to §0.3. No kernel or timing ran. A locked rebuild
  passed; details are retained in `results/protocol_deviations.json`.
- Phase 0 inventories use seeded architecture initialization; weights and
  datasets are DN-1's Phase 2 work. Manifests explicitly set
  `accuracy_eligible=false`. This cannot substitute for pretrained numerical
  or model gates.
- Exports use an isolated CPU environment with torch 2.14.1, torchvision
  0.29.1, timm 1.0.30 and transformers 5.19.0. Package locks are committed.
  Existing LLM regression exports use their original torch 2.13 environment.
- verified: baseline `check-policy` rejects an architecture-macro token in
  a comment. Only that comment is reworded on `dnn-moe`; the reference is
  unchanged. Candidate `check-policy` passes. Default code generation must
  still pass the byte/SASS/token comparison.
- verified: the stale baseline test failure is retained as a failure, not
  disabled or reinterpreted as success. Its disposition remains outstanding.
- verified: the copied CPU guard test can lose `/proc/<pid>/stat` between
  existence/open/read as an already terminated worker disappears. The test
  now accepts the same absent-process condition during either syscall; its
  pidfd exit assertion and zombie-state assertion are unchanged. Guard and
  scheduler policy implementations are unchanged.
  verified: this host process-cleanup test passes in 50 fresh processes
  (`results/framework_cleanup.json`); it provides no GPU synchronization
  evidence for §8.A.

## Q1–Q7 and next work

Q1: verified only for upstream exportability; TileMega execution is pending.
Q2–Q7: unverified. Predictions are inferred and remain preregistered.

The reference bank, model inventories and preregistration are committed before
Phase 1. Implement CI-1 through
CI-7 in order with unit tests; resolve the native test build failure while
preserving its original numerical assertion. No Phase 4 timing is authorized
by an incomplete correctness gate.
