# DM-1 working report

Status: **incomplete, Phase 0**. This file is an evidence ledger, not a claim
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
- Final HEAD and commit count: pending completion. M0/M1/M2: not reached.
- verified: both production compilers build in their own `build-dm`.
- verified: GPU lock `/root/r14_work/gpu.lock`; R13 guard policy copied
  unchanged, SHA256 in `start.json`. No performance measurements taken.

## Implementation ledger

| Item | State and evidence |
|---|---|
| Phase 0 framework | verified: copied R13 scripts, 12 framework/selection/identity/recipe tests pass |
| Preregistration | verified: §3 retained verbatim in predictions; solver winners sealed before control-arm audit |
| Reference bank | in progress: fixed R13D geometry, separate reference/candidate builds and source identities |
| DN-1 inventories | verified: five upstream models and masked BERT exported, before/Core fixtures committed |
| DN-1 NAFNet source | verified: unchanged architecture/dependencies match upstream file SHA256; license preserved |
| DN-1 weights/data | preparation only: NAFNet/ResNet18/MobileNetV2 strictly load; SIDD CRC passes; BERT/MobileNetV1 safetensors identities recorded |
| CI-1 through CI-7 | not implemented |
| DN-2 through DN-11 | not implemented |
| MO-1 through MO-9 | not implemented |

## Tables and gates

T1: `results/T1_phase0.json` records a verified failure in both toolchains:
`norm_prologue_gemm_test.cu` lacks the RMSNorm header and calls a removed
`PagedGemmTaskBody::Run` overload. This is reproduced in the unchanged
reference. Full native test build and ctest therefore cannot pass. A separate
production-plan sealing queue does not count as passing G-REG.
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

T3–T12: pending implementation and the required gates. G-DNN and G-MOE:
not run. §8.A: no synchronization path has a 50/50 claim. No new TaskBody
has entered a performance matrix. No performance or calibration result is
reported.

## Scope decisions and deviations

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

Finish and commit the reference bank before Phase 1. Implement CI-1 through
CI-7 in order with unit tests; resolve the native test build failure while
preserving its original numerical assertion. No Phase 4 timing is authorized
by an incomplete correctness gate.
