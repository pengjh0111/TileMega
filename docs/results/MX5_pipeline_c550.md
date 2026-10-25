# MX-5 C550 pipeline and specification audit

Verified source audit at `338989a707176bd8e7217343e4b37c4393d4a5be`,
2026-10-25, branch `tilemega-metax`, 13 commits after MX-4 `22a352242`.
Fetched origin; local and remote HEAD agree. This audit precedes resumed
implementation. Two uncommitted resource experiments were preserved under
`runs/mx5/state/resume_resource_six.patch` and removed before the next trial.

Authoritative specification: `/root/Prompt/TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
The user authorized recording differences and proceeding with the closest
compliant implementation. This does not waive correctness or acceptance gates.

## Current executable pipeline (verified)

| Step | Current source and behavior |
|---|---|
| LLM export | `scripts/mx_serving_cell.sh:25`: model config → serving export → exported graph → bridge JSON; checkpoint weights are loaded later. MoE registers its export pytree. |
| Compile entry | `tools/commands/compile.cpp:571`: decoder JSON/decode → `mx_decode::TryCompile`. `tools/commands/mx_decode.cpp:102`: BuildModelPlan; `:103`: DM plans return −1 to the old route. |
| Dense analysis | `mx_decode.cpp:114`: ImportPlan → AnalyzeMxDecode. `lib/Codegen/MxDecodeProgram.cpp:165` retains the MX-4 qkv whole-head restriction. |
| Dense selection | `configs/targets/xcore1000.json:2360` still points to the MX-4 table. `lib/Solver/MxDecodeSolver.cpp:249` forms the second candidate by replacing one class; `mx_decode.cpp:281` measures two candidates. |
| Dense codegen | `MxDecodeProgram.cpp:316` emits old `MxDecodeGemm` with qkv RoPE; `:327` emits old attention. There is no StageProgram integration yet. |
| Dense device program | `DecodeProgram.cuh:190`: Embed → L×[QKV, attention, O, gate_up, down] → head. `DecodeRuntime.cuh:52`: 5L+2 stages; `:252`: 512 threads. Last down still prefetches qkv before head. |
| Artifact/ABI | `mx_decode.cpp:120` captures identity, `:141` generates/compiles with cucc, `:178` emits manifest. `DecodeRuntime.cuh:105` exposes decode/L1 ABI; `:223` checks occupancy; `:260/:268` expose single/multiple steps. |
| Bind weights/state | `serving/plan.py:44` loads ABI and identity; `weights.py:234` gathers recipes and packs/deduplicates weights; `state.py:66` allocates shared KV, tokens and RoPE. Buffer names/recipes are the cross-plan contract. |
| Serve | `engine.py:40` checks seq/capacity, `:51` checks KV labels. Prefill launches once, then 1023 decode steps with past 64..1086. `execution.py:45` resolves mx_static to L1 and the kernel loop. |
| Measure/check | `measure.py:80` measures N=1024 and N=1 requests; `:139` computes TPOT=(E2E−TTFT)/1023. `mx_check_modes.sh:10` runs C-2 then teacher-forcing C-1; MoE HF check already selects grouped_mm. |
| MoE | `compile.cpp:826` sets moe_batch only in the old route; `:838–887` builds/imports/lowers the DM plan. No AnalyzeMxMoe or native prompt path exists yet. |
| DNN | `compile.cpp:738` builds DnnModelPlan then enters DM lowering/ModelHarness. `dnn/cli.py:58` defaults L2, `:229` defaults build-dm; there is no time command. `dnn/check.py:30` provides NativeForward ABI/identity/external-name validation to reuse. |

MX-5 v2 GEMM, head, attention, StageSyncV2 and LastArriverV2 currently have
unit/bench clients only. Their numerical results do not establish model-level
correctness, resource fit, or speed. The committed sync policy is not yet
consumed by model codegen. Production defaults remain MX-4.

## Required final pipeline (stated by the specification)

Dense keeps the MX-4 prefill structure/pins, rebuilding timing plans with
watchdog 0 and correctness twins with watchdog 1. DT-KV additionally requires
prefill's KV writer to emit whole blocks in the selected fragment layout.
The native decode becomes QKV → attention → O → gate_up → down for each layer,
then head; embedding is read directly by layer-0 QKV/O, giving 5L+1 barriers.
DownLast prefetches head; head prefetches next-step layer-0 QKV weights.

StageProgram must emit constexpr body types/shapes, by-value bindings,
external buffer tables, and kernels without runtime stage descriptors.
Launch shape, occupancy, scratch, 64-byte monotonic stage counters, token
broadcasts, watchdog, diagnostics, and prefetch loads must agree with manifests.
The existing L1 ABI and mx_static Python selection remain compatible.

MoE analyzes ModelPlan directly and runs six stages per layer: QKV, attention,
O, router/top-k, grouped gate_up, grouped down/combine; head ends the step.
Its prompt plan comes from the decode bridge and executes past 0..63 inside
one launch, with head only at 63. Prompt/decode must share external recipes.

DNN branches after BuildDnnModelPlan and before legacy lowering into its own
StageProgram client, specialized forward bodies, native packing and forward
ABI. The five covered models must reject legacy executor options/pathways.
Its time command must use the baseline's 50-forward event protocol; calling
NativeForward's validation routine would include copies/synchronization/clones.

## Ordered gates and evidence requirements

| Stage | Requirement | Verified status at audit |
|---|---|---|
| A / CP-0 | Environment/drift, five decision tables, 5×2000×104 litmus; push/report | Passed. Drift +0.4039%; 576 / fence+add / P-B / X-A / block_fragments; zero litmus violations. |
| B / G-B | v2 bodies, numerical tests, steady-loop MIR, zero stack, complete shared-body bench/table | Incomplete. Tests exist; 54/434 GEMM and 2/30 head kernels still spill. Representative MIR only; GEMM smoke bench only. |
| C / G-PROGRAM / CP-1 | StageProgram+dense integration; five development cells C-1/C-2 and TPOT; ledger | Not started. Each development cell must not regress against MX-4; D7 diagnoses failures. |
| D / G-SOLVER | Measured class domain, exact k-best top-4, two processes each; all 25 dense cells | Not started. pred/meas outside [0.9,1.1] requires attribution. |
| E / G-MOE | Native decode/prompt, numerical tests, five batches; B1/B16 route checks | Not started. C-1/C-2 and 64-step top-8 routing comparisons required. |
| F / G-LLM / CP-5 | Full 30-cell LLM acceptance, report/CSV, push/report | Not started. No distinguishable dense regression; slow MoE cells require per-layer attribution. |
| G / G-FWD | Native DNN, shared-body bench/top-4, 25 synthetic-weight checks and timings; DT-F | Not started; follows LLM acceptance. Only DT-F may be cut for budget. |
| H / G-FINAL | Final reports, selection/resources/ledgers, pins, provenance and discrepancies | Pending. No final performance or completion claim exists. |

LLM: BF16 greedy, ISL/OSL 64/1024, warmup 1 + timed 2, two fresh processes.
C-1 is 512 teacher-forced steps; C-2 is bitwise L1_separate/L1_loop.
DNN: two fresh processes, each warmup 50 forwards, two event-timed groups of
50 forwards, medians. All 25 synthetic-weight correctness checks must pass.
Only uninstrumented end-to-end results select defaults. Distinguishable means
greater than max(1%, inter-round range); P-PAIR alternates arms on one card.

Final coverage is 25 dense + 5 MoE + 25 DNN cells, each with solver selection.
External baselines are read from MX-4 CSVs, never remeasured. GPU work always
uses mx_gpu_job.sh, card order 7/6/5/4, at most two cards and one job per card,
900-second timing/1800-second other-job limits, MoE NEEDS_MIB=40960, 60 card-hours.
At this audit the guarded upper bound is 1.0887 card-hours; no GPU job is active.

## Reconciliation and immediate continuation

- Derived P_w is an implementation specialization, not a searched coordinate;
  this recorded difference resolves fixed kLoads versus geometry-dependent
  width. The full R_next measurement at P_w=0 and derived-width dispatcher
  are still pending; the existing bench uses a compile-time P_w macro.
- A proposed 512-thread exception for persistent stack below MTreg227 has not
  been implemented at plan level. MIR diagnosis and prescribed prefetch
  reductions must precede it; zero stack and register limits remain required.
  Spilling below 227 does not contradict a spec that also checks stack size.
- TN128/WN4/B16 residual tests need two early 16-byte vectors per thread;
  the report records the difference from the stated one-vector bound.
- StageSyncV2 still needs watchdog support before correctness-plan integration.
  A matching kv_layout string alone cannot validate prefill/decode memory layout.
- mx_serving_cell still needs prefill_w preference, watchdog0 timing builds,
  top-4 selection, MoE native routing/40GiB reservation and ≤900-second candidate
  timing guards. Existing legacy options are not MX-5-compliant final routes.
- The ignored progress file was stale and is being rebuilt from source/artifact
  evidence. Historical measurement rows and failed experiment logs remain intact.

Resume at the open B resource defect. Preserve the failed hoisting/uniformity
trials, inspect the final runtime ring-tail selector in MIR, then test any
minimal fix for numerical stability, synchronization, resources and waits.
Complete B's bench/table before C; do not prune failed legal candidates or
substitute legacy bodies to declare the gate passed.
