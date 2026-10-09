# R14 sm_89 — D1 recovery accepted; final comparison queued

- Specified baseline: `76beaea5e2d66e3311b36d020f470c4f016406d0`.
- Initial local HEAD: `9aebaf6553247ec83c79bc8f101e61ad4ce564fd`; fast-forwarded before implementation.
- Prompt: `/root/Prompt/TileMega_R14_prompt.md`; SHA256 `c7e5771383873ae450c153d873bc633cf4481d9be5cea0f65e216e5b7073330e`.
- This is an intermediate checkpoint. Final HEAD, total commits and final acceptance will be recorded after Phase D.
- `R13_review.md` was not found; the saved prompt supplies the reviewed findings.
- Unrelated pre-existing changes in PLACE_EFT2/summary.md and SYNC_V2/sass_identity/meta.tsv remain untouched.

## Implementation

| ID | Status | Commit / location |
|---|---|---|
| Framework | Complete; CPU framework tests pass | f6599a0c0; gpu_guard.py, scheduler.py |
| Predictions / rules | Registered before timing | 586c51b5d; predictions.json, choose_r14.py |
| FX-23 | Implemented; final position-coded checks and five-architecture compilation pass | b3a6af1b3; AttentionPageLayout.h, IndependentAttentionTaskBody.h, PagedAttentionTaskBody.h |
| FX-24 | Implemented; fixed and joint repaired search-only replays pass | bb42f31bd; StageFlowModel.cpp, stage_flow_test.cpp |
| FX-25 | Implemented; unit checks and all four diagnostic artifact identities pass | 7c2cd2436; build/identity.py, compile.cpp, ServingRuntime.cuh, identity_join.py |
| TR-4 | Partial: diagnostics reviewed; sampled stage and task profiles implemented, all five stage/task trace medians below 2%; sampled extrema remain estimates | 377c674d2, 5ec398310, 686dc3afc; ServingTrace.cuh, ServingTaskProfile.h, ledger_r14.py |
| RW-3 / AT-1 | Implemented; model numerical/C-1/C-2 checks pass; default pipeline off | df96abb7f, 821c64cf6, 9fa448f3a; Phase-B evidence below |
| AT-3a | Implemented; five-architecture compile and position-coded tests pass; default unchanged | 0c3d9ac45, 68410eb88, dd7af352c; ServingAttentionPVSwap.h |
| AT-2 / SK-1 | Implemented; three required fresh-process cases each pass 50/50; measured LA/fill variants remain slower | c0849f8a2, 68c592729, ac731bc10, 84def10d5; ModelHarness.cuh, MonotonicLastArriver.cuh, SkeletonSearch.cpp |
| EP-1 / RA-1 | Implemented; five-architecture compilation and model checks pass; measured variants lose | 5434f44bb, a8588fb82, 9f0c36c2d; ServingEpilogue.h, PagedAttentionTaskBody.h |
| GV-1 | Partial TN8/16 epilogue domain; production numerical/C-1/C-2 pass; measured variants slower | 54756a6e3, 771aea037, 5ae2ffc8d; ServingGemv.h, ServingGemvTaskBody.h |
| SL-6 | Partial acceptance: both CLI builds complete; candidate admission limited by budget; final E2E remains pending | e95dbf3ee, 96205607b, a28bbb629; SkeletonSearch.cpp, cli.py, integrated_selection.py |
| C-RW1 / C-EP2 / C-AT4 | Implemented and correct, all rejected by retention; defaults off | 1b00e23ef, bb48614c7, 45f482860; IndependentAttentionTaskBody.h, ServingEpilogue.h, ServingPages.cuh |
| Phase D | D0/D1 and recovery smoke/family review complete; D2/D3 queued | 9d1270bb0; phase_d_final.py, make_phase_d_final.py |

## Evidence / T1–T12

| Table | Checkpoint |
|---|---|
| T1 | Phase A completed; paired anchor and baseline rule in results/phase_a_acceptance.json |
| T2 | Diagnostics complete; full/timer/store median overhead +0.53%/+0.13%/+1.57%, but non-base outliers prevent stable attribution |
| T3 | results/T3_phase_b_resources.tsv binds all 100 artifacts to identity and resources |
| T4–T7 | Phase-B matrices and task profiles collected; T4_phase_b.tsv and T5_phase_b_tasks.tsv; derived attention/coverage views remain to assemble |
| T8 | results/T8_phase_c.tsv: three conditional trials are correct but slower; none retained |
| T9 | results/T9_d1_selection.tsv records all admitted/unmeasured candidates, past times and reviewed selection |
| T10 | Four-cell final comparison pending; no final performance-gate claim |
| T11 | Old attention: 26 failures; initial repair: 768 cases, zero failures. Final Full predicate numerical rerun and multi-architecture checks pass; 64-step three-arm smoke token/KV mismatches zero |
| T12 | Old fixed and joint sm120 failures reproduced. Both repaired searches pass; fixed/joint conservation checks 6/468 |

FX-23 evidence: `raw/FX23/numerical_summary.json`, old/fixed numeric logs and binary SHA256 records.
The layout host test covers ownership, release quorums and full-page validity. The final source's GPU rerun is archived in raw/phase0_completed.tar.xz.
These numerical tests do not constitute 50-process synchronization validation.
FX-24 evidence: `raw/FX24/reproductions_partial.tar.xz`, minimal before/after logs, host_tests.log.
The negative fixed residual is -9.0949470177292824e-13 bytes: cohort averaging exceeded the task's own 4098.9888378587175-byte traffic by floating-point rounding.
Per-task subtraction preserves provenance and checks conservation without clamping. No model defaults changed.
The old joint command's cached options were not archived; reconstruction uses the recorded config/export and omits an unavailable sm120 seed manifest. This limitation is recorded rather than claiming an exact command replay.

## Q1–Q6

- Q1 (verified): four R13D rebuilds preserve all 1024 tokens across three rounds; all four C-1 checks pass. N1' and B0h' also pass C-1. Llama B1 selects N1' under the registered rule. Old affected nonpaged timings remain invalid correctness baselines (inferred).
- Q2 (verified): Ec128 improves Qwen3 B1 by 5.19%; PV swap improves Qwen3 B16 by 1.48%. Measured LA variants are slower. See paired Phase-B table below.
- Q3 (verified): measured fill and GEMV variants lose to their same-round controls; they do not enter defaults.
- Q4 (verified/inferred): register pipeline has no resolvable E2E improvement; sampled class profiles trigger only the Llama B16 resident-two experiment. Cause attribution remains incomplete.
- Q5 (verified): PV swap reduces paged spill and improves Qwen3 B16; noinline is slower. Final spill comparison remains pending.
- Q6 (verified): both multi-past CLI builds finish; selected Ec256/mma16 in all four cells, but most structural variants were budget-unmeasured. Final selection quality awaits D2.

## Queue and resume

`queue/queue_phase0.json` contains P0_correctness → P0_build → P0_smoke → three trace rounds → P0_trace_analyze.
Scheduler PID is recorded in `/root/r14_work/scheduler.pid`; authoritative state is `scheduler/state.json`.
P0_correctness waits on process-completion file descriptors, then checks evidence; it performs no sleep/progress polling.
The flow and numerical/compiler check runners are `/root/r14_work/flow/run.sh` and `/root/r14_work/phase0/check.sh`.
All numerical tests/builds hold `/root/r14_work/gpu.lock`; all timing uses the copied R13 guard and its occupancy checks.
All seven original Phase-0 steps, both RW-3 numerical steps and all 36 Phase-A steps are done. Phase B is accepted; Phase C retention and the pending Phase-D queue are recorded below.
On resume read `scheduler/progress.tsv` once, then state.json and the completed step's summary. Do not start a second scheduler.
If any correctness step fails, no diagnostic timing can start; fix that failure and explicitly reset only its failed/skipped dependents.
Phase-A binaries/source snapshots are preserved. Validated RW-3 and partial AT-1, and numerically gated AT-3a, are now integrated for further development.

## Deviations / next work

- Independent 16 KiB private double-buffer storage exceeds sm_89 shared memory; both policies at 16 KiB are tested via the paged transport, Independent uses 8 KiB. This does not enable an invalid runtime configuration.
- Legacy R13 reference artifacts lack the new identity fields; preserve their binary/source SHA and unknown provenance explicitly. New artifacts enforce identity_schema=1.
- Next: accept Phase D joint selection, evaluate PlanFamily, then complete final comparison and validation. No losing Phase-C optimization is enabled by default.
- R15 scope remains unimplemented: multi-page stages, phase-subgraph handoff, shared simulator/codegen execution description, partial evaluation, architecture-specific collectives and prefill.

Development checkpoint `001403629` has been merged. Main sources are frozen for Phase B (`queue/queue_phase_b.json`). See development.md for implementation limits.

## Phase-0 review and TR-4 limitations

Verified evidence: `results/phase0_acceptance.json`, `results/T12_audit.json`, `results/T2_diagnostic.json`.
All cited Phase-0 raw measurements, identities, resource logs and final checks are in `raw/phase0_completed.tar.xz`; membership/SHA256 is in `raw/phase0_evidence_manifest.tsv`. Guard sampling is in `raw/phase0_guards.tar.xz`.
The guard rejected intermediate occupied/interfered attempts; each final diagnostic attempt returned 0. Nevertheless full-trace round 0 and stores-only round 1 were outliers. Do not infer that all remaining variance is caused by instrumentation or that the GPU was certainly uncontaminated.
TR-4 now moves the tasks-end store after barrier arrival and permits rotating 1/8 CTA stage sampling. Sampled extrema/tails are estimates, explicitly labeled by ledger_r14.py.
Task profiles are separate, sample 1/8 CTAs, and report the leader's intervals. The CLI currently restricts them to L1; nonpaged CUTLASS's first-page readiness is not directly instrumented (zero means unavailable, not zero wait). This is a remaining TR-4 limitation, not a completed metric.
An initial trace-only build failed because the nonpaged L1 dispatcher lacked a profile scope; 686dc3afc fixes that scope and retains the original per-task barrier. The corrected full serving trace build passes.
Phase-B model correctness and the three required 50-process checks have now passed; the new Phase-C frontier and single-slot reuse still require their own checks.

## Phase-A acceptance

Verified: results/phase_a_acceptance.json; raw/phase_a_completed.tar.xz with raw/phase_a_evidence_manifest.tsv. Guard rejected occupied/interfered attempts; only successful final attempts enter the tables.

| Cell | Old R13D TPOT ms | Rebuilt R13D TPOT ms | Drift | Adopted baseline TPOT ms |
|---|---:|---:|---:|---:|
| llama_B1 | 2.82274 | 2.85861 | +1.27% | 2.84817 |
| qwen3_B1 | 4.16167 | 4.23549 | +1.77% | 4.23549 |
| llama_B16 | 3.18082 | 3.20018 | +0.61% | 3.20018 |
| qwen3_B16 | 5.41712 | 5.49964 | +1.52% | 5.49964 |

All four rebuild drifts exceed the ±0.5% prediction; token stability passes the specified stop condition. The cause is not yet isolated, so these are corrected baselines, not a claimed performance win. N1' is 2.84817 ms versus 2.85861 ms for the paged rebuild; their difference exceeds both ranges.
Stage/task trace median overhead spans −0.09% to +1.15%; instrumented tokens match base. Stage extrema use rotating 1/8 CTA samples and remain estimates.
AT-3a initial compile failed in its test due to ambiguous `E` (CuTe namespace); dd7af352c fixes the test name without changing expected values. Fixed compile/numeric steps pass; evidence is raw/implementation_numeric_checks.tar.xz and its manifest. EP/RA checks also pass: argmax has 45 shapes repeated three times; unchanged paged GEMM has 33 cases. These are intermediate implementation checks, not final model acceptance.
Nonpaged LA initially failed the host lowering gate and then lacked the nonpaged arrival include. ac731bc10 and 84def10d5 correct these implementation errors. Historical failed steps remain recorded; `_v2` checks are explicit retries. The v2 host/architecture and single-process tests pass; All three LA_model_build_v2 builds and their three-arm 64-step smoke checks pass (tokens and every KV cache match). No 50-process reliability conclusion is made.
SL-6 integration detail: average linearly interpolated measurements at integer pasts 64..1087, holding the past1000 endpoint thereafter; retain half per pilot, then three fresh finalist rounds. Spill and execution identity are retained. Ec/attention/LA variant construction and required GEMV family are integrated; full CLI selection remains unvalidated.
L1 small-chunk order changes only execution ordinals; logical g-major dependency indices remain unchanged. EP tail spreading uses a coprime CTA permutation, default off with parallel argmax; L2 retains the solved placement. These latest mapping changes await complete harness/model validation.

## Latest implementation review

Verified raw evidence: `raw/la_gemv_completed.tar.xz`, its SHA256 manifest, and `results/implementation_acceptance.json`.

| LA case | Runtime / queued / elided stages | Split GEMMs | Smoke |
|---|---:|---:|---|
| Llama B1 | 99 / 83 / 16 | 0 | L2 separate, L1 separate, L1 loop: token/KV identical |
| Qwen3 B1 | 171 / 143 / 28 | 0 | Same three arms pass |
| Llama B16 | 131 / 83 / 48 | 32 | Same three arms pass |

Standalone GEMV position-coded tests cover M/N/K tails, row/tiled layout and legal epilogues; sm_80/89/90/100/120 compile, only sm_89 executes. This is not yet end-to-end GEMV acceptance.
GEMV TN8/16 cannot own DN's complete 32-column square-sum block or a SwiGLU pair: these combinations are explicitly rejected; TN32 retains all epilogues. This is a partial implementation deviation, not a changed correctness criterion.
GEMV candidates account for the independent attention shared-memory union, retain implementation-specific resource/cache keys, and carry implementation into manifest and identity. Existing MMA keys remain unchanged.
Performance defaults remained unchanged through Phase B. Its three prescribed fresh-process cases now each pass 50/50.

## Phase-B freeze

Verified GEMV production evidence: `raw/gemv_integration_completed.tar.xz`, SHA256 manifest and `results/gemv_integration_acceptance.json`; both B1 models have zero token/KV mismatches across the three smoke arms.
`queue/queue_phase_b.json` contains 72 steps and 100 fixed/trace artifacts. Bpre rebuilds the frozen compiler and runs host/architecture checks; Bpre_numeric reruns arithmetic gates; B0b builds all declared arms, then per-cell B0c smoke gates performance. Failed nonbaseline artifacts are excluded with records.
B1–B5 each use three paired rounds, followed by task trace, per-arm C-1/C-2 and the three required 50-process cases. Qwen B1 GEMV also includes its same-geometry nonpaged control (AT_la_ref), because the selected baseline is paged. No Phase-C change is enabled.
SL-6 budget detail: the configured wall-clock budget stops admission of new second-stage pilots/builds; already admitted builds and three finalist confirmation rounds finish. The first-stage search remains bounded by its search budget and candidate count, not an interruptible global hard deadline. This is a remaining budget-enforcement deviation and must be reported against actual D1 durations.
The multi-past variant grid uses Ec={32,64,128,256,512,capacity}, mma16/pvswap, and nonpaged LA=0/1. Defaults remain unchanged before measurement.
Resume by reading progress.tsv once, then the completed step's result and guard record. Do not change runtime/compiler sources while this queue builds artifacts.

## Phase-B acceptance and Phase-C preparation

Verified: results/phase_b_acceptance.json, T3_phase_b_resources.tsv and T4_phase_b.tsv. All 72 scheduled steps returned 0, all 100 artifacts verify, all 50 model variants pass C-1/C-2. The three protocols are Llama B1 attention LA, Qwen3 B1 attention LA, and Llama B16 nonpaged combine LA: each 50 passed, zero failed. Final guard attempts accepted; no within-matrix canary exceeds 2%. This does not rule out slow drift between phases.
Original matrix/correctness/resource records are archived in raw/phase_b_review.tar.xz and phase_b_review_manifest.tsv. Full task profiles/logs are packaged by C_archive into raw/phase_b_completed.tar.xz before new builds; commit that full archive on resume.

| Cell | Same-round best supported change | TPOT ms | Relative baseline |
|---|---|---:|---:|
| Llama B1 | Baseline (B2) | 2.84820 | 0% |
| Llama B16 | AT-pv | 3.17325 | −0.98% |
| Qwen3 B1 | AT-ec128 | 4.16402 | −5.19% |
| Qwen3 B16 | AT-pv | 5.62227 | −1.48% |

These are separate registered matrices, not combined final defaults. All other choices, sample ranges and discernibility are retained in T4_phase_b.tsv. Qwen3 Phase-B baselines are slower than Phase A; do not attribute that drift to an optimization.
All fill/GEMV trials, EP-1 and noinline lose; register pipeline's Qwen3 B16 change is not distinguishable from its range. The model and numerical correctness tests passed even for losing variants.

Phase-C decision was committed before implementation/timing (09237464c); predictions were registered in d57d10331. Triggered cells: C-RW1 Llama B16, C-EP2 Qwen3 B16, C-AT4 Qwen3 B1. C-AT3b, C-RW2 and C-GV2 did not trigger. PlanFamily waits for D1.
- C-RW1: one-buffer independent attention plus four/two-stage GEMMs and TN128 head; fixed case requests residency 2. The measured family is retained by seed_resident2 (c5701f112).
- C-EP2: Store/Residual directly write rounded fragments; only SwiGLU partner values and DN's exact ordered square sums use shared rearrangement. Small GEMV residuals keep the old fallback.
- C-AT4: paged B1 L1 publishes one context event per KV group in a dedicated L1 bank; o_proj waits for that output frontier instead of all attention CTAs. Other barriers and L2 semantics remain intact. New wait site is 13.
Correctness arguments: the final writer releases each complete context; consumer acquire plus compute barrier precedes A reads; the next ordinary stage barrier orders the remaining graph. The bank is separate from L2 and monotonically indexed by L1 iteration. The tested frontier and single-slot paths each now pass 50 fresh processes; the slower variants remain disabled by default.
Queue definitions contain six immutable artifacts and 22 steps: full evidence archive; compiler/host and five-architecture checks; position-coded single-buffer and bitwise fragment numerics; builds and smoke; three paired rounds per cell; full C-1/C-2; two 50-process cases. A rejected conditional arm is recorded and excluded, not substituted.
Integration details (d4e7729a3): decode first-level searches share one third of the budget; already built execution baselines remain measurable when new structural admission ends. Finalist confirmation can exceed wall budget and is reported. features_by_batch applies only the three conditional flags to their triggered decode cells, leaving prefill unchanged. CPU selection/config/identity tests pass (raw/C_definition_checks).
queue/queue_phase_c.json was published to the existing scheduler (8d0808064); Phase C performance/correctness collection is accepted below. Final T9–T10 acceptance remains pending.
Read-only monitoring: `watch -n 10 'python3 /root/TileMega/docs/experiments/SERVING_R14/status.py --prefix D'` shows each step as done/pending/running/failed; use `--prefix ''` to include all R14 queues. GPU occupancy/retry policy remains unchanged.

## Phase-C acceptance and Phase-D preparation

Verified: results/phase_c_acceptance.json, phase_c_retention.json and T8_phase_c.tsv. Of the original 22 steps, 21 pass; C_arch fails before compilation because its command omits CUTLASS tools/util/include. Production sm_89 builds and arithmetic tests pass. 761d75d34 preserves all original include/macro options and creates architecture-pinned compile-only specimens; C_arch_v2 now passes for sm_80/89/90/100/120 (sm_89 executed; others compiled only).

| Conditional item | Cell | Baseline TPOT ms | Candidate TPOT ms | Relative | Correctness / fresh processes | Retain |
|---|---|---:|---:|---:|---|---|
| C-RW1 | Llama B16 | 3.19906 | 3.27723 | +2.44% | C-1/C-2 pass; 50/50 | No |
| C-EP2 | Qwen3 B16 | 5.70875 | 5.92546 | +3.80% | C-1/C-2 and baseline token equality pass | No |
| C-AT4 | Qwen3 B1 | 4.39166 | 4.58783 | +4.47% | C-1/C-2 and baseline token equality pass; 50/50 | No |

All 18 final GPU guards accept; no within-matrix canary exceeds 2%. These checks cannot exclude the previously recorded drift between phases. All six artifacts match their execution identities. C-RW1 actually uses residency 2 and 49152 B shared memory. C-EP2's ordered BF16 square sums pass 72 shapes repeated three times. Evidence, including original failed C_arch, is raw/phase_c_completed.tar.xz with phase_c_evidence_manifest.tsv.
The full Phase-B evidence is committed as phase_b_completed.tar.xz.part00/part01 to keep each file below 100 MiB; phase_b_archive_parts.tsv records each checksum and the complete archive checksum. Restore with `cat raw/phase_b_completed.tar.xz.part* > raw/phase_b_completed.tar.xz`; phase_b_evidence_manifest.tsv binds every member.

Phase D definitions contain 11 steps: C_arch_v2; frozen compiler/CPU checks; selective calibration plus five fresh bandwidth processes; four rebuilt controls and smoke; each model's SL-6 build, selected-plan smoke and PlanFamily audit. They were published in 3e2e9c7f9 as queue/queue_phase_d1.json to the existing scheduler. configs/e2e/*_r14.json retain ordinary double-buffer plans and disable frontier/direct epilogue. Llama B16 additionally searches the required single-buffer resident-2 family alongside ordinary plans (1162a8e08); the Phase-C fixed-trial loss is retained as evidence and does not enable a default.
D1 is guarded for the entire compile/measure command, rather than only its internal GPU sections: hidden interference yields 75 and cache-assisted retry; children inherit LOCK_HELD. This stronger exclusion also holds the GPU lock during CPU compilation. Default APIs and device code are unchanged by this preparation.
PlanFamily is screened from compatible same-GEMM, same-execution candidates using the three-past envelope; this is an optimistic trigger estimate, not a claimed two-segment gain. D2/D3 are deferred until D1 resolves this specified code dependency. If triggered, implement compatible two-segment switching and include measured switch cost before final comparison.
CPU selection/config/family tests pass (raw/D_*tests.log, D_static_validation.log). On resume read progress.tsv once, inspect D1_planfamily_*.json, then generate guarded D2/D3. Do not change compiler/runtime sources while D0/D1 builds are queued. R14 remains incomplete until final comparison, C-1/C-2 and any uncovered synchronization path checks finish.

## D1 failure acceptance and recovery (2026-10-08)

Verified: raw/D1_recovery/acceptance.json and archived original logs. The initial D queue ended with 5 done, 2 failed and 4 skipped. D0, rebuilt-control smoke and the repaired architecture check pass; both D1 builds fail with SIGSEGV (-11), before completing their first nonpaged decode B1 search. Their selected-plan smoke/family steps were skipped. Earlier 75 retries are external occupancy/interference; final guard results are child exit 1, not contamination.
Root cause: SkeletonSearch.cpp::SetServingStructure moved the active plan/classes into its cache before BuildModelPlan rejected a GEMV TN8 argmax tile. The next candidate accessed this moved-from state. gdb_before.log directly reproduces SearchContext::Evaluate SIGSEGV after rejection; original search tails preserve the argmax error for both models.
f5ce261bb builds and validates a replacement before changing active state. Three CPU ctests pass; serving_search_rejection covers both incremental modes, with illegal-candidate rejection and later legal scores identical to the control. No device, synchronization, price or default-selection code changed. This is a necessary post-freeze correctness repair, recorded as a deviation; all nine D0 calibration stamps still match (calibration_stamps.json).
Recovery uses nine new `_v2` steps with separate baseline and driver output paths, preserving failed records and original artifacts. Controls and both models' D1 are rebuilt under the repaired compiler, followed by smoke and PlanFamily audit. The existing scheduler, GPU guard and exit-75 policy remain in use; no live scheduler state is reset. Final D2/D3 still follow successful D1 and its conditional-family decision.
Prior successful D0/control/architecture evidence is raw/phase_d_pre_recovery.tar.xz, with phase_d_pre_recovery_manifest.json; failed-run evidence is raw/D1_recovery/. Generated sources and binaries are registered by path/hash, not committed.
Resume by reading scheduler/progress.tsv once and results/D1_planfamily_*_v2.json. Monitor all D steps with `watch -n 10 'python3 /root/TileMega/docs/experiments/SERVING_R14/status.py --prefix D'`; old failed/skipped names intentionally remain visible beside the new recovery steps.

## Recovery result review (original 2026-10-08 snapshot)

Verified: results/D1_recovery_progress.json and raw/d1_llama_review.tar.xz, with d1_llama_review_manifest.json. Of the nine recovery steps, six are done and three remain pending at this snapshot. Rebuilt controls and selected Llama B1/B16 artifact identities verify; all four controls and both selected Llama plans pass 64-step smoke. Llama D1 final guard returns 0. Qwen3 D1's latest attempt returns 75/preflight occupied after 34 attempts; its smoke/family steps wait for successful build. Scheduler remains alive; no new compiler failure is recorded.
Llama B1 selects pages/L2/loop, Ec256/mma16, with integral confirmation samples 2.887087/2.888612/2.884280 ms. Llama B16 selects the same execution/attention combination, but its samples 3.516804/7.479870/3.506781 ms are not reliable performance evidence; other B16 finalists also have large ranges despite the accepted guard. Build/identity/arithmetic acceptance does not close timing acceptance.
One bounded recollection of all three B16 finalists is queued as D1_reconfirm_llama_B16_v2 (e3d89b898), retaining the original samples. It rotates three rounds, uses the identical binaries and three-past protocol under the existing GPU guard, applies the registered 2% reference-canary rule, and never rewrites cached choices or defaults automatically. If still unstable, report that outcome; do not keep recollecting.
Budget limited structural admission: B1 has six piloted executions and three confirmed finalists out of 108 candidates; B16 has nine and three out of 180. The remaining 102/171 candidates are budget-unmeasured. PlanFamily reports no trigger only within this measured subset; it does not exclude benefits from unmeasured Ec/implementation families. SL-6 coverage and final selection quality remain limitations pending the final comparison.
The recollection uses raw/D1_review/plans_llama.json, captured in the committed archive; restore that member if resuming on a clean checkout. Final D2/D3 are still pending. Do not report R14 complete from these build/smoke results.

## Recovery acceptance and final queue (2026-10-09)

Verified: all nine recovery steps and the one bounded B16 recollection finish successfully. Four selected artifacts match their 64-step smoke identities; both models' PlanFamily audits are not_triggered within the budget-admitted subset. Qwen3's final accepted guard returns child exit 0; prior exit-75 attempts remain recorded, without accepting their timings.
Evidence: results/D1_recovery_acceptance.json, raw/d1_recovery_completed_review.tar.xz and d1_recovery_completed_review_manifest.json. The archive contains inputs, choices, aggregate predictions, resource/identity files, measurements, smoke and guards; expanded edge/task/price dumps not used by any reported number are excluded and listed by path/size. Binaries and generated sources remain registered by path/hash.

| Cell | Reviewed D1 pg / mode / loop | Integral confirmation ms (3 rounds) | Piloted / total | Budget-unmeasured | Actual kernel spills |
|---|---|---|---|---:|---|
| Llama B1 | pages / L2 / yes | 2.887087, 2.888612, 2.884280 | 6 / 108 | 102 | No |
| Llama B16 | pages / L2 / no | 3.509447, 3.521283, 3.501730 | 9 / 180 | 171 | No |
| Qwen3 B1 | pages / L2 / yes | 4.390193, 4.371947, 4.384237 | 6 / 108 | 102 | No |
| Qwen3 B16 | pages / L1 / no | 5.681081, 5.670789, 5.671946 | 6 / 108 | 102 | Yes |

All selected attention variants are Ec256/mma16. These are three-past candidate measurements, not final E2E results. B16's stable recollection replaces the noisy confirmation only after review: L2 separate median 3.509447 ms versus loop 3.518002 ms; the 0.24% difference is not discernible. Minimum median still selects separate under the preregistered candidate rule. Original samples/choices remain archived; only serving sidecars and run plans.json change, never manifests, identities or binaries.
All rebuilt controls and selected prefill/decode artifacts share one source_digest/compiler_sha256 state, despite documentation-only HEAD differences. Final rows verify actual executor, loop use, prefill identity and trace exclusion; Qwen3 B16's selected spilling kernel remains explicitly marked.
Final definitions: 41 steps, including 12 paired 1024-token rounds with vLLM, four C-1 jobs, four C-2 jobs and four explicit paged 50-process cases, plus CPU checks. Each TM arm uses the same new prefill, L1; R13D is its historical decode binary with that common prefill. R14F reads reviewed sidecars via auto. Final paged protocols cover paths absent from B6's nonpaged cases; no reliability claim is made before they finish.
Five CPU definition tests pass; raw/D_final_definition/{cpu_tests.log,static_validation.json,recovery_state.json} bind the prepared work. Guard requirements are unchanged; device memory comes from the recorded property query. Any marked final canary may be replayed once; it cannot silently enter final defaults.
0fed1fd6c published queue/queue_phase_d_final.json. This historical definition is retired by the implementation-completion review below; its 41 tasks had not started. Historical D1 failures/recovery remain preserved.

## TR-4 / GV-1 / SL-6 implementation completion (2026-10-09)

Verified code gaps, not merely missing tests: nonpaged GEMM/GEMV profiles had no first-ready observer; TN8/16 GEMV rejected DN residuals and SwiGLU; old budget admission could measure only base variants and still publish a choice. Earlier Phase-B GEMV results cover the implemented TN32 trials, not the full narrow family. Earlier D1 choices are historical, incomplete-coverage evidence.

| Item | Completion commit / location | Validation |
|---|---|---|
| TR-4 | 258d6e73c; ServingProfiledMainloop.h:55, ServingGemmTaskBody.h:124, ledger_r14.py::tasks | GEMM/GEMV first-ready numerical probe passes; model profiles/overhead pending |
| GV-1 | 26d7f4c14; ServingDeferredNorm.h:10, SkeletonSearch.cpp:175, ServingEpilogue.h | TN8/16/32, both layouts, every epilogue and DN numeric test passes; narrow real models pending |
| SL-6 | 06090c9ea; integrated_selection.py:13/31/56, build/budget.py:8, cli.py:435 | CPU budget/coverage/selection and identity tests pass; fresh D1 pending |
| Test repair | bf6ba3cc4; serving_task_profile_test.cu, serving_gemv_test.cu, skeleton_search_isolation_test.cpp | Stable host/device architecture and immediate launch-error checks; rejection case now uses illegal TN4 |

Evidence: raw/completion_repair_initial.tar.xz and completion_repair_initial_manifest.json preserve initial failures, fixes and checks. Three targeted ctests and 21 selection/config/identity Python tests pass. First-ready probe's initial zero readings were a test architecture-tag mismatch, not accepted trace data. Five-architecture checks are queued, not claimed passed here.
Implementation detail: narrow residual writers use distinct eight-column sum slots, consumers reconstruct the existing 32-column sum tree, and embedding preserves its sequential sum. Narrow SwiGLU uses plan-wide gate/up interleave 4/8; default plans keep 16 and the original sum representation. Direct-register GEMV marks readiness after its first vector load. These support the specified narrow family without adding an optimization beyond GV-1.
Budget deviation: R14 configurations now allow 10800 s per phase/batch instead of 1800 s, to build and attempt the required Ec/body/LA/execution dimensions. Deadline includes subprocess sessions; three confirmation rounds reserve time inside it. Missing coverage or incomplete confirmation fails with evidence and publishes no winner. No model coefficients, correctness thresholds or GPU guard limits change.
Scheduler was stopped safely with no running tasks (raw/completion_repair/scheduler_pause.json). The old final definition is preserved at retired_queues/queue_phase_d_final.json. New queue/queue_completion_repair.json has 21 steps: compiler/host/architecture checks, numerics, ten fixed/trace builds, default-path CU/resource/SASS comparisons, smoke, four narrow-model C-1/C-2 checks, new traces/paired overhead, calibration and both models' fresh D1/smoke/family audits. Performance waits for numerical correctness. No old recollection or sidecar is reused as the new D1 choice.
Resume with one read of scheduler/progress.tsv and this queue's status; inspect results/D1_planfamily_*_v3.json before defining new D2/D3. Monitor: `watch -n 10 'python3 /root/TileMega/docs/experiments/SERVING_R14/status.py --queue queue_completion_repair.json'`. R14 final performance/correctness acceptance remains pending; this is implementation completion plus queued verification.
