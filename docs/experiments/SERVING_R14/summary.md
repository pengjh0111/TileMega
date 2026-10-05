# R14 sm_89 — Phase 0 reviewed; Phase A in progress

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
| TR-4 | Partial: diagnostics reviewed; sampled stage and task profiles implemented, Phase A overhead acceptance pending | 377c674d2, 5ec398310, 686dc3afc; ServingTrace.cuh, ServingTaskProfile.h, ledger_r14.py |
| RW-3 / AT-1 | Isolated development branch: RW-3 code and AT-1 legality subset implemented; not accepted or merged | See development.md; RW-3 standalone 108 and paged 33 numerical cases pass |
| AT-3a | Implemented in isolated development, validation queued; default unchanged | 0c3d9ac45, 68410eb88; ServingAttentionPVSwap.h |
| AT-2, SK-1, GV-1, RA-1, EP-1, SL-6 | Not started | Required implementation remains |
| Conditional Phase C / Phase D | Not started | Decisions remain subject to registered evidence |

## Evidence / T1–T12

| Table | Checkpoint |
|---|---|
| T1 | No baseline timing yet |
| T2 | Diagnostics complete; full/timer/store median overhead +0.53%/+0.13%/+1.57%, but non-base outliers prevent stable attribution |
| T3 | Identity schema binds source snapshot, generated CU, binary, target, flags, ptxas and executed kernel; four diagnostic artifacts verified |
| T4–T10 | Not collected |
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

- Q1: verified only at position-coded task level; model token regression and C-1 remain pending. Old affected Llama B1 nonpaged timings cannot establish a correct baseline (inferred).
- Q2–Q6: unanswered; no performance evidence yet.

## Queue and resume

`queue/queue_phase0.json` contains P0_correctness → P0_build → P0_smoke → three trace rounds → P0_trace_analyze.
Scheduler PID is recorded in `/root/r14_work/scheduler.pid`; authoritative state is `scheduler/state.json`.
P0_correctness waits on process-completion file descriptors, then checks evidence; it performs no sleep/progress polling.
The flow and numerical/compiler check runners are `/root/r14_work/flow/run.sh` and `/root/r14_work/phase0/check.sh`.
All numerical tests/builds hold `/root/r14_work/gpu.lock`; all timing uses the copied R13 guard and its occupancy checks.
All seven original Phase-0 steps and both RW-3 numerical steps are done. Phase A has 36 queued steps, starting with Apre → A0 → A0s; B/C/D are not queued.
On resume read `scheduler/progress.tsv` once, then state.json and the completed step's summary. Do not start a second scheduler.
If any correctness step fails, no diagnostic timing can start; fix that failure and explicitly reset only its failed/skipped dependents.
Main source is frozen for Phase A. Independent code development stays in the separate worktree.

## Deviations / next work

- Independent 16 KiB private double-buffer storage exceeds sm_89 shared memory; both policies at 16 KiB are tested via the paged transport, Independent uses 8 KiB. This does not enable an invalid runtime configuration.
- Legacy R13 reference artifacts lack the new identity fields; preserve their binary/source SHA and unknown provenance explicitly. New artifacts enforce identity_schema=1.
- Next: run Phase A, verify baseline C-1/C-2 and trace overhead, then apply the registered baseline rule. Continue mandatory implementations separately.
- R15 scope remains unimplemented: multi-page stages, phase-subgraph handoff, shared simulator/codegen execution description, partial evaluation, architecture-specific collectives and prefill.

Independent CPU development proceeds in `/root/r14_work/development` (`r14-development`, latest implementation checkpoint `95eb2b159`), without changing the Phase-0 source/tools. See `development.md` for the exact boundary and evidence.

## Phase-0 review and TR-4 limitations

Verified evidence: `results/phase0_acceptance.json`, `results/T12_audit.json`, `results/T2_diagnostic.json`.
All cited Phase-0 raw measurements, identities, resource logs and final checks are in `raw/phase0_completed.tar.xz`; membership/SHA256 is in `raw/phase0_evidence_manifest.tsv`. Guard sampling is in `raw/phase0_guards.tar.xz`.
The guard rejected intermediate occupied/interfered attempts; each final diagnostic attempt returned 0. Nevertheless full-trace round 0 and stores-only round 1 were outliers. Do not infer that all remaining variance is caused by instrumentation or that the GPU was certainly uncontaminated.
TR-4 now moves the tasks-end store after barrier arrival and permits rotating 1/8 CTA stage sampling. Sampled extrema/tails are estimates, explicitly labeled by ledger_r14.py.
Task profiles are separate, sample 1/8 CTAs, and report the leader's intervals. The CLI currently restricts them to L1; nonpaged CUTLASS's first-page readiness is not directly instrumented (zero means unavailable, not zero wait). This is a remaining TR-4 limitation, not a completed metric.
An initial trace-only build failed because the nonpaged L1 dispatcher lacked a profile scope; 686dc3afc fixes that scope and retains the original per-task barrier. The corrected full serving trace build passes.
No new synchronization reliability conclusion is made. Full model correctness and 50-process requirements remain outstanding.
