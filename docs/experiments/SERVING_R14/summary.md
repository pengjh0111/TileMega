# R14 sm_89 — Phase 0 checkpoint, not final acceptance

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
| FX-23 | Implemented; initial numerical evidence collected, final checks queued | b3a6af1b3; AttentionPageLayout.h, IndependentAttentionTaskBody.h, PagedAttentionTaskBody.h |
| FX-24 | Implemented; fixed replay passes, full repaired search running | bb42f31bd; StageFlowModel.cpp, stage_flow_test.cpp |
| FX-25 | Implemented; generation/build validation queued | 7c2cd2436; build/identity.py, compile.cpp, ServingRuntime.cuh, identity_join.py |
| TR-4 | Partial: timer/store diagnostics prepared; lower-overhead implementation awaits diagnosis | 377c674d2; ServingTrace.cuh, ledger_r14.py |
| AT-1, AT-2, AT-3a, SK-1, GV-1, RW-3, RA-1, EP-1, SL-6 | Not started | Follow Phase A baseline |
| Conditional Phase C / Phase D | Not started | Decisions remain subject to registered evidence |

## Evidence / T1–T12

| Table | Checkpoint |
|---|---|
| T1 | No baseline timing yet |
| T2 | Three diagnostic variants plus uninstrumented control queued; no overhead conclusion |
| T3 | Identity schema binds source snapshot, generated CU, binary, target, flags, ptxas and executed kernel; actual rebuilt artifacts pending |
| T4–T10 | Not collected |
| T11 | Old attention: 26 failures; initial repair: 768 cases, zero failures. Final equivalent Full predicate and multi-architecture checks pending |
| T12 | Old fixed and joint sm120 failures reproduced. Fixed repaired search passes; repaired joint search running |

FX-23 evidence: `raw/FX23/numerical_summary.json`, old/fixed numeric logs and binary SHA256 records.
The layout host test covers ownership, release quorums and full-page validity. The final source's GPU rerun is separate evidence.
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
At checkpoint, P0_correctness is running; subsequent steps are pending. Phase A/B/C/D are not queued.
On resume read `scheduler/progress.tsv` once, then state.json and the completed step's summary. Do not start a second scheduler.
If any correctness step fails, no diagnostic timing can start; fix that failure and explicitly reset only its failed/skipped dependents.
Source is frozen for these checks/builds; continue Phase-0 work after the diagnostic evidence is available.

## Deviations / next work

- Independent 16 KiB private double-buffer storage exceeds sm_89 shared memory; both policies at 16 KiB are tested via the paged transport, Independent uses 8 KiB. This does not enable an invalid runtime configuration.
- Legacy R13 reference artifacts lack the new identity fields; preserve their binary/source SHA and unknown provenance explicitly. New artifacts enforce identity_schema=1.
- Next: close FX-24 joint replay and FX-23 final tests, validate FX-25 rebuilt identities, diagnose TR-4, implement its lower-overhead trace, then rebuild/anchor Phase A.
- R15 scope remains unimplemented: multi-page stages, phase-subgraph handoff, shared simulator/codegen execution description, partial evaluation, architecture-specific collectives and prefill.
