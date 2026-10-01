# R12c implementation freeze and queued diagnostics

Status: implementation complete; GPU acceptance and attribution pending. This is an interim report, not a performance or correctness acceptance.

- Prompt: `/root/Prompt/TileMega_R12c_prompt.md`.
- Prompt SHA256: `9cd721a147925caa97530746b72671caa18b49aae5b2fb61633b32438aebca97`.
- Baseline actual HEAD/origin: `a1264e59de6764e7a463e184fd11ba9c621bf54e`.
- Frozen implementation/evidence HEAD: `ccc3db5ba`; 16 commits before this documentation commit, 17 including it. Publication HEAD is the commit containing this report, resolved by `git rev-parse tilemega`.
- Frozen source fingerprint: `a9dfaa0dc3bb44780a9bd3eda31a7b8b9b4ede93c9c765b8771b3ca166e2a12f`.
- Immutable Phase A tools: `/root/r12c_work/diag`, commit `397209def`. Main source changes do not affect those tools or existing binaries.

## Implementation and checks

| ID | Implementation | Commit | Key location | Verification |
|---|---|---|---|---|
| D-0 | Complete: archive, binary registry, T0 | 26d42559c | archive_r12b.py; arms.py; analyze.py | Archive manifest, hashes and tables committed |
| FX-12 | Complete: paged decode only; explicit nonpaged baseline | e3a6b0453 | python/tilemega/cli.py:329 | No decode_pg_choice in code; B2 cache/manifest check pending |
| FX-13 | Complete: seed carried into top-3; origins; selected geometry | 1fabdf3a4 | lib/Solver/SkeletonSearch.cpp:878; tools/commands/compile.cpp:773 | Build passed; B2 seed/origin evidence pending |
| FX-14 | Complete: per-page lookahead; historical stream-byte accounting | 32e2da746 | executor/ServingPages.cuh:116; PagedGemmTaskBody.h:130; PagedAttentionTaskBody.h:51 | Bodies compiled; D128 smoke/time comparison queued |
| FX-15 | Complete: register squares and unified xor reduction | 78ced0c0b | Backend/ServingEpilogue.h:206; test/unit/serving_epilogue_test.cu | TN32/64/128 test compiled; GPU assertions queued |
| FX-16 | Complete: hoisted Watch; compile-time watchdog switch | fbab316f9 | executor/Async.cuh:52; PageRing.cuh:74; Watchdog.cuh:36 | EventSync.cuh unchanged; runtime/compile ablations queued |
| FX-17 | Complete: DN/LA/watchdog/guard-budget controls; stage counts; stub | aeddc6840 | tools/commands/compile.cpp:340; ModelHarness.cuh E2E_STAGES; measure_stub.py | Default generated CUDA byte-identical to reference |
| FX-18 | Complete: fresh candidate instance per past | 3c9f97183 | python/tilemega/serving/measure_candidate.py:124 | Python compiled; A2 measurement queued |
| FX-19 | Complete: explicit paged protocol cases; fidelity; geometry pin | 206b0c9d0; 5d195bcb9 | SERVING_R12B/make_protocol_cases.py:5; fidelity.py; pin_case.py:13 | 26/28 fixed inputs valid; two explicitly rejected |
| FX-20 | Complete: retry 75 and lock ownership propagation | 0505e4033; 5d195bcb9 | vllm_baseline.py; trace.py; cli.py; check_protocol.py | Guard/tool CPU tests passed; external interference testing pending |
| Framework | Complete: guard, dynamic scheduler, anchors, both queues | 397209def; 8c76a8916; ccc3db5ba | gpu_guard.py; scheduler.py; anchor.py; definitions/queue_*.json | Unique names, dependency DAG, timing guards verified |

Locations without a directory prefix are relative to SERVING_R12C or include/tilemega/Codegen/tasks/executor as indicated; the commit diffs disambiguate exact files.

Executed checks: host ctest 4/4 (serving_pruning, serving_lag, serving_task_index, skeleton_search_isolation); guard unit tests 2/2; diagnostic-tool tests 4/4; Python syntax checks passed. Core compiler and three CUDA test executables built. Logs: implementation/{build,host_tests,framework_tests,tool_tests}.log; machine-readable evidence: implementation/checks.json.

Default `.cu` SHA256: `708cd9fe2db0e5f817761bc62b42e981bfc549d440d83a67c372678b499d799f`; baseline and current files under `/root/r12c_work/ref_cu/` compared byte-for-byte. The target snapshot is `/root/r12c_work/target_r12b.json`, SHA256 `c2c03ad6e9534130762cc88d423aac336077a0bd040db6d337a1e31e3c1ec8b9`. Neither generated CUDA nor binaries are committed.

The last checked compiler fingerprint was `033b22e7c00e3fa095ec57f18728a485ccdb569180c87feeb74c4ccc9c9c13f0`; the final Python trace retry correction changed the source fingerprint. Bpre rebuilds and checks the frozen fingerprint before Phase B. This Python-only correction does not change generated CUDA or any calibration section stamp.

## T0: archived evidence, not a new paired E2E measurement

Candidate-protocol medians (ms), with the three raw samples retained in results/T0_pg_choice.tsv:

| Cell | R12b pages | R12b l2 | Selected |
|---|---:|---:|---|
| Llama B1 | 4.125152 | 3.877642 | l2 |
| Llama B16 | 4.755047 | 4.418162 | l2 |
| Qwen3 B1 | 6.031072 | 5.483562 | l2 |
| Qwen3 B16 | 7.396800 | 6.483894 | l2 |

Verified: all four decode selections were nonpaged. The archived paged searches lacked the seed in the measured shortlist. Their recorded solve times were 817.466/838.535/1198.410/1337.937 s (same cell order); these are historical records, not R12c budget results. Full geometry, search-budget lines and shortlist data: results/T0_plans.tsv. Historical per-past values: results/T0_past.tsv.

Raw inputs: MANIFEST.tsv and r12b_archive.tar.xz (530 files, 8,625,963 original bytes), archive SHA256 `7957910c48ef6e2fc7496e28d04b34c928c187645e691bd99ef562eb14940f34`. Extract with `tar -xJf r12b_archive.tar.xz` from this directory. arms.json records paths and binary hashes; unavailable original compiler provenance stays explicitly unknown.

## T1–T9 and attribution status

| Table | Required observations | Status at freeze |
|---|---|---|
| T0 | Historical selection, geometry, search, per-past data | Complete, committed |
| T1 | A1 paired E2E, floors, past buckets, slopes | Queued; not accepted |
| T2 | Executor/tool/evolution/V3/prefetch/DN/geometry/watchdog effects | A4 and B8 queued |
| T3 | Candidate protocol vs E2E at past192/575/1000 | A2 queued |
| T4 | S1P runtime ablations | A3 queued |
| T5 | New top-3 origins, seed, fidelity, solve duration | B2 queued |
| T6 | D/watchdog and MS controls; stage counts; step trends | B1/B3 queued; two invalid arms below |
| T7 | Chain residuals, full-page dependency waits, loader busy fraction, stream floor, optional ncu | B6 queued |
| T8 | Four-cell R12cP/R12bP/R10C/vLLM comparison | B5 queued |
| T9 | Smoke, C-1/C-2, repeated tokens, fresh processes | B0c/B7 queued |

Q-A [stated]: the candidate explanations remain unresolved; no controlled new attribution data are accepted yet.
Q-B [verified, limited]: archived short candidate measurements chose l2 in four cells; this alone does not establish full-request superiority. H1–H4 require A1/A2 and the fixed-geometry controls.
Q-C [stated]: paged residual-gap attribution needs A3/B1/B6; the 902 GB/s stream floor is an explicit diagnostic convention, not a new measured bandwidth claim.
Q-D [stated]: DN/LA values await B3. The Llama B1 DN-off paged controls cannot preserve their source geometry under the target class partition.
Q-E [stated]: final paged position and selection quality await B2/B5/fidelity.

## Bounded omissions, deviations and unresolved evidence

1. Reference CUDA and target were saved before Phase A Python edits, earlier than the prescribed step3: Python enters the source fingerprint, so this preserves an actual baseline compiler check. Default-source equality passed.
2. R10-C's original measurement tool has hardcoded warmup plus three repeats and lacks a repeats option. The tm_old adapter uses its supported interface, retaining those repeats inside each outer round; R10C_new isolates the tool effect. This adds work and changes that arm's within-round aggregation, which T2 must report.
3. Two fixed arms are rejected before compilation: llama_B1 P-noDN and P-noDN-noLA. Their same-DN donor class0 includes source GEMMs with different geometry. The required pin assertion rejects these rather than changing geometry or solver classification. Evidence: implementation/pin_preflight.json. Other fixed inputs pass (26/28); actual build/smoke failures remain possible.
4. Some historical binaries lack recorded compiler commit/source fingerprint. Paths, hashes, manifests and available target metadata are registered; missing provenance is not invented.
5. GPU numerical assertions, multiarch builds and SASS FP64 audit are queued, not claimed passed. No synchronization or race conclusion is made.
6. The report is interim because §0.1 explicitly requires ending the session after queue submission. Final raw outputs, configs and T1–T9 will be archived and committed on resume; no not-yet-observed number is reported as a result.

## Queue and recovery

Phase A has 36 steps. Phase B has 48 steps, including 28 fixed jobs plus two trace compilations, GPU unit checks, arch/SASS audits and all prescribed final comparisons. Queue definitions are committed in definitions/queue_a.json and queue_b.json. Bpre refreshes the compiler; B0a recalibrates only invalid sections. Fixed builds always use the saved R12b target, preserving their model inputs.

The one scheduler was launched from the immutable diag worktree. Live queue directory: `/root/r12c_work/queue`; state/logs: scheduler/; step outputs: raw/. Phase B is added atomically without restarting the scheduler. Dispatch evidence is implementation/dispatch.json. Shared lock: `/root/r12c_work/gpu.lock`.

Preflight requires six samples at 5s intervals: utilization≤5%, power≤idle+30W, hidden memory≤1024MiB, sufficient free memory and no detected external owner. During a step the guard samples every5s; interference returns75, invalidates the whole round and requeues after cooldown. These are the specified detection rules, not a claim that all possible interference is detectable. Deadline:96h. Only the scheduler polls occupancy; the assistant does not poll progress.

On resume, first read scheduler/progress.tsv once, then state.json and failed-step summaries. Analyze with `python3 analyze.py`; perform the bounded Q-A…Q-E analysis and permitted follow-ups only after data exist. Before committing new results, archive raw directories excluding .so/.cu/.o, retain MANIFEST/hashes and all cited files. Do not change frozen sources after B0a.

Next actions: complete the queued GPU numerical/arch checks and A/B observations; evaluate R12 acceptance honestly, report unresolvable comparisons and failed arms; write the requested location-specific next-round plans without implementing additional optimizations.

## Commits before this documentation commit

- `26d42559c` experiments: archive the r12b diagnostic inputs
- `3c9f97183` experiments: measure candidates at explicit past values
- `0505e4033` runtime: propagate shared gpu ownership and retry exits
- `397209def` experiments: schedule guarded r12c diagnostic rounds
- `e3a6b0453` runtime: build paged decode without executor selection
- `1fabdf3a4` analysis: carry the seed into the measured serving shortlist
- `32e2da746` backend: refill lookahead before each page using stream bytes
- `78ced0c0b` backend: accumulate deferred norm squares during output stores
- `fbab316f9` runtime: hoist watchdog context and allow compiling it out
- `aeddc6840` runtime: expose deferred norm and reducer ablation switches
- `206b0c9d0` experiments: pin explicit paged plans and report all candidates
- `650a80fea` experiments: preregister the r12c default selection rules
- `8c76a8916` experiments: prepare the frozen r12c diagnostic matrix
- `2237885af` experiments: account for the paged stream bandwidth bound
- `5d195bcb9` experiments: validate same norm class partitions for ablations
- `ccc3db5ba` experiments: freeze guarded queues and retain check evidence

## 2026-10-01 progress inspection and bounded recovery

Verified snapshot: implementation/progress_review.json and recovery_snapshot.json. At inspection, the original84 steps comprised6 done,35 pending,1 failed and42 skipped. Phase A had three accepted rounds (Llama B1 rounds0/2; Qwen3 B1 round0); B1 third rounds and both B16 cells were incomplete. Twenty fixed builds succeeded, eight were rejected; their SASS audit reported FP64=0 for20 binaries. GPU numerical unit tests, B0a calibration and final acceptance had not started. Only scheduler progress was read once; diagnosis then used specific failure files.

The guard had returned75 on24 occupied preflights and10 running-interference attempts. Those rounds remain invalid. Sufficiently short interference can escape the specified5s sampling/three-sample hidden-memory threshold; accepted guard status alone is not an unconditional contamination guarantee.

| Preliminary accepted E2E (s) | Llama B1, two-round median | Qwen3 B1, one round |
|---|---:|---:|
| vLLM | 3.105398 | 4.519084 |
| R12bN L1 | 2.924663 | 4.406450 |
| R12bN L2 | 3.436837 | 5.494247 |
| R12bP | 3.497880 | 6.312932 |
| S1P | 3.271427 | — |
| S1N | 3.724751 | — |

These are Phase A's existing binaries, not final R12c results. Three rounds are required. R10C's two Llama E2E samples were3.225005/3.922260 s despite nearly stable vLLM; its large spread prevents a resolved attribution at present. No final throughput gate is claimed.

Two problems were located and repaired:
- B0b's trace command inherited `-o plan.so.top1.candidate.so`, but the wrapper replaced only the final `plan.so` string. nvcc exited0 while writing the old candidate path, and the wrapper then crashed reading the absent trace output. `fixed_builds.py:trace_command` now replaces the actual `-o` value and catches failures per trace arm; successful fixed outputs are verified by SHA and reused. The path regression test passes. This is a diagnostic-script failure, not observed CUDA deadlock.
- All six R10G-0/1/2 builds failed with `attention coordinate changed GEMM class count`. `SearchContext::SetServingStructure` recreated ServingOptions with DN's defaulttrue, dropping DN-off when adjusting argmax/attention. Commit74ed58fc2 preserves the previous plan's norm_ss state. This completes FX-17's prescribed switch behavior; default DN, cost model, kappa and Kphase defaults are unchanged. Replay verification is queued, not yet passed.

The two previously rejected Llama B1 P-noDN controls retain their nonuniform-class rejection. No geometry is silently changed. The remaining failed R10G controls are replayed under the corrected compiler.

Recovery deliberately pauses only our scheduler while atomically editing its queue; no other user's process is signalled. A snapshot showed no running child and B0a attempts=0.46 uniquely named recovery steps replace the failed/skipped dependency chain, plus refreshed compiler, audit and report steps. Original histories stay intact. B0a/Bunit remain pending but now require the repaired compiler. Successful A rounds and20 fixed .so files are not rerun. Read definitions/queue_recovery.json and implementation/recovery_dispatch.json to distinguish recovered work from historical skipped records.

All cited failure logs, search rejections, accepted-round outputs and the SASS report are committed in recovery_evidence.tar.xz (289 files), indexed by recovery_MANIFEST.tsv. Diagnostic CPU tests now pass5/5; raw log: implementation/recovery_tool_tests.log. Compiler rebuild, host regression tests, baseline CUDA identity and six DN-off build replays run through the scheduler and shared lock. Freeze amendment preceded the first calibration; no calibration result is invalidated.

Updated source fingerprint: `df0691e26345a70daf9b138c4048f8afe776fbb69e9abac7f5cab64f7e2aef36`. Initial source freeze and17-commit delivery above remain historical checkpoints; recovery commit list is discoverable with `git log a1264e59d..tilemega`. Next step is to let the repaired, guarded queue execute and inspect it on the next user-triggered resume.
