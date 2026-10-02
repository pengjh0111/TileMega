# R12c final collection and regression attribution

Status: primary and one-time canary queues finished; specified corrections verified. Performance targets are unmet; T7 chain reconstruction is incomplete. No new GPU jobs are pending.

- Prompt SHA256: `9cd721a147925caa97530746b72671caa18b49aae5b2fb61633b32438aebca97` (`/root/Prompt/TileMega_R12c_prompt.md`).
- Baseline actual HEAD: `a1264e59de6764e7a463e184fd11ba9c621bf54e`.
- Frozen source fingerprint: `df0691e26345a70daf9b138c4048f8afe776fbb69e9abac7f5cab64f7e2aef36`; last serving C++ amendment: `74ed58fc2`. Final publication HEAD is the closing docs commit containing this report (`git rev-parse tilemega`); its full SHA and total count are supplied with delivery.
- Commit history: `implementation/commits.tsv` lists the 26 commits preceding the closing `docs: record the final r12c results` commit (27 total). Evidence HEAD: `9b7ed1bc2f36e0e796ff0b254d4b4c10466ac8a7`. All measurements precede these analysis/documentation-only commits.

## Implementation and self-checks

| ID | Status; commit | Key code and executed evidence |
|---|---|---|
| D-0 | Complete; 26d42559c | archive_r12b.py, arms.py; r12b_archive.tar.xz and MANIFEST.tsv |
| FX-12 | Complete; e3a6b0453 | python/tilemega/cli.py:329; all four decode plans are pages/L2, no decode_pg_choice |
| FX-13 | Complete; 1fabdf3a4 | lib/Solver/SkeletonSearch.cpp:878, tools/commands/compile.cpp:773; exactly one seed-origin row in each of eight shortlists, selected_classes.tsv emitted |
| FX-14 | Complete; 32e2da746 | PagedGemmTaskBody.h:130, PagedAttentionTaskBody.h:52, executor/ServingPages.cuh:116; hooks precede AcquireEmpty, count stream bytes; no per-task prefetch_ahead remains |
| FX-15 | Complete; 78ced0c0b | Backend/ServingEpilogue.h:207,253,277; register square accumulation; TN32/64/128 GPU tests pass |
| FX-16 | Complete; fbab316f9 | executor/Async.cuh:52, PageRing.cuh:74, Watchdog.cuh:36; Watch constructed outside spin, compile switch tested; EventSync.cuh unchanged |
| FX-17 | Complete; aeddc6840,74ed58fc2 | compile.cpp:340; DN/LA/watchdog controls, E2E_STAGES, measure_stub.py; preserve DN in SetServingStructure |
| FX-18 | Complete; 3c9f97183 | serving/measure_candidate.py:124; 108 per-past candidate observations, fresh instances |
| FX-19 | Complete; 206b0c9d0,5d195bcb9 | fidelity.py, pin_case.py, R12B/make_protocol_cases.py; explicit paged protocol plan; fixed geometry and partition checks |
| FX-20 | Complete; 0505e4033,5d195bcb9 | vllm_baseline.py, trace.py, cli.py, check_protocol.py; exit75 and lock ownership propagate to guard/scheduler |
| Framework | Complete; 397209def,ccc3db5ba,3206e664f,ea7a0d52c | guarded scheduler/anchors; output-path and own-orphan cleanup repaired; CPU regression tests pass |
| Analysis | Partial | analyze.py trace_review recovers page counters at actual past575; realized-chain and precise launch-gap attribution unavailable |

Host regression tests 4/4; GPU body tests 3/3; seven smoke controls pass; default generated CUDA is byte-identical to the saved reference (SHA256 `708cd9fe2db0e5f817761bc62b42e981bfc549d440d83a67c372678b499d799f`). Five architecture compilations (sm_80/89/90/100/120) pass; sm_89 executed, other paths only compiled. SASS FP64=0 for all 36 audited serving binaries. Evidence: implementation/final_contract.json, recovery_arch_report.json, raw/Bsass/sass.json and archived test logs.

Decode geometry: TM16/TN128/TK64, two stages; split qkv/o/gate_up/down/head is 4/4/1/4/1 for Llama and 2/4/1/4/1 for Qwen3. Ec256, Rq4/2 respectively. All have κ1, one CTA/SM, 160 threads, five 16384-byte pages, 8192-byte activation region, 8256-byte scratch, 91136-byte total shared memory. Defaults: D=0, watchdog compiled off, K-phase mask31, V3 poll0. Binary/CUDA paths and hashes: implementation/final_artifacts.json.

## T0–T9: evidence and results

Per-request TPOT=(E2E−TTFT)/1023; tables take the three-round median and max−min range. A “resolvable” pair exceeds the larger arm range. Floats/empty TSV cells are retained unchanged in the raw tables.

T0 — archived candidate medians (pages/l2 ms; historical data, not new clean E2E):

| Cell | Pages | l2 | Choice |
|---|---:|---:|---|
| retest-llama B1 | 4.1252 | 3.8776 | l2 |
| retest-llama B16 | 4.7550 | 4.4182 | l2 |
| qwen3 B1 | 6.0311 | 5.4836 | l2 |
| qwen3 B16 | 7.3968 | 6.4839 | l2 |
| llama B1 | 4.1252 | 3.8776 | l2 |
| llama B16 | 4.7550 | 4.4182 | l2 |
| priority-llama B1 | 4.1252 | 3.8776 | l2 |
| priority-llama B16 | 4.7550 | 4.4182 | l2 |

T1 — controlled A1 TPOT medians, ms (three rounds; complete TTFT/floor/past/slopes in results/T1.tsv and past_observations.tsv):

| Cell | R12bN L1 | R12bN L2 | R12bP | S1P | S1N |
|---|---:|---:|---:|---:|---:|
| llama_B1 | 2.858 | 3.358 | 3.413 | 3.188 | 3.637 |
| llama_B16 | 3.176 | 3.657 | 4.568 | 3.951 | 4.284 |
| qwen3_B1 | 4.304 | 5.366 | 6.168 | — | — |
| qwen3_B16 | 5.787 | 6.415 | 7.310 | — | — |

T2 — B1 TPOT effects, after/before − 1; pairwise effects are not additive. All rebuilt R10G and N-R12b controls preserve source residency/grid.

| Factor | Llama | Qwen3 |
|---|---:|---:|
| executor | +17.49% | +24.67% |
| measurement tool | +0.43% | +0.30% |
| code evolution | +4.51% | +3.87% |
| V3 | -2.58% | -2.68% |
| L2 prefetch | -1.39% | -1.22% |
| DN | -3.71% | -4.27% |
| executor at R10 geometry | +8.52% | +8.16% |
| geometry/search | -6.21% | -1.03% |
| compiler headers | -0.12% (unresolved) | -2.64% |
| watchdog runtime lower bound | -0.18% (unresolved) | -0.03% (unresolved) |

T3 — candidate-vs-E2E relative protocol difference (paged divided by nonpaged, maximum absolute difference across past192/575/1000): Llama B1 0.31%, B16 1.00%; Qwen3 B1 2.35%, B16 0.91%. None exceeds the specified 3% H3 criterion. Full 108 observations: results/T3.tsv.

T4 — S1P runtime ablations, TPOT change vs loop (results/T4.tsv):

| Arm | Llama B1 | Llama B16 |
|---|---:|---:|
| separate | -3.44% | -5.64% |
| rotate | -0.10% (unresolved) | +0.01% (unresolved) |
| kphase0 | +1.68% | +1.44% |
| L1 | -5.86% | -8.56% |
| nowd | -0.04% (unresolved) | +0.16% (unresolved) |

T5 — all four decode seeds win; prediction/measurement ordering is poor. Full 24 candidates, geometry, origins and prefill results: results/T5.tsv.

| Decode cell | Predicted ms | Measured ms | Measured/predicted | Kendall τ | Solve s |
|---|---:|---:|---:|---:|---:|
| llama_B1 | 3.373 | 3.068 | 0.910 | -1.000 | 1919.8 |
| llama_B16 | 3.994 | 3.635 | 0.910 | -1.000 | 887.8 |
| qwen3_B1 | 5.378 | 4.691 | 0.872 | -1.000 | 2005.1 |
| qwen3_B16 | 6.932 | 6.457 | 0.931 | -0.333 | 1277.3 |

All eight solve/build times are 887.8–2322.4s, above the configured600s total budget; no ≤10min claim is made. Source records and runtime stage statistics: results/runtime_structure.tsv.

T6 — fixed-geometry B1 medians, ms; choose_defaults_r12c.py selects D0 and watchdog0 by its preregistered rule.

| Arm | Llama B1 | Llama B16 |
|---|---:|---:|
| P-base | 3.208 | 3.873 |
| P-D64K | 3.322 | 4.069 |
| P-D128K | 3.334 | 4.120 |
| P-noWD | 3.051 | 3.641 |
| P-base-sep | 3.101 | 3.759 |
| P-D128K-sep | 3.039 | 3.737 |

MS controls use separate launches. DN-off raises paged B16 TPOT2.64%; LA-off changes B1/B16 by +0.60%/−0.88%. Nonpaged DN-off increases B1/B16 TPOT1.36%/1.36% in L2 and2.62%/1.46% in L1. P-noDN and P-noDN-noLA B1 are rejected for nonuniform geometry within a target class. Full values and spreads: results/T6.tsv.
D128 loop slopes are −1.89/+63.96µs per100 tokens (B1/B16), vs separate +4.59/+59.23; growing past and timing are confounded. This does not establish a past-independent drift. E2E_STAGES: decode runtime/queued/elided147/130/17 for Llama and255/226/29 for Qwen3. Split-K combines remain queued; the elided counts cover merge and argmax only.

T7 — existing instrumented single-step trace at past575; these are trace timings, not accepted E2E performance:

| Batch | Kernel span ms | Stream floor ms (902GB/s) | Span/floor | Full ring + dependency wait, mean CTA ms |
|---|---:|---:|---:|---:|
| 1 | 3.876 | 2.761 | 1.404 | 0.472 |
| 16 | 4.723 | 3.074 | 1.536 | 0.351 |

Partial: original TRACE_V2 dependency_graph rejects unqueued LA reducer stages; therefore realized chain length/span, per-link residuals and attribution are unavailable. Nonfull loader fractions0.469/0.502 are upper bounds including other work, not measured loader busy fractions. One launch cannot measure adjacent-launch gaps; ncu is unavailable. No missing metric is replaced with zero. The original B6 wrapper exited0 despite analysis errors; logs are archived.

T8 — final paired medians after one permitted canary repeat; every cell has three rounds.

| Cell | TTFT ms | TPOT mean/p50/p90 ms | E2E s | tok/s | TM/vLLM ± range | E2E/Σfloor |
|---|---:|---|---:|---:|---|---:|
| llama_B1 | 4.615 | 3.054/3.041/3.070 | 3.128 | 327.3 | 0.9925; range 0.0073 | 1.085 |
| llama_B16 | 34.138 | 3.641/3.599/3.924 | 3.759 | 4358.7 | 1.0137; range 0.0155 | 1.167 |
| qwen3_B1 | 6.437 | 4.709/4.697/4.765 | 4.824 | 212.3 | 0.9384; range 0.0024 | 1.188 |
| qwen3_B16 | 47.131 | 6.362/6.311/7.330 | 6.555 | 2499.3 | 0.9401; range 0.0013 | 1.256 |

The ratio column is the median of paired-round ratios; “range” is max−min, not an error bar. Same-cell R10C and R12bP E2E/TPOT values and vLLM rows are in results/T8.tsv. R12c improves over R12bP in every cell, but is slower than R10C in three. Floors use the archived same-cell R12b convention for all arms.

T9 — correctness and inherited R12/R12b gates:

| Gate | Result | Evidence |
|---|---|---|
| G-1 | FAIL overall; Llama B1 passes, other3 fail | Four ratio medians vs0.957/1.077/0.940/1.048 and per-cell ratio ranges; final_review.json |
| G-2 | PASS, four TM cells and four vLLM checks | gap≤0.5 ratio100%; TM maximum gap0/0.25/0/0.375 in cell order |
| G-3 | PASS | Four1024-step arms per cell; timed-repeat, phase and loop comparisons have zero token mismatches |
| G-4 | PASS, 50/50 | Explicit paged Llama B16,64steps; L1 separate/L2 separate/L2 loop/no-phase comparison |
| G-5 | FAIL overall; Llama B16 only passes | Ratios in T8 |

## Q-A…Q-E and next steps (proposals only)

Q-A [verified]: unified L2 is a major nonpaged regression: +17.49/+15.17/+24.67/+10.84% TPOT in cell order; B1 TTFT also rises0.457/0.355ms. Same R10 geometry reproduces +8.52/+8.16% (Llama/Qwen3). V3, L2 prefetch and DN individually help, while same-design code evolution adds4.51/3.87%. Tool differences are only0.43/0.30%. Runtime watchdog effects are unresolved. [inferred] These controlled pairs reject a single “DN causes slowdown” explanation; they do not yield an additive decomposition. Next: inspect task-event/FIFO overhead at ModelHarness.cuh:2867 and ServingPages.cuh WaitDependencies/Publish; no change made.
Q-B [verified]: restricted S1N is8.3/17.1% slower than searched nonpaged Llama; H4 is supported. S1P beats searched R12bP for Llama, supporting H1. H2 is cell-dependent: Qwen3 B1 paged slope69.02 vs nonpaged9.52µs/100tokens, whereas Llama B16 paged slope is lower. H3 is not supported by its3% test. Next: compare whole-past candidate objectives and actual winning geometry; no selection/model change made.
Q-C [verified, partial]: K-phase helps1.4–1.7%; rotation is indistinguishable in S1P; device loop is3.4–5.6% slower than separate launch in those controls. Compiling watchdog out helps4.9/6.0%; D64/D128 hurt rather than cover a measured gap. T7 gives paged stream-floor distance and local full/wait intersection only. [inferred] Whole-chain attribution requires LA-aware trace nodes and recorded phase-ready timing; repair TRACE_V2/analyze.py:338 and actual-past handling before claiming link residuals.
Q-D [verified, bounded]: DN helps the valid controls; LA has small, opposite-signed batch effects. B1 paged DN cannot be isolated under the prescribed unchanged geometry. The final48/84 inserted split-K combines are still queued (runtime stages minus queued differ by merge+argmax only). Next: inspect SelectServingHandoffs self-edge selection at HandoffPass.cpp:432 and runtime kHandoffAutoCombine resolution at ModelHarness.cuh:2180; no extra LA optimization made.
Q-E [verified]: paged service is now exercised and improves over R12bP, but three cells miss R10C and three miss vLLM. Every decode seed wins, despite a worse prediction; τ−1 in three cells and−1/3 in the fourth. [inferred] Candidate pricing/ranking deserves priority over longer blind search. Next: recalibrate/prioritize only with user authorization; current round changes neither cost model, κ nor K-phase defaults.

## Omissions, deviations, provenance and queue closure

- All specified FX items are implemented; diagnostics are partial at T7. B4 and the bounded3600s search-budget extension were not run. The primary matrices are closed at the user’s requested delivery; no assertion about the causal effect of longer budget is made.
- Two B1 DN-off fixed arms were rejected rather than silently changing geometry. Other28/30 fixed builds pass, including both trace builds; all seven scheduled smoke controls pass. Historical binary compiler provenance remains unknown where the original records omitted it.
- R10C old tools retain their hardcoded internal warmup/three repeats; new tools use one repeat per outer round. The R10C_new pair quantifies that difference. All named R10G/N-R12b rebuilt controls preserved grid/residency.
- Trace parsing needed a CPU-only correction after collection: stream/page statistics now use actual past575 and exclude never-launched Params rows. No serving code or calibration stamp changed. Chain/reducer handling remains unresolved, rather than fabricated.
- B0b originally failed on trace output naming, skipping42 historical dependencies; recovery reuses accepted binaries. DN-off propagation was repaired before first calibration. An owned orphan vLLM worker was proven and removed; guard cleanup was corrected without changing thresholds or other users’ processes. Original failure evidence remains archived.
- Canaries: original A1 Llama B16 round2 and B5 Llama B1 round2 were preserved in collection_evidence.tar.xz, then repeated exactly once. No canary remains after replacement. These were variability flags, not proven external interference.
- Guard preflight: six5s samples, utilization≤5%, power≤idle+30W, hidden memory≤1024MiB, required free memory and no detected external owner. Runtime interference invalidates/requeues a whole round with75. This cannot establish absence of undetectable interference. Policy and all samples are archived.
- Final archive: final_evidence.tar.xz and final_MANIFEST.tsv (6153 files,370962773 original bytes); SHA256 `69c2e2f118e2e4404059b50fd89a81075215f760df4d9c851040e63ad9e1de79`. Restore with `tar -xJf final_evidence.tar.xz` from SERVING_R12C. Archive includes current raw rounds, modes/HF/protocol records, guards, scheduler history, tables and final_metadata. Binaries/generated CUDA are excluded; paths/hashes are in final_artifacts.json. Prior r12b/recovery/orphan/collection archives preserve overwritten attempts.
- Scheduler state:90 done,1 historical failed,42 historical skipped,0 pending/running; replacements are done. The existing scheduler remains idle for dynamic queues. No new GPU tests were started during closure. Recovery, if requested later: inspect scheduler/progress.tsv once and the archived state; never reset completed steps.
- Publication: closing commits are local on tilemega. The final response records push success or supplies `git push origin tilemega`; format-patch fallback is `/tmp/round12c-patches/`.
