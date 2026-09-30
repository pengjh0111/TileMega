# R12b closure status

Baseline: `01b4ee43254c8995871ac6e1adeefedbf89e7dea`. R12b prompt SHA256: `4037520e5f69e772297ce67de581a5d0fea8158d96260e883a38863a383c9c40`. The R12 prompt remains in force; queue 4 code freeze and final HEAD are recorded by their commits.

Kernel provenance: queue 4 used the R12b implementation at `b5fd77590`, compiler/runtime source fingerprint `cfa66b2d4a69d902af7a73ff2437f0ce7989a0923db7d3a3578a4ebb572f8765` (Qpre checked it). Subsequent commits through the priority-queue change do not modify the fingerprinted compiler/runtime source set. The measured Llama/Qwen3 B1 decode `.cu` files were generated on 2026-09-29 at 10:57:16 / 16:10:30 UTC; their selected `.so` SHA256 values are `d4c3433229c4b6686b85e5644b7835845f1dd9de4a0ecdde51a34651a4b0a465` / `1d478c09225f66b9398ec9c82d82454ffdfff6f28db6c88144f19c467a00be6a`. Benchmark commands name these same libraries. Both builds enable SYNC_V3; generated decode stages contain no RMSNorm and no old codegen lag table. Thus the partial B1 figures are post-R12b results of its current nonpaged branch, not historical R10/R11 binaries.

`pg=l2` means the current nonpaged/L2-prefetch data path, whereas `mode=L2` means the event executor; these are independent choices. R12 SL-4 explicitly requires comparing `pg=pages` and `pg=l2` (both with DN). R12b EX-1 removes executor L1/L2 selection, not that PG comparison. Using the selected plan follows the test-case source specified in R12b T-5, but silently falling back when loop arms receive a nonpaged plan does not satisfy the requested loop/phase coverage. This was not caught before the protocol run; the twelve observations must not be presented as that coverage.

| Item | Commit | Code and status |
|---|---|---|
| FX-8 | `2685cccf9` | `ServingLag.h`, `ModelHarness.cuh`, `ServingRuntime.cuh`: lag edges use expanded runtime stages; aggregate rows validated; generated `kLagDependencies` removed. Host test passed. |
| FX-11 | `41fd3c52a` | `SkeletonSearch.cpp`, `Codegen.cpp`: paged κ=1, otherwise no kPhase; host rejects unsafe phase rows. Build passed. |
| FX-9 | `3947690ec` | `Watchdog.cuh`, `EventSync.cuh`, `PageRing.cuh`, both executors, `ServingRuntime.cuh`: twelve wait sites and mapped 12-word record. `serving_watchdog` CUDA self-test passed. |
| FX-10 | `75639c375` | `HandoffPass.cpp`, `ModelHarness.cuh`: split lm_head cannot use argmax LA. Build passed. |
| EX-1/2 | `a5f2eb8ff`, `3947690ec` | `compile.cpp`, `cli.py`, `engine.py`, `ServingRuntime.cuh`: L2-only selection and service; L1 remains explicit cross-check; L1 step-loop launch removed; dead auto-handoff branch removed; exit 75 propagates. Runtime validation pending. |
| T-1 | `d6c21b3d0` | `smoke.py`: same-instance L2 separate/loop and L1 with token and KV comparison; integrated before candidate timing. S-0 passed. |
| T-2/3/4 | `3947690ec`, `a5f2eb8ff` | Runtime phase mask and placement ablation; V3 poll macro and compile options. Build passed; performance controls pending. |
| T-5/6 | `63e47f4bd`, `91ed843c6` | Four-arm C-2, protocol cases, fidelity and fixed default rule. Python syntax passed; GPU checks pending. |
| Queue | `91ed843c6`, `85f215557` | Serial GPU queue with 75-only retry. Queue 4 and its delayed full rerun were cancelled by the user on 2026-09-30; the replacement queue runs only paired E2E for both models at B=1/16. |

The R12 lag-index checker shows a concrete stale generated Llama decode candidate with **99 spec stages and 163 runtime stages**; its first KV edge reads runtime `kGemmCombine` instead of attention, and its producer reads `kFusedAttention` instead of merge. The candidate has κ=1. Evidence: [lag_index_evidence.txt](lag_index_evidence.txt). This proves the numbering defect exists; it does **not** establish that it alone caused F-338's B16 hang. S-0's second-launch reproduction and watchdog record decide that question.

| Check | Status | Evidence |
|---|---|---|
| Host lag unit | PASS | `ctest -R '^serving_lag$'` |
| Watchdog CUDA self-test | PASS | `ctest -R '^serving_watchdog$'` |
| Compiler and targeted CUDA units build | PASS | `/root/r12_work/r12b_code_build.log` |
| S-0, Llama B16 plan + 64-step smoke + default candidate timing | PASS | `/root/r12_work/r12b_s0/s0.done`; `smoke64/smoke.json`: L2 separate, L2 loop, L1 separate, 0 token/KV mismatches, watchdog null; selected plan pg=pages, mode=L2, κ=1. Synthetic candidate timing: L2 4.2182 ms/step (`measure/measurements.json`), not an E2E result. |
| S-1 / S-1b / S-3 | S-1/S-1b PASS; S-3 pending | Queue 3 completed S-1 and S-1b; S-3 was blocked by external GPU memory. |
| Final plans | Built, 8/8 selected `.so` files | `runs/r12b-{llama,qwen3}/plans.json`; all selected decode plans use pg=l2 and L2 execution. |
| EV-3 four cells, C-1/C-2, SL-5 | Partial E2E; full checks pending | `partial_results.json`, `partial_results.tsv`; paired B1 rounds only. |
| Fresh-process token comparison | 12/12 completed processes passed; remaining 38 cancelled by user | `protocol_partial.json`; selected nonpaged pg=l2 plan, so loop labels fall back to separate launches and do not exercise paged-loop/K-phase protocols. |
| Architecture compilation and FP64 SASS audit | Pending | no claim before actual audit |

| Gate | Status | Basis |
|---|---|---|
| G-1 end-to-end uplift | Pending | EV-3 not run |
| G-2 C-1 | Pending | EV-3 not run |
| G-3 C-2 four arms | Pending | EV-3 not run |
| G-4 50 fresh processes | Not met; remaining tests cancelled by user | 12/12 completed; original requirement 50/50. No synchronization/race conclusion is claimed. |
| G-5 TM/vLLM ≥1 | Pending | EV-3 not run |

S-1 256-token TPOT p50 (ms): Llama B1 pages 3.1683 / l2 3.5809; B16 pages 3.6076 / l2 4.0448. S-1b at B16: phase mask 31 3.6035, all off 3.6639; D=0 3.6024 versus D=128 KiB 5.3156; poll 0 3.6055 versus 200 ns 3.6823. The preregistered rule selected mask 31, D=0, poll=0 (`defaults.json`). These are control results, not EV-3.

Queue 1 failed on illegal prefill control geometry and a prefill smoke range check (`243154920`). Queue 2 failed when smoke called the unsupported nonpaged step loop and the compiler fingerprint was stale (`8a1746013`). Queue 3 reached both controls and Llama build, then Qwen3 paged seed failed the obsolete attention/GEMM smem union check; Llama bench, protocol and S-3 hit CUDA OOM while approximately 45 GiB was allocated by a GPU process invisible in the compute-app list. Queue 4 completed both model builds, but Llama/Qwen3 benchmarks failed during vLLM KV-cache initialization: requests for 33.22/36.26 GiB exceeded the then-free 10.44/32.01 GiB. These errors propagated as exit 1 rather than retryable 75. C-1/C-2 were consequently skipped. The protocol check reached 12 passing children and then retried external occupancy (75) until the user cancelled it.

Deviations and limits: S-0 restricts its solver domain to TN=128, TK=64 and split ∈ {1,2,4,8} to expose the repeated-launch bug without recalibration; this is a diagnostic build, not the final search. The stale lag-index evidence comes from a Llama B1 decode candidate with split-K, because the archived B16 top-1 file lacks the old lag table; S-0 establishes that a repaired B16 diagnostic plan no longer hangs, but cannot isolate which defect caused F-338. `E2E_LAG` prints validated numeric TaskKind values rather than the requested symbolic kind names. Q2's fixed-geometry control allows its restricted split coordinate to be selected by the compiler rather than hardcoding SL-1's class-specific split vector. These choices do not change EV-3's search or gates.

The first S-0 attempt had a test-environment failure: its candidate subprocess used `/usr/bin/python3`, which lacks `transformers`. The successful rerun and queued commands use `/root/venvs/tilemega-torch213-cu126/bin/python`; no CUDA correctness conclusion is drawn from the first attempt.

S-0 passed. Prior `queue_run*/` directories remain evidence; queue 4 reused Q0–Q4 from queue 3 because the calibration, exports, control kernels and default rule were unaffected. Its cancellation is recorded in `queue_run4/QUEUE_CANCELLED` and `Q9.cancelled`.

The three-hour [delayed retest](delayed_retest.py) reached its start time but remained blocked on queue 4. It was cancelled before starting any retest step when the user replaced the full rerun with E2E-only measurements on both models, B=1/16. `retest_3h/timer.json` records cancellation.

Completed historical measurements (1024 generated tokens, B1; one timed generation per listed round, not the required three-round matrix):

| Model / round | TM TTFT / TPOT ms | vLLM TTFT / TPOT ms | TM / vLLM E2E s | TM / vLLM tok/s | Throughput ratio |
|---|---|---|---|---|---|
| Llama / 0 | 4.479 / 3.356 | 25.218 / 3.062 | 3.4381 / 3.1573 | 297.84 / 324.33 | 0.9183 |
| Qwen3 / 0 | 6.458 / 5.370 | 36.324 / 4.406 | 5.5003 / 4.5439 | 186.17 / 225.36 | 0.8261 |
| Qwen3 / 1 | 6.440 / 5.367 | 35.335 / 4.388 | 5.4967 / 4.5245 | 186.29 / 226.32 | 0.8231 |
| Qwen3 / 2 (TM only) | 6.440 / 5.366 | — | 5.4955 / — | 186.33 / — | — |

TPOT above is `(E2E−TTFT)/1023`. TM measured step mean/p50/p90 (ms) are Llama 3.343/3.336/3.367 and Qwen3 round 0 5.361/5.370/5.408, round 1 5.356/5.366/5.405, round 2 5.356/5.365/5.405. Accepted historical rounds passed their recorded before/after power and exclusivity checks; continuous hidden-allocation monitoring was absent, and external allocations later interrupted both matrices. Treat these numbers as partial observations rather than G-1/G-5 results. Source hashes and guard counts are preserved in `partial_results.json`.

The replacement [priority queue](priority_queue.json) invokes only `python -m tilemega bench` and its CPU report through [priority_bench.py](priority_bench.py), reuses the eight `.so` files, and writes fresh results under `runs/r12b-priority-{llama,qwen3}/`. Each of four cells has warmup 1, paired repeats 3, 1024 new tokens, fixed prompt IDs and vLLM 0.30.0. No remaining 50-process, HF/mode, S-3 or rebuild step is scheduled. This user-directed reduction leaves those original gates unfulfilled.

The fixed guard policy is `priority_policy.json`: before starting, six idle samples 5 s apart must show no compute PIDs, memory ≤256 MiB, utilization ≤5% and power ≤51.81 W (unchanged 21.81+30 W threshold). During bench, sample every 5 s; foreign visible PIDs reject immediately, and invisible allocation >256 MiB for three consecutive samples rejects the attempt. Existing per-request before/after power guards still apply. Initialization OOM and occupancy return 75 to the existing 20-minute retry queue (13 attempts maximum). `bench_acceptance.json` distinguishes a complete accepted model benchmark from partial/rejected files; `occupancy_guard.jsonl` preserves observations. These guards detect the observed interference mechanism; they cannot establish exclusivity for all possible invisible workloads.

On resumption, read `priority_run/progress.tsv` once and each model's `bench_acceptance.json`. The agent does not poll the queue. The old queue and delayed launcher must remain cancelled.

If the second launch still stalls, the next step is the site/row/need/value in the mapped watchdog record, not another open-ended timing run. If any performance gate fails, compare the L2 placement and phase ablations before changing geometry or the execution protocol.
