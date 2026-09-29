# R12b closure status

Baseline: `01b4ee43254c8995871ac6e1adeefedbf89e7dea`. R12b prompt SHA256: `4037520e5f69e772297ce67de581a5d0fea8158d96260e883a38863a383c9c40`. Current rerun code freeze HEAD: `8a1746013` (documentation commit follows). The R12 prompt remains in force.

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
| Queue | `91ed843c6` | `queue.py` (51 lines), `queue.json`: serial GPU queue, 75-only retry. First run ended with test-path failures; repaired queue is pending rerun. |

The R12 lag-index checker shows a concrete stale generated Llama decode candidate with **99 spec stages and 163 runtime stages**; its first KV edge reads runtime `kGemmCombine` instead of attention, and its producer reads `kFusedAttention` instead of merge. The candidate has κ=1. Evidence: [lag_index_evidence.txt](lag_index_evidence.txt). This proves the numbering defect exists; it does **not** establish that it alone caused F-338's B16 hang. S-0's second-launch reproduction and watchdog record decide that question.

| Check | Status | Evidence |
|---|---|---|
| Host lag unit | PASS | `ctest -R '^serving_lag$'` |
| Watchdog CUDA self-test | PASS | `ctest -R '^serving_watchdog$'` |
| Compiler and targeted CUDA units build | PASS | `/root/r12_work/r12b_code_build.log` |
| S-0, Llama B16 plan + 64-step smoke + default candidate timing | PASS | `/root/r12_work/r12b_s0/s0.done`; `smoke64/smoke.json`: L2 separate, L2 loop, L1 separate, 0 token/KV mismatches, watchdog null; selected plan pg=pages, mode=L2, κ=1. Synthetic candidate timing: L2 4.2182 ms/step (`measure/measurements.json`), not an E2E result. |
| S-1 / S-1b / S-3 | Pending rerun | First and second queues stopped during S-1; `queue_run*/progress.tsv`. |
| EV-3 four cells, C-1/C-2, 50 fresh processes, SL-5 | Pending | `runs/r12b-{llama,qwen3}/`, `protocol/`, `fidelity.json` |
| Architecture compilation and FP64 SASS audit | Pending | no claim before actual audit |

| Gate | Status | Basis |
|---|---|---|
| G-1 end-to-end uplift | Pending | EV-3 not run |
| G-2 C-1 | Pending | EV-3 not run |
| G-3 C-2 four arms | Pending | EV-3 not run |
| G-4 50 fresh processes | Pending | protocol queue not run |
| G-5 TM/vLLM ≥1 | Pending | EV-3 not run |

No complete S-1, S-1b, S-3 or EV-3 result is available yet. First queue Q2 used an illegal uniform prefill geometry; Q5/Q6 rejected prefill candidates because smoke applied decode's past interval to prefill. Commit `243154920` repaired those defects. Second queue completed one S-1 pages/B1 arm (256-token TPOT p50 3.1713 ms) but its nonpaged decode candidate failed smoke: the ABI exports `tm_plan_launch_steps` but returns an error for nonpaged plans. Q5/Q6 were blocked by a stale compiler source fingerprint after the Python change. Commit `8a1746013` skips the loop arm for nonpaged smoke and adds a queue preflight that rebuilds the compiler and checks its fingerprint. Q11 lacked `plans.json` in both runs. Neither queue reached EV-3; these isolated control timings do not establish an end-to-end gate. The fallback `defaults.json` is provisional and Q4 will rewrite it after S-1b. SL-5 will read top-3 predictions and measured L2 candidates.

Deviations and limits: S-0 restricts its solver domain to TN=128, TK=64 and split ∈ {1,2,4,8} to expose the repeated-launch bug without recalibration; this is a diagnostic build, not the final search. The stale lag-index evidence comes from a Llama B1 decode candidate with split-K, because the archived B16 top-1 file lacks the old lag table; S-0 establishes that a repaired B16 diagnostic plan no longer hangs, but cannot isolate which defect caused F-338. `E2E_LAG` prints validated numeric TaskKind values rather than the requested symbolic kind names. Q2's fixed-geometry control allows its restricted split coordinate to be selected by the compiler rather than hardcoding SL-1's class-specific split vector. These choices do not change EV-3's search or gates.

The first S-0 attempt had a test-environment failure: its candidate subprocess used `/usr/bin/python3`, which lacks `transformers`. The successful rerun and queued commands use `/root/venvs/tilemega-torch213-cu126/bin/python`; no CUDA correctness conclusion is drawn from the first attempt.

S-0 passed. After repairing two validation-queue failures, a third unattended queue uses the model venv, `PYTHONPATH=/root/TileMega/python`, `TILEMEGA_BIN=/root/TileMega/build-phase12/tools/tilemega`, and `TILEMEGA_GPU_LOCK=/root/r12_work/serving_gpu.lock`. On resumption, first read `queue_run3/progress.tsv`; do not restart a running queue. `queue_run/` and `queue_run2/` are preserved as failure evidence.

If the second launch still stalls, the next step is the site/row/need/value in the mapped watchdog record, not another open-ended timing run. If any performance gate fails, compare the L2 placement and phase ablations before changing geometry or the execution protocol.
