# TileMega Round 1 — M1 diagnosis (EX-D1 + EX-D2 + EX-C1)

## 1. Identification

| item | value |
|---|---|
| baseline commit | `4cdebdf13495bb9a11c2be839755d4faff0bea1f` |
| prompt SHA256 | `9f7d74272afd6dda76d2f6c994211911279ee25ec870bb2bd9a96955fcc61cc3` |
| device | NVIDIA GeForce RTX 4090 (sm_89), driver 610.43.02, CUDA 12.8 |

Commits, in the order §9 fixes:

| # | commit | message |
|---|---|---|
| 1 | `477f5432` | `trace: measure the globaltimer resolution` |
| 2 | `11c0199f` | `trace: add per-slot execution timestamps` |
| 3 | `37c4a1a6` | `trace: reconstruct hops and head-of-line blocking` |
| 4 | `f2f5b558` | `place: add cross-stage continuous round robin` |
| 5 | `5c04690e` | `place: measure the rotated placement arms` |
| 6 | `fc5dd139` | `place: bound the available scheduling headroom` |
| 7 | `17b014d5` | `runtime: drop the unread task flag and stale header` |
| 8 | `0eac374f` | `experiments: add the sm_120 runners for round one` |
| 9 | (this commit) | `docs: record the round one diagnosis` |

The repository's only remote is `origin`
(`https://github.com/pengjh0111/TileMega.git`) and this environment has no push
rights to it, so §9's fallback applies: the nine commits are exported as
`git format-patch 4cdebdf1..HEAD` into **`/tmp/round1-patches/`**
(`0001-trace-measure-the-globaltimer-resolution.patch` …
`0009-docs-record-the-round-one-diagnosis.patch`).

## 2. Gate results

### §3.7 EX-D1

| gate | verdict | number |
|---|---|---|
| D1-a correctness, 50 fresh processes × 4 cells | **PASS** | 50/50 in all four cells |
| D1-b default-build SASS byte-identical | **PASS** | both diffs 0 bytes, 6802420 bytes each |
| D1-c trace on/off paired 25 rounds, ratio ≤ 1.02 | **PASS** | worst 1.0167 |
| D1-d reconstruction within 5% of `l2_ms` | **FAIL** | 12.10%–14.51% (§6 below) |
| D1-e `hop(j) < 0` count is 0 | **PASS** | 0 in all four cells |

### §4.6 EX-D2

| gate | verdict | number |
|---|---|---|
| D2-a modes 0 and 5, 50 fresh processes each cell | **PASS** | 50/50 in all eight cells |
| D2-b hand-computed `base[]` matches the code | **PASS** | 30/30 stages, grid 256, 0 mismatches |
| D2-c 8 combos × 4 cells × 25 rounds, no missing sample | **PASS** | 800/800 |
| D2-d `FORK` line produced by the script | **PASS** | rule 2, see §4 |
| D2-e bounds + simulated makespan, reference and real width | **PASS** | 4 reference + 2 real-width cells |

### §5 EX-C1

| gate | verdict | number |
|---|---|---|
| `kLastTaskOfStage` gone, no readers | **PASS** | `git grep` empty over lib/ include/ tools/ test/ |
| `GeneratedLlamaRuntime.cuh` gone, no includers | **PASS** | absent, no `#include` |
| build passes | **PASS** | 77/77 targets |
| all CTest passes | **PASS** | 44/44, incl. `frontend_import_test` |
| generated `.cu` byte-identical | **PASS** | gqa2 and mha4 both identical |

## 3. `verify.py` output

Run it directly: `python3 docs/experiments/PLACE_ROTATE/verify.py`. It re-derives
every gate above from the committed raw logs and dumps — it re-runs `analyze.py`
on the committed dumps, recomputes the perturbation ratios from the per-round
logs, re-runs `summarize.py` and `fork.py`, re-runs `base_example.py`, rebuilds
the generator and byte-compares a fresh `.cu` against `raw/c1/*_before.cu`, and
runs the build and CTest. It reads no summarized conclusion. Complete output,
also committed as `raw/verify.txt`:

```
[PASS] D1-a gqa2 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/TRACE_V2/raw/final/correct_gqa2_s4
[PASS] D1-a gqa2 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/TRACE_V2/raw/final/correct_gqa2_s128
[PASS] D1-a mha4 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/TRACE_V2/raw/final/correct_mha4_s4
[PASS] D1-a mha4 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/TRACE_V2/raw/final/correct_mha4_s128
[PASS] D1-b: default-build SASS diff bytes {'gqa2.diff': 0, 'mha4.diff': 0}
         evidence: /root/TileMega/docs/experiments/TRACE_V2/sass_identity
[PASS] D1-c: paired l2_ms median ratio gqa2_s4=1.0157 gqa2_s128=1.0167 mha4_s4=1.0163 mha4_s128=1.0149, worst 1.0167 <= 1.02
         evidence: /root/TileMega/docs/experiments/TRACE_V2/raw/final
[PASS] D1-e: 4 cells, hop(j) < 0 count 0
         evidence: /tmp/round1-verify-az92bih2/analysis/analysis.tsv
[FAIL] D1-d: reconstructed critical path vs measured l2_ms, worst gqa2_s4 14.51% (cause reported in analysis.md, definition not relaxed)
         evidence: /tmp/round1-verify-az92bih2/analysis/analysis.tsv
[PASS] D2-a p0 gqa2 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p0_gqa2_s4
[PASS] D2-a p0 gqa2 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p0_gqa2_s128
[PASS] D2-a p0 mha4 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p0_mha4_s4
[PASS] D2-a p0 mha4 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p0_mha4_s128
[PASS] D2-a p5 gqa2 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p5_gqa2_s4
[PASS] D2-a p5 gqa2 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p5_gqa2_s128
[PASS] D2-a p5 mha4 s4: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p5_mha4_s4
[PASS] D2-a p5 mha4 s128: 50/50 fresh processes RESULT status=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final/correct_p5_mha4_s128
[PASS] D2-b: BASE_EXAMPLE grid=256 stages=30 mismatches=0 verdict=PASS
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/log/base_dump_gqa2_s4_p5.out
[PASS] D2-c: 32 arm x placement x cell combinations x 25 rounds, none missing
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/final
[PASS] D2-d: FORK rule=2 r_neither=0.2240 ci=[0.1891,0.2791] r_full=0.6705 ci=[0.6598,0.7392] cells=4
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/fork.txt
[PASS] D2-e: reference cells 4/4, real-width {'128': 'PASS', '4': 'PASS'}
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/headroom.tsv
[PASS] C1 flag removed: no kLastTaskOfStage under lib/ include/ tools/ test/
         evidence: git grep kLastTaskOfStage
[PASS] C1 header removed: GeneratedLlamaRuntime.cuh absent and unincluded
         evidence: /root/TileMega/include/tilemega/Codegen/tasks/GeneratedLlamaRuntime.cuh
[PASS] C1 build: tilemega-compile builds from the current tree
         evidence: /root/TileMega/build-portable
[PASS] C1 generated sources: gqa2=identical mha4=identical
         evidence: /root/TileMega/docs/experiments/PLACE_ROTATE/raw/c1
[PASS] C1 ctest: 100% tests passed, 0 tests failed out of 44
         evidence: /root/TileMega/build-portable

VERIFY gates=25 hard=25 failed=1 list=D1-d
```

Exit status 1, as §10 requires when a hard gate fails.

## 4. Fork

```
FORK rule=2 r_neither=0.2240 ci=[0.1891,0.2791] r_full=0.6705 ci=[0.6598,0.7392] cells=4
```

Rule 2: the placement gain is large and statistically unambiguous, and it does
not survive intact into the correct kernel. §4.4 routes the next round to
**EX-E1** (a Plan contract the solver writes and the host materializes) and
**EX-S2** (EFT placement and ordering). That ordering is also what the offline
numbers argue for independently: the busiest worker's own queue is 81–85% of
measured `l2_ms` on the reference models and 93–97% at real width, while the
entire critical-path cost of synchronization is 10.24 µs.

## 5. Key numbers

| quantity | value |
|---|---|
| `%globaltimer` tick | 1024 ns, cross-SM spread 0 ns |
| `clock64` resolution / cross-SM offset spread | 40 cycles (≈16 ns) / 3.54e9 cycles (≈1.41 s) |
| per-SM offset uncertainty after calibration | 18.99 ns (1σ) |
| trace v2 perturbation, worst cell | 1.0167 |
| hop p50 / p90 / max | 1024 ns / 1024–7168 ns / 46–54 µs |
| `hop(j) < 0` | 0 |
| `queue_lb` / measured `l2_ms`, reference | 81.2%, 84.6%, 83.0%, 84.1% |
| `queue_lb` / measured `l2_ms`, real width | 97.0% (seq 4), 93.0% (seq 128) |
| `cp_lb_sync − cp_lb_nosync` | 10.24 µs in every cell |
| workers receiving any task at seq=4 | 16 of 256 |
| reclaimable HOL, fraction of stall time | 18.85%, 52.99%, 56.80%, 63.98% |
| max queue length, mode 0 → mode 5 | 30→1, 34→18, 60→2, 80→47 |
| cross-worker edge fraction, mode 0 | 95.1%–99.6% |
| mode 5 / mode 0 `l2_ms`, `full` arm | 0.6582, 0.7421, 0.6521, 0.7461 (pooled 0.6705) |
| mode 5 / mode 0 `l2_ms`, `neither` arm | pooled 0.2240 |
| EFT makespan / measured, reference | 0.34, 0.43, 0.17, 0.28 |
| EFT makespan / measured, real width | 0.26 (seq 4), 0.39 (seq 128) |

## 6. Deviations from the prompt

1. **D1-d fails and was not repaired.** A critical path whose node weight is the
   measured `run_end − run_begin` and whose cross-worker edge weight is the
   measured hop p50 reconstructs `l2_ms` to 12.10%–14.51%, against the 5% gate.
   Per H7 the definition was not relaxed and no parameter was tuned. Instead the
   residual was measured: decomposing the reconstructed chain into task, wait,
   pre-run barrier, publish and gap sums exactly to the chain span, and the one
   term §3.6's node weight omits is the producer's own `publish_end − run_end`
   (43–106 µs per chain). A separate, clearly labelled column adding only that
   term brings the error to 2.08%–3.17%. Both numbers are reported; the §3.6
   definition remains the primary one. Recorded as F-132.
2. **Commit trailers.** The session environment asks for a
   `Co-Authored-By: Claude Opus 5` trailer. The repository's `CLAUDE.md` and
   this prompt's §1.1 both forbid trailers of any kind. The repository
   convention was followed: all nine commits are a single line under 72
   characters with no trailer.
3. **`clock64` columns added to `TaskTraceV2` and `slots.tsv`.** This is §3.5's
   own conditional branch firing (the measured tick, 1024 ns, exceeds 100 ns),
   not a departure from §3.1: the field order §3.1 fixes is preserved as a
   prefix and `run_begin_clk`/`run_end_clk` are appended.
4. **Grid is 256, not the 128 used in §4.1's illustration.** The hand-computed
   `base[]` example is therefore given at the actual grid the binaries run at.
   All 30 stages were checked, not only the first five.
5. **`DOC_RESTRUCTURE/audit.py` is now inconsistent with the tree, and was not
   run.** Its check K18 asserts `kLastTaskOfStage` still exists, which EX-C1's
   mandated deletion makes false; its check A5 asserts zero code change since
   `059f8532`, which this round's commits 2, 4 and 7 already falsify. H1 forbids
   modifying it, and running it writes `facts.tsv`/`audit_summary.txt` under
   `docs/experiments/`, which H1 also forbids, so it was left untouched. It is
   not registered in CTest. A later round should update K18 or retire the
   auditor.
6. **`realwidth.sh` defaults to `build-portable`, not `build-phase12`.** The
   first real-width attempt failed: `build-phase12/tools/tilemega-compile`
   predates the schedule table in `RuntimeVariantDesc` and emits an initializer
   that no longer compiles. The failure and its cause are recorded in
   `raw/realwidth/`; the rerun with the current-tree generator produced both
   cells, and no reference-model number was substituted at any point.
7. **`sass_identity/meta.tsv` records `head_commit = fc5dd139`.** The check was
   re-run on the working tree that became commit 7, so `head_commit` names its
   parent — a file cannot record the hash of the commit that contains it. The
   tree compiled carried every source edit of this round, EX-C1's deletions
   included, and both diffs were empty.
8. **`docs/experiments/**/*.log` is matched by `.gitignore` line 32**, which H1
   forbids editing, while H5 requires the raw logs to be committed. They were
   added with `git add -f`. Compiled binaries under `raw/bin/` were deliberately
   left untracked.

9. **The mandated real-width run left new files in `docs/experiments/REALMODEL/raw/`.**
   §4.5 requires driving `REALMODEL/run.sh`, and that script writes its own
   `correctness_*.tsv`, `timing_*.tsv` and `size_*.tsv` for each cell it runs.
   No existing file there was modified. Those side-effect files and the compiled
   `model_s*` binaries were left untracked; the logs this round's own conclusions
   rest on are under `PLACE_ROTATE/raw/realwidth/` and are committed.

9. **`REALMODEL/run.sh` left new files in its own directory.** §4.5 makes the
   real-width run mandatory and that script writes `raw/correctness_r1w_s*.tsv`,
   `raw/timing_r1w_s*.tsv` and `raw/size_r1w_s*.tsv` under
   `docs/experiments/REALMODEL/`. No existing file there was modified. They were
   left untracked so this round adds nothing to a directory H1 fences off; the
   copies this round's conclusions rest on live under
   `PLACE_ROTATE/raw/realwidth/` and are committed. Compiled binaries under
   `raw/bin/` and `raw/realwidth/model_s*` are likewise untracked.

## 7. Open questions, and what to run on sm_120

1. Does the mode-5 gain hold on Blackwell, where the SM count, the L2 topology
   and the `%globaltimer` tick may all differ? Run, in order:

   ```
   bash docs/experiments/TRACE_V2/run_sm120.sh
   bash docs/experiments/PLACE_ROTATE/run_sm120.sh
   ```

   Both refuse to run unless `nvidia-smi --query-gpu=compute_cap` reports
   `12.0`, reject inherited `TILEMEGA_*` overrides, and write `PASS`/`FAIL` to
   `raw_sm120/status.txt`. **Neither has been run on sm_120**; both were checked
   only by their own CPU-side `SELF_CHECK=1` path on the 4090, which touches no
   GPU. The Blackwell run must produce its own `FORK` line rather than inherit
   this one.
2. Re-measure the `%globaltimer` tick first (`run_sm120.sh` does this before
   anything else). If it is finer than 1024 ns, the per-hop distribution becomes
   informative at single-hop granularity, which it is not here.
3. The gap between `r_full = 0.6705` and `r_neither = 0.2240` is the part of the
   rotated placement's headroom that the current publish protocol re-serializes.
   Whether that is EX-E3's six-step protocol or EX-E4's prefetch is not
   separable from this round's data.
4. Whether an EFT schedule of the kind `headroom.py` simulates is legal under a
   window executor is untested; F-130's lifting and elision arguments assume
   strict FIFO.

## 8. §0 exclusions — per-item confirmation

| excluded | confirmation |
|---|---|
| change the sync protocol (`NotifyTask`/`ArriveEvent`/`EventPoll`/`WaitTaskDependencies`) | Untouched. With `TILEMEGA_TRACE_V2` at its default 0 the SASS of both reference kernels is byte-identical to baseline (D1-b), which is a stronger statement than source identity. |
| reduce the CTA barrier count | Untouched. Trace v2 added no `__syncthreads()`, no `__threadfence()` and no atomic; all five stamps are `threadIdx.x == 0` stores next to existing barriers. |
| introduce an execution window W or change FIFO semantics | Untouched. `max_worker_task_refs` and the stage-major queue construction are unchanged; mode 5 changes only which worker owns a task. |
| change CG's `tilemega.placement` | Untouched. No frontend or dialect file was modified this round. |
| touch `BuildVariantSchedule` | Untouched. `lib/Codegen/Codegen.cpp` is unmodified; the round's only source changes are the three files H1 permits. |
| migrate scheduling decision authority | Untouched. Mode 5 is a host-side compile-time heuristic, deliberately not a solver output — which is precisely what F-136 records as the reason EX-E1 is next. |
| change `CostModel` or `ChainDP` | Untouched. No file under `lib/Analysis` or `lib/Solver` was modified. |
| add prefetch | Untouched. No TaskBody was modified. |
| consume `TaskPlacement::slot` | Untouched. Still written and unread; explicitly deferred by §5 and recorded as in-progress in the TODO ledger. |
