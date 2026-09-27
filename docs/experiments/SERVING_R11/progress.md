# R11 progress ledger (2026-09-26)

Baseline: `4bf26fb85ada63fccea101b323f3d7a09351eca2`. Prompt SHA256:
`b4b13f18596924eac17624de218735bcf8196878862dd7a6d8f00d67db71e62a`.

This ledger retains earlier queue snapshots for provenance. The latest state
is at the end; earlier statements that a queue is running are superseded by
its later completed-result entry.

The user reduced batch-dependent performance sweeps to B=1 and B=16 for both
models. This is four end-to-end cells and eight plans, not the original ten
cells and twenty plans. The original ten-cell G-9 cannot be inferred from the
four-cell result. See `scope_amendment.md`.

The R10 baseline closure is complete on the immutable baseline worktree:
four same-session full-request TileMega/vLLM comparisons, eight SASS audits,
C-1/C-2, exact request DRAM floors, and the Qwen3 pruning timeout are in
`r10_control/`. Four-cell throughput-ratio geometric mean is 1.0038. All
eight plan solves exceeded the 600-second target.

R11 implementation with direct evidence:

| Workstream | Current state | Evidence |
|---|---|---|
| OPS-1 | Unified tool and host runner implemented; clean build 142.21→140.94 s, default executable targets 122→27, ctest 90/90 | `ops/` |
| PG-0 / PG-1 | L2 prefetch and paged decode executor implemented; four 50-process protocol cells each passed for each mode (200/200 per mode) | `protocol_results.json`, `pages/` |
| AT-3 | Independent-warp decode attention and 60 small-shape cases implemented | `pages/attention_torch.json` |
| AR-1 | Capability-dispatched paths, five target-architecture compile/SASS checks, host tensor-map checks and native sm_89 hwcheck complete; other targets remain compile-only by P-21 | `arch_primitives/`, `single_page_loader/arch_compile.json` |
| SV-18 | Two-phase paged Level 1/individual fluid modes and 8/16 KiB page-coordinate scan implemented; restricted Llama B1 CPU search ranked 16 KiB first in 86 evaluations | `solver/` |
| TF-1 | Access proofs and IR pass implemented; real-model norm→GEMM decisions carry original input/weight buffers into the paged GEMM activation loader. Four disjoint norm handoffs now pass one IR rewrite on each real model. Stage removal/replanning, last-arriver/direct execution, and solver pricing remain open | `handoff/` |
| OPS-2 / SY-1 | End-to-end CLI and calibrated waits implemented; full cache/run acceptance remains open. SOLO+RED+BARRIER_V2 completed 200/200 fresh-process token comparisons; fixed-geometry performance changed by less than 0.4% in the two selected cells | `ops/`, `calibration/`, `sync/`, `sync_ablation/` |
| EV-2 | Final solved-plan matrix not yet run; fixed-geometry PG ablation completed at Llama B1 and Qwen3 B16, and both PG modes regressed against off | `pg_ablation/` |

The requested PG-1 diagnostics are available for all four B endpoints in
`page_diagnostics/summary.tsv` with raw evidence archived. At past 575 the
instrumented realized chains have 135/119/232/204 links (Llama B1/B16,
Qwen3 B1/B16). Their spans are 6.383/6.667/9.985/10.928 ms against exact
CG floors of 2.537/2.826/3.573/4.583 ms; residual bubbles are
28.5/32.3/27.6/31.1 µs per link. The page-full∩dependency-wait means per CTA
per step are 3.552/3.744/7.229/7.372 ms. Median device-visible launch gaps
are 3.072/4.096/3.072/3.072 µs. Trace instrumentation perturbs the kernel;
non-instrumented performance is measured separately.

`verify.py` currently reports 14/18 structural checks PASS. K-6 requires
the unfinished last-arriver runtime handoff, K-11 requires handoff search pricing,
K-15 requires EV-2 measurement policy and rounds, and K-16 requires final
serving SASS audit. These are open items, not waived checks.

The first production-speed PG control holds geometry and L2 placement fixed.
Llama B1 off/L2/pages E2E is 3.3439/3.6652/6.3403 s; Qwen3 B16 is
6.0990/6.4943/10.9151 s. The page protocol remains correct, but PG-1 is
currently 1.90×/1.79× slower than off. Instrumented chain attribution puts
5.405 of Llama B1's 6.383 ms in gate/up, down and lm_head. A targeted
8/16 KiB control then reduced Llama B1 E2E from 6.3300 to 4.7511 s with
the same 101376 B shared allocation, leaving 1.42× over PG off. This is a
larger page-layout effect than the Level 1 prediction. Isolated full-grid
class controls measured paged/standard GEMM stage ratios of 1.29/1.50/1.99/
3.15 for QKV/gate-up/down/lm_head; the latter three class gaps sum to about
3.02 ms per decode step under an additive approximation, near the observed
2.93 ms/token full-request regression. This localizes the immediate fix to
`PagedGemmTaskBody` and the page-ring transfer protocol.

A corrected multi-page B loader now routes each vector by the CuTe physical
page in one pass. The initial logical-range shortcut failed a numerical test
and was discarded. The corrected version passed 49 small-shape cases, 50
fresh-process Llama B1 protocol checks, and one 1024-token L1/L2 plus HF
teacher-forced check. On unchanged 16 KiB geometry its lm_head stage fell
from 1.1203 to 0.7235 ms, and same-session E2E from 4.7492 to 4.4851 s.
This is still 1.34× the separate PG-off control. Five arch compile/SASS
checks passed for this source, with FP64 count zero; only sm_89 ran here.
See `page_vector_once/`. The other three endpoint cells and final solver
plans have not been regenerated with this improvement.

Supplemental fixed-geometry full-request checks now cover both models at
B=1/16: L1/L2 token mismatch count is zero across 34,816 positions and all
four HF teacher-forced C-1 cells pass. This does not complete EV-2, whose
solver-selected plans, fused/unfused check, and timed repeats remain open.
The isolated page-ring handshake probe measures about 241 ns per page cycle
for both page sizes with 512 B copied; it is a lower bound on full-page cost.
See `fixed_page_correctness/` and `page_handshake.md`.

For the current physical-page routing source, three additional B-endpoint
50-process checks are running (Llama B16, Qwen3 B1/B16). The four trace-enabled
binaries are built. `run_page_vector_diagnostics.py` is queued behind all four
fresh-process checks and will recompute the requested chain/page-full/launch-gap
metrics on this source. A subsequent fixed-geometry queue compares optimized
PG-1 with PG off in all four endpoint cells, with both arms held under
the same predeclared contamination policy. These queues do not constitute
final solver-selected EV-2 evidence.

One CPU-side search correction is now in source: PG-1's paged GEMM body fixes
its internal mainloop to two stages, so the R10 `stages` coordinate cannot
change that kernel. The serving PG-1 search domain is reduced to stage 2 and
explicit evaluations are priced at stage 2. This prevents the old latency
term from assigning an unrealized benefit to 3–16 stages, and logs the domain
reduction as `PG_STAGE_EQUIVALENCE`. It does not alter the already compiled
fixed-geometry protocol binaries. The unified tool builds and two related
host tests pass. A restricted CPU-only search priced 100 configurations
without error and reduced the six candidate domains by 74–83%; the best
Level 1 score was 3.739 ms. See `solver/stage_equivalence/`. The code contract
remains 14/18, and real plan selection remains to be measured.

The queued endpoint comparison uses each page arm's exact selected CG.
Qwen3 B1's historical PG-off binary used another candidate, so its off arm
will be rebuilt from the page arm's CG before timing. The evidence archiver
requires 200/200 fresh-process checks, all 1023 decode steps per cell, and a
realized chain at past 575; it retains raw traces and binary hashes. K-15's
code checker now validates per-round pollution decisions against the declared
power threshold instead of accepting a policy file by its keys alone.

The cold resource prewarmer also enumerated pipeline depths 3–16 after PG-1
had removed them from the search. PG-1 now requests stage-2 probes only:
at `max_m=16` this reduces the compiled wrapper set from 39 to 7. The source
fingerprint and two related host tests pass. This is an exact search-cost
reduction; the full-plan wall-time budget is still open. See F-309 and
`solver/prewarm_stage2.json`.

The current-source four-cell trace queue and its 200/200 fresh-process gate
have completed. At past 575 the realized chain lengths are 135/119/232/204;
their spans are 4.847/5.088/7.268/8.209 ms versus exact CG DRAM floors
2.537/2.826/3.573/4.583 ms. Residual bubbles are 17.1/19.0/15.9/17.8
µs per link. The full-page-and-dependency-wait intersection remains
2.711/2.702/4.972/4.705 ms per CTA per step, while median adjacent-launch
gaps are 4.096/4.096/3.072/3.072 µs. Raw evidence and all 1023 exact
per-step floor comparisons are archived in `page_vector_diagnostics/`.
The four-cell non-instrumented PG-off/PG-1 queue is now running separately.

The full-domain CPU-only Llama decode B1 PG-1 search evaluated 345
configurations in 316 s, with best Level 1 score 3.7298 ms; this excludes
top-M materialization, final compilation and GPU selection. Its old resource
prewarm compiled 39 shapes in 94.44 s, while a cold run with the new
stage-2-only filter compiled seven in 30.97 s. See F-309 and
`solver/full_stage2_llama_B1/`.

The handoff rewrite now fails closed at both CUDA lowering entry points.
`ApplyHandoffs` has not rebuilt the serving runtime stage table, so compiling
its marked graph would otherwise produce a falsely fused binary. The host
`handoff_ir` test covers both rejection paths. TF-1 and K-6/K-11 remain open;
the next implementation step is runtime stage replanning and execution of the
last-arriver/direct handoff choices, not merely removing this guard. See F-311.

The non-instrumented, matched-geometry PG-off/PG-1 endpoint comparison has
completed across Llama/Qwen3 × B=1/16. PG-1/off E2E ratios are
1.339/1.301/1.434/1.365, respectively; PG-1 remains slower in every
cell. Three timed 1024-token runs and TTFT runs per arm, round-level guard
decisions, binary hashes and SASS FP64=0 for the page arms are archived in
`page_vector_e2e/`. This is a controlled fixed-geometry diagnostic, not
solver-selected EV-2. See F-312.

The CPU-only Qwen3 B16 PG-1 search finished 355 valid evaluations in
688 s; best Level 1 score 6.7554 ms. Search alone exceeds the 600 s
per-plan budget. The largest logged phase is piece pricing/release (418 s
over 1065 calls). Raw search and timing are in
`solver/full_stage2_qwen_B16/`. A shared keyed cache across structural
attention coordinates is built; 20 Llama and seven Qwen scores match full
preparation exactly, with Llama time 85.18→51.50 s. See F-313/F-316.

Isolated, same-geometry class microbenchmarks pinpoint the paged down
projection: 1.90×/1.86× the standard collective for Llama B1/Qwen3 B16,
at roughly 320/315 GB/s versus 607/585 GB/s. Qwen3's paged L1 kernel also
spills 168 bytes per thread. Evidence is in `page_vector_classes/`; this
transport test does not replace full-request correctness or performance.
See F-314.

An isolated Qwen3 B16 `__noinline__` overlay on paged attention lowered
reported L1 spill loads from 332 to 32 bytes/thread, but was slower than the
unchanged binary in both candidate timing orders. The production TaskBody
remains unchanged; see F-315 and `noinline_attention/`. This rules out a
simple out-of-line change as the PG-1 repair.

The shared-cache full-domain Qwen3 decode B16 CPU search has now completed:
355 valid evaluations, unchanged best key/score (6.7554 ms Level 1), and
approximately 494 s versus 688 s before the cache fix. Pricing/release
fell from 418 to 266 s. Top-M, final nvcc builds and GPU selection were
excluded, so the 600 s **plan** budget is still unverified. A sampled stack
during a slow attention structure transition reached ISL cardinality from
`DeriveModelDramFloor`, but a one-case phase probe measured floor counting at
only 1.07 s versus 31.73 s in pricing/release and 12.30 s in relation
preparation. Floor counting now has a separate `dram_floor` timing phase.
See F-317 and `solver/shared_cache_full_qwen_B16/`.

The archived PG-1 trace also yields a reproducible, **effective**
historical-KV byte-rate check: Llama B1/B16 = 90.2/764.7 GB/s and Qwen3
B1/B16 = 141.2/783.6 GB/s at past 575. This counts logical historical K/V
bytes and is not a hardware DRAM counter; instrumentation perturbs timing.
The final selected-plan AT-3 throughput gate remains open. See F-318 and
`page_vector_diagnostics/attention_bandwidth.tsv`.

The native `doctor --hwcheck` command has now passed on sm_89 using one
explicit seed geometry per phase instead of a full coordinate search. Five
CUDA tests passed, prefill/decode `.so` SASS had FP64=0, and 64/64 L1/L2
tokens matched on the same instances. Both plan builds took 88.0/94.7 s;
the next fully cached run took 30.3 s including tests and generation.
The command fixes its wrapped-device lookup and quoted `-arch` parsing.
This advances AR-1's native check and validates the smoke cache; it does
not complete OPS-2's full `run` acceptance, TF-1 or EV-2. PDL remains
unexecuted on sm_89. See F-319 and `arch_primitives/hwcheck_sm89/`.

The single-page paged-GEMM loader no longer computes an impossible
per-vector cross-page branch. Isolated down-stage medians improved by 27.5%
(Llama B1) and 27.3% (Qwen3 B16), while the paged stages remain slower than
the standard collective. The changed source passed `paged_gemm` and
`page_ring` ctests. Four fixed-geometry serving libraries are being rebuilt
and will receive new 50-process checks before any full-request timing is
attributed to this revision. See F-320 and `single_page_loader/`.

Those four serving libraries are now built and their SASS audits report zero
FP64 instructions. The same generated Llama decode plan compiles for sm_80,
sm_89, sm_90, sm_100 and sm_120; only sm_89 is executable on this machine.
Four trace-enabled libraries from the changed header are built. The fresh
50-process protocol checks are still running and have not yet gated an E2E
claim. A queued follow-up checks for 200/200 passes, then measures matched
PG-off/PG-1 complete requests and collects trace chain/page/launch metrics.
The trace-report contract test passes exact-floor, residual-bubble, CTA-local
overlap and adjacent-launch calculations. The final selected-plan EV-2 and
the four failing structural checks remain open.

An additional all-coupling access-proof audit of the rank-1 B1 decode CGs
found 35/49/0 legal recompute/last-arriver/smem-direct edges for Llama and
59/85/0 for Qwen3. These are candidates, not selected or executed handoffs;
Qwen3 rank 1 is not its saved R10 winner. The existing `handoff_ir` test still
passes. See F-321 and `handoff/eligibility_summary.json`.

The current-source single-page follow-up has completed on all four batch
endpoints. All 200 new-process protocol checks passed, with distinct PIDs and
five matching token hashes per process. Under the predeclared guard, the
same-geometry off/pages 1024-token E2E medians are 3.346/4.211 s and
3.678/4.511 s for Llama B1/B16, and 5.006/7.231 s and 6.106/8.246 s for
Qwen3 B1/B16. The page arm remains 1.259/1.227/1.445/1.350× slower.
These are fixed-geometry diagnostics and cannot close EV-2 or G-9. The
requested four-cell trace is also archived: past-575 chain lengths
135/119/232/204, residual 15.8/17.4/16.3/18.4 µs per link over the exact
CG DRAM floors, page-full∩dependency-wait 3.13/3.31/6.37/6.46 ms per CTA
per step, and adjacent-launch-gap medians 4.096/3.072/4.096/4.096 µs.
Trace instrumentation perturbs timing. Raw per-process results, 1023 exact
per-step floors, binary hashes and guard decisions are in
`single_page_loader/{e2e,diagnostics}/`. See F-323.

Current workstream count: R10-C, OPS-1, SY-1, PG-0 and native AR-1 checks
are complete to their applicable scope; OPS-2, PG-1, AT-3, TF-1 and SV-18
are partial; solver-selected EV-2 has not started. In particular, the
handoff runtime and search coordinate, multi-model cached `run` acceptance,
the page-path performance repair and the final selected-plan matrix remain
open. Structural `verify.py` remains 14/18 PASS (K-6/K-11/K-15/K-16 open).

The first full OPS-2 `run` on source `96129ccae` completed Llama B=1 with
real weights and 1024 generated tokens. E2E was 3.8909 s, `E2E/ΣT_floor`
1.498, C-1 passed with 1024/1024 zero gaps, and C-2 had zero L1/L2
mismatches. The same invocation repeated with all calibration, export and
plan layers hitting cache and E2E 3.8896 s. Prefill and decode solves took
867.372 and 776.872 s, so G-7 is still failed for this sample. Evidence is
under `ops2_full_smoke/`; this is a one-cell tool smoke, not EV-2. A debugger
sample exposed repeated `DramFloor::Evaluate` parsing on identical theta
bindings. Commits `43f6b993a` and `8b4f90fb5` fix clean-build dialect output
directory creation and cache the exact bound floor respectively; prefill and
decode unit comparisons returned bitwise-identical Level 1 scores. The new
source's per-plan wall time is being remeasured separately.

The cached-floor source's separate Llama B1 build took 788.426 s for prefill
and 641.505 s for decode. Both remain over the 600 s budget, and the
artifact cache also differed from the first run, so no isolated cache
speedup is claimed. Top-3 compiled occupancy probes are now prepared on the
MLIR thread and compiled concurrently. A three-candidate decode smoke found
resident=1 for all three, with `megakernel_compile` wall time 83.923 s;
four host solver tests passed. Full plan-budget validation for this latest
code was subsequently completed on Llama B1: prefill/decode solved in
597.918/493.758 s, both within 600 s, with resource phases
99.670/86.758 s. The remaining three endpoint cells and TF-1/EV-2 still
have no complete budget or end-to-end acceptance. See F-325 and
`parallel_resource_probes/`.

The same two winners completed a real Llama B1 1024-token request at
4.0844 s E2E (`E2E/ΣT_floor=1.573`), with 0/1024 L1/L2 token mismatches,
C-1 passing at maximum HF gap 0.125, and FP64 SASS count zero for both
selected binaries. This is diagnostic only: handoff was off and no vLLM arm
ran in that session. Raw results are in `parallel_resource_probes/full_run/`.
