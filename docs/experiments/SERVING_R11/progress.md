# R11 progress ledger (2026-09-26)

Baseline: `4bf26fb85ada63fccea101b323f3d7a09351eca2`. Prompt SHA256:
`b4b13f18596924eac17624de218735bcf8196878862dd7a6d8f00d67db71e62a`.

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
| AR-1 | Five target-architecture compile checks and SASS branches recorded; native sm_89 only executed | `arch_primitives/` |
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

Supplemental fixed-geometry full-request checks now cover both models at
B=1/16: L1/L2 token mismatch count is zero across 34,816 positions and all
four HF teacher-forced C-1 cells pass. This does not complete EV-2, whose
solver-selected plans, fused/unfused check, and timed repeats remain open.
The isolated page-ring handshake probe measures about 241 ns per page cycle
for both page sizes with 512 B copied; it is a lower bound on full-page cost.
See `fixed_page_correctness/` and `page_handshake.md`.
