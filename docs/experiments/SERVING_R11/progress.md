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
| TF-1 | Access proofs and IR pass implemented; real-model norm→GEMM decisions now carry the original input/weight buffers into the paged GEMM activation loader. Stage removal/replanning, last-arriver/direct execution, and solver pricing remain open | `handoff/` |
| OPS-2 / SY-1 | End-to-end CLI and calibrated waits implemented; full cache/run acceptance remains open. SOLO+RED+BARRIER_V2 completed 200/200 fresh-process token comparisons; its performance ablation is queued | `ops/`, `calibration/`, `sync/` |
| EV-2 | Not yet run | — |

The requested PG-1 diagnostics are available for all four B endpoints in
`page_diagnostics/summary.tsv` with raw evidence archived. At past 575 the
instrumented realized chains have 135/119/232/204 links (Llama B1/B16,
Qwen3 B1/B16). Their spans are 6.383/6.667/9.985/10.928 ms against exact
CG floors of 2.537/2.826/3.573/4.583 ms; residual bubbles are
28.5/32.3/27.6/31.1 µs per link. The page-full∩dependency-wait means per CTA
per step are 3.552/3.744/7.229/7.372 ms. Median device-visible launch gaps
are 3.072/4.096/3.072/3.072 µs. Trace instrumentation perturbs the kernel;
non-instrumented performance is measured separately.

`verify.py` currently reports 13/18 structural checks PASS. K-5/K-6 require
the unfinished runtime handoffs, K-11 requires handoff search pricing,
K-15 requires EV-2 measurement policy and rounds, and K-16 requires final
serving SASS audit. These are open items, not waived checks.
