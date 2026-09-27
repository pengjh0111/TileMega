# R11 serving report

Baseline: `4bf26fb85ada63fccea101b323f3d7a09351eca2` (`origin/tilemega` at start). Prompt: `/root/Prompt/TileMega_R11_prompt.md`, SHA256 `b4b13f18596924eac17624de218735bcf8196878862dd7a6d8f00d67db71e62a`.

The user narrowed the final EV-2 acceptance to **Llama B=1 and B=16**. The original ten-cell G-9 geometric mean is therefore **unmeasured**, irrespective of the two-cell result. The measured controls and the original R10-C four-cell result remain available in [R10 controls](../SERVING_R11/r10_control/paired_final.tsv). The precise scope change and its consequences are recorded in [scope_amendment.md](scope_amendment.md).

## Implementation and evidence

The current code has the unified `tilemega` driver, cached Python end-to-end command, target-driven synchronization options, L2 prefetch, a static paged shared-memory executor with a loader warp, warp-partitioned decode attention, access-proved recompute and last-arriver handoffs, arch-dispatched async copy and PDL, and a Level 1 search with paged capacity and handoff coordinates. Source-level checks are implemented in [verify.py](verify.py). The selected-plan end-to-end run and its archived measurement decisions will be recorded below after completion.

R10-C current-source controls (Llama/Qwen3, B=1/16) produced TileMega-to-vLLM throughput ratios **0.957/1.077/0.940/1.048**, geometric mean **1.0038**. All four C-1/C-2 controls passed; all eight plan SASS audits had zero FP64 instructions. The b22 queue was stopped as instructed, and Qwen3's unpruned G-6 control timed out at 5,400 s, so the original ten-cell R10 G-9 and Qwen3 pruning-equivalence gate were not passed. See [paired_final.tsv](r10_control/paired_final.tsv), [plans_summary.tsv](r10_control/plans_summary.tsv), and [R10 closure](../SERVING_R10/summary.md).

The separately instrumented, fixed-geometry PG-1 Llama decode run measured at past 575:

| B | Realized chain links | Chain span / CG DRAM floor | Residual excess per link | Page ring full while waiting on dependencies, mean CTA time/step | Adjacent launch gap, mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 135 | 4.665 / 2.537 ms | 15.76 µs | 3.126 ms | 3.627 µs |
| 16 | 119 | 4.902 / 2.826 ms | 17.45 µs | 3.306 ms | 3.567 µs |

These are trace-build measurements, not selected-plan E2E latency. The full per-step floor and raw traces are in [diagnostics](single_page_loader/diagnostics/summary.tsv). In the four fixed-geometry controls, the PG-1 page arm was slower than PG-off by 1.259/1.227/1.445/1.350× (Llama B1/B16, Qwen3 B1/B16); 200/200 fresh-process page-protocol checks passed. See [fixed-geometry controls](single_page_loader/e2e/summary.tsv). This is a concrete performance failure, not a claim of PG speedup.

The target-driven SOLO+RED+BARRIER_V2 combination passed 200/200 fresh-process token comparisons over Llama/Qwen3 B=1/16. Its fixed-geometry Llama B1 and Qwen3 B16 E2E changes were only 6.3364→6.3170 s and 10.9173→10.9083 s, respectively. The generated target wait policy and hop calibration are in [calibration](calibration); the protocol evidence is in [sync](sync/combo_fresh50_summary.json). These results do not justify attributing a material end-to-end gain to EX-E3.

The arch-dispatched primitive probe compiled for sm_80/89/90/100/120. The sm_90/100/120 SASS contains `UTMALDG`, `UBLKCP`, `SYNCS.*` and `ACQBULK`/`PREEXIT`; sm_80/89 emits `LDGSTS` without those instructions. Those other architecture paths were **not executed** on the sm_89 machine. Host tensor-map shape checks passed, but the driver returned `CUDA_ERROR_NOT_SUPPORTED` for 12/12 encodes here, so hardware encoding remains for the target device. See [arch report](arch_primitives/report.json), [tensor-map check](arch_primitives/tensor_map.json), and the runnable `python -m tilemega doctor --hwcheck` command.

OPS-1 reduced default executable targets from 122 to 27; a clean Release build on this host took 142.21 s before and 140.94 s after. New and old compile commands emitted byte-identical `.cu` for one serving and one legacy input. The recorded pre-change test-name set and R11 ctest run contain the same 90 cases, all passing at that checkpoint. See [tool counts](ops/executable_counts.json), [build times](ops/build_times.json), [byte comparisons](ops/byte_identity.json), and [ctest](ops/ctest_full_90.log). The OPS-2 cache has independently shown a full Llama B1 run and a repeat with calibration/export/plan hits, but this is not a new-machine first-run proof. See [OPS-2 smoke](ops2_full_smoke/).

## Incomplete implementation and declared deviations

- **TF-1 `smem_direct`:** The specification requires a full `DirectHandoff<P,C>` CG-to-runtime path. Current code proves access and worker adjacency, rewrites the fused IR, provides a page helper, and passes a synthetic positive runtime-projection test. A reference-graph 1:1 edge with synthetic adjacent placement now passes `VerifyHandoffAccess` and `ApplyHandoffs` (`HANDOFF_DIRECT two_stage_access=PASS ir_rewrite=PASS`), while the separate 160-thread page helper test is queued after EV-2. The lowering still does not emit a complete executable serving `smem_direct` handoff. There are zero eligible edges in the two real decode CGs, so the permitted fallback remains incomplete at codegen. The next step is to make this synthetic two-stage CG execute through generated serving code.
- **SV-18(d):** The specification requires a top-3 per-class measured-rate check and top-M rerank for deviations over 20%. The current class microbenchmarks compare paged against the old collective, but they do not estimate `measured/model` for every chosen class and do not rerank top-M. The down class shows a 1.90×/1.86× paged/old slowdown (Llama/Qwen3); integrate this rate observation into the candidate model and rerun ranking.
- **SV-18(g):** The specification allows equal neighboring B winners to share one interval plan. The current runtime binds one B per plan and `StructureInvariant` checks literal stage counts, so no cross-B interval plan is emitted. Implement a B-parameterized host queue and prove both endpoint windows before merging.
- **TF-2 scoring:** The specification prices each access-proved edge handoff in Level 1 and reruns top-M fluid simulation with those choices. The current handoff mask price is a stage-class approximation and top-M simulation remains on the event plan; final compiler timing tests an access-proved handoff binary and retains it only for a measured gain of at least 2%. A restricted Llama B1 case priced a handoff but selected no legal runtime edge, demonstrating this mismatch. See [handoff_closure](handoff_closure/README.md).
- **EV-2 scope:** The original 20-plan, ten-cell matrix is reduced by the user's explicit instruction to four plans and two Llama cells. All ten-cell gates remain unmeasured, not passed.

## Pending final evidence

The final Llama B=1/16 solve, 1024-token runs, paired vLLM measurements, HF checks, SASS archive, full `verify.py` output, gate table, plan budgets, ablations, and selected-plan attribution are being collected. This section must be replaced with measured values before final submission.
