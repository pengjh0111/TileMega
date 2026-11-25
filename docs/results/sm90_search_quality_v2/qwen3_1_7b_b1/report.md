# H800 SQ2 search selection quality

For Qwen3-1.7B at batch 1 on H800, the estimated legal space contained 221.1 million configurations. The search scored 702 configurations, refined 24 shortlisted candidates, and compiled 6 finalists. Pooled search, replay, and finalist construction consumed 5.34 minutes. The audit measured 414 distinct programs, including 100 uniform samples. The lowest measured TPOT was 1.6094 ms. The predicted winner achieved 1.6130 ms (0.22% regret), and finalist measurement selected 1.6130 ms (0.22% regret). Across the same measured population, pooled ranking achieved Spearman rho of 0.897 and retained 52.4% of the measured top-five-percent set in the shortlist. For this workload, Eq. (5) and the native template produced the same shortlist plans. These timings use distinct processes with separate confirmation measurements.

Qwen3-1.7B, batch 1, TP1, BF16, ISL64/OSL1024. The shared checkpoint uses deterministic seed-0 weights.
Each process ran one warmup and three timed requests at N1 and N1024. TPOT is the request-wall median difference divided by 1023.
The first process screened 256 positions against HF and checked single-step/persistent token equality outside the timing interval. The guard records interference and device UUID.
The audit submitted the frozen shuffle and its reverse across the assigned H800 lanes.
Valid process receipts by GPU UUID (audit and confirmation): {'GPU-676e5df2-ee6c-11b3-6faa-e1ab1911e0c3': 264, 'GPU-ac78f16f-955d-c08b-e6fe-4e4554c8954d': 269, 'GPU-73c1a37e-d3c0-dc24-b81d-6a5563ebe677': 186, 'GPU-701b606f-5e57-e434-5e2b-d5cc774c3c9e': 139}.
After the original GPU outage, pending items were dispatched individually to available H800s, including devices 0 and 1 once authorized. The frozen candidate order supplied the queue, and guard-rejected items returned to its tail. Accepted prior results were preserved.
Timing and confirmation span physical H800 UUIDs. See gpu_failover.json for the device mapping and deviations.md for the incident and transport change.
The search budget was 600 s with four additional seeds from seed 20261124. Q uses a 1.5% band, round-robin implementation signatures, and an 8–24 bound. S is capped at six with at most two from each structural class.
P includes the frozen 30-per-decile sample of E, the full shortlist, 100 uniform draws, and the baseline coordinates. The audit oracle is the best measured in P. Deduplication requires CUDA, task/event IR, and execution-manifest equivalence.

Attempt status counts: {'valid': 858, 'guard_rejected': 199, 'infrastructure_failed': 43}.
Criteria met: {'C1_selection': True, 'C2_ranking': True, 'C3_cost': True, 'C4_local_vs_global': False}. Criteria requiring review: C4_local_vs_global.

| Method | Audit TPOT (ms) | Audit regret (%) | Confirmation TPOT (ms) | Confirmation gap (%) | Compiles | Search + compile (s) |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | 1.6094 | 0.00 | 1.6090 | 0.00 | 414 | unavailable |
| TileMega (predicted) | 1.6130 | 0.22 | 1.6257 | 1.03 | 6 | 320.12 |
| TileMega (measured) | 1.6130 | 0.22 | 1.6257 | 1.03 | 6 | 320.12 |
| B-seed | 1.6907 | 5.05 | 1.6903 | 5.05 | 1 | 82.42 |
| B-iso | 1.6130 | 0.22 | 1.6257 | 1.03 | 1 | 213.44 |
| B-pool | 1.6130 | 0.22 | 1.6257 | 1.03 | 1 | 207.39 |
| v1-q079 | 1.7968 | 11.64 | 1.8098 | 12.48 | 1 | unavailable |
| v1-q135 | 1.6924 | 5.16 | 1.6925 | 5.19 | 1 | unavailable |
| B-rand-k (expectation) | 1.6492 | 2.47 | unavailable | unavailable | 6 | unavailable |

Confirmation gaps use the independently confirmed TPOT of the audit oracle identity. A confirmation median requires all five processes. B-rand-k is an expectation from P. The measured mode chooses within S.

| Cost stage | Candidates | Time |
|---|---:|---:|
| Legal space estimate (95% interval) | 221,091,226 (220,005,199–222,174,079) | 13.85 s Monte Carlo |
| Pooled search | 702 | 138.12 s |
| Eq. (5) shortlist materialization and replay | 24 | 11.79 s |
| Finalist construction and capacity check | 6 | 170.20 s |
| Exhaustive E, extrapolated sequential compile and measurement | 702 | 37.32 h |
| Exhaustive legal space, extrapolated sequential compilation | 221,091,226 | 181029.9 days |

Finalist compile-command work sum: 437.56 s. See costs.json for additional audit costs. The extrapolations multiply the candidate counts by median per-candidate costs without parallel speedup.

The local comparison scanned 11 matched attention pairs. Reversal outside the recorded noise ranges: True.

**Fig. SQ-1.** Funnel counts and phase costs. U is the uniformly sampled legal-domain distribution. The dashed line marks the audit oracle.

**Fig. SQ-2.** Best-retained regret over compile budgets. Replay ends at the measured Q size. Random shading covers the 5th–95th percentiles of finite-population subsamples. The markers use the actual S selection and class cap.

**Fig. SQ-3.** Pooled prediction against measured TPOT on valid distinct programs in P.

**Fig. SQ-4.** Midpoint isolated attention stage cost and end-to-end TPOT for the matched attention decision. The rule and noise check are in numbers.json.

The frozen hashes, compact candidate tables, checks and plotting script accompany this report. Full receipts, binaries, generated programs and checkpoint remain in runs/.
