# TileMega R5 — critical-path costs and joint configuration selection

1. **Provenance and ordering.** Baseline `a001b0ac53197551fe1e5e115528a9ea40d24783`; branch `tilemega`. External prompt `/root/Prompt/TileMega_R5_prompt.md`, SHA256 `7773c35f1f45e920048cf2d86f36e84ba06472efa1ffe1f22c6205a2c982e586`. The frozen FORK5 commit `495e7c10` precedes first search implementation `60f6e4bc` (H4). The final source HEAD is stamped in [the identity manifest](../PHASE/sass_identity/manifest.json); its direct child contains only identity artifacts. This report cannot contain its own future commit hash. The complete final list is `git log --reverse --format="%h %s" a001b0ac53197551fe1e5e115528a9ea40d24783..HEAD`. Commits existing when this report was rendered:

```text
b02e5a90 trace: audit the critical path node durations
ecf2f03c solver: rank plans before simulating them
6fbed6e1 trace: time the phases inside a task body
495e7c10 experiments: split the critical path task cost
5a626aef solver: price publication rather than polling
60f6e4bc solver: search configurations against the L2 objective
0eac0cbe experiments: validate task phase instrumentation
1b83eb6b experiments: measure the searched reference configurations
311b3201 solver: bound memory while materializing search graphs
f72777b6 solver: reuse a graph only after exact relation matching
ed33516e experiments: measure real width and selected task phases
be87310a experiments: add the sm_120 runners for round five
e2fd4a6f docs: record the round five configuration results
b1ad2fac experiments: serialize every verifier evidence path
7b0bc245 experiments: isolate target search failures by cell
```

2. **Gate results.** FAIL is retained; no threshold or coverage is relaxed. A report gate marked PASS means the required measurement/report is present, not that every candidate improved. Correctness coverage is four selected reference cells × 50, two selected real-width cells × 50, four phase-instrumented reference cells × 50, and twelve re-solved SEQSCAN cells × 50 (600/600).

| Gate | Result | Hard | Raw evidence |
|---|---|---|---|
| D3-a/H2 | PASS | yes | docs/experiments/PHASE/sass_identity |
| D3-b | PASS | yes | docs/experiments/PHASE/raw/correctness |
| D3-c | PASS | yes | docs/experiments/PHASE/raw/measure |
| D3-d | PASS | yes | PHASE/raw/trace + JOINT/raw/*/{phase,operand_probe} |
| D3-e/H4 | PASS | yes | docs/experiments/PHASE/raw/trace |
| S1c-a | FAIL | yes | docs/experiments/SIMULATOR/r5/evaluations.tsv |
| S1c-b | FAIL | yes | docs/experiments/SIMULATOR/r5/evaluations.tsv |
| S1c-c | PASS | no | docs/experiments/SIMULATOR/r5/replay.tsv |
| S3-a | PASS | yes | docs/experiments/JOINT/raw |
| S3-b | PASS | no | docs/experiments/JOINT/raw/*/measure |
| S1c-d | PASS | no | docs/experiments/JOINT/raw/*/measure |
| S3-c | FAIL | yes | docs/experiments/JOINT/raw/*/measure |
| S3-d/f | PASS | no | docs/experiments/JOINT/raw/*/trace |
| S3-e | PASS | no | docs/experiments/JOINT/raw/real* |
| E4-conditional | PASS | no | docs/experiments/PHASE/raw/trace |

3. **Complete raw-data verifier output.** `python3 docs/experiments/JOINT/verify.py` runs all gates before returning nonzero for hard failures.

```text
D3-a/H2 PASS models=2/2 bytes_identical final_source_stamp_valid evidence=/root/TileMega/docs/experiments/PHASE/sass_identity
D3-b PASS gqa2:s4=50/50 gqa2:s128=50/50 mha4:s4=50/50 mha4:s128=50/50 evidence=/root/TileMega/docs/experiments/PHASE/raw/correctness
D3-c PASS gqa2:s4:p0=1.043578[1.041379,1.045690] gqa2:s4:p5=1.013986[1.011042,1.016719] gqa2:s128:p0=1.023541[1.023256,1.024834] gqa2:s128:p5=1.026545[1.024775,1.026957] mha4:s4:p0=1.045860[1.044360,1.048031] mha4:s4:p5=1.030627[1.028377,1.032572] mha4:s128:p0=1.025911[1.025067,1.026882] mha4:s128:p5=1.021204[1.020359,1.021609] real:s4:p0=1.004464[1.001075,1.008344] real:s4:p5=1.006492[1.005754,1.007293] real:s128:p0=0.984078[0.959777,1.010655] real:s128:p5=1.006492[1.005771,1.007652] evidence=/root/TileMega/docs/experiments/PHASE/raw/measure
D3-d PASS baseline_nodes=135568 supplemental_nodes=113544 max_relative_closure_error=0.000000000 evidence=PHASE/raw/trace + JOINT/raw/*/{phase,operand_probe}
D3-e/H4 PASS FORK5 rule=3 load_share=0.006 fixed_share=0.232 math_share=0.763 cells=4 evidence=/root/TileMega/docs/experiments/PHASE/raw/trace
S1c-a FAIL reference_full_us=140259.531 real_full_us=138348.284 evidence=/root/TileMega/docs/experiments/SIMULATOR/r5/evaluations.tsv
S1c-b FAIL points=18 coarse_rho=0.561240985 full_rho=0.880288958 actual_top1_coarse_rank=1 evidence=/root/TileMega/docs/experiments/SIMULATOR/r5/evaluations.tsv
S1c-c PASS dumps=68 abs_relative_p50=0.045399639 p90=0.108475523 max=0.138277099 evidence=/root/TileMega/docs/experiments/SIMULATOR/r5/replay.tsv
S3-a PASS gqa2:s4=50/50 gqa2:s128=50/50 mha4:s4=50/50 mha4:s128=50/50 SEQSCAN=PASS evidence=/root/TileMega/docs/experiments/JOINT/raw
S3-b PASS achieved=4/4 gqa2:s4:top3=0.409039[0.405965,0.413588] gqa2:s128:top2=0.549580[0.548132,0.550373] mha4:s4:top3=0.460194[0.458923,0.462054] mha4:s128:top3=0.553799[0.550798,0.559244] evidence=/root/TileMega/docs/experiments/JOINT/raw/*/measure
S1c-d PASS gqa2:s4:predicted_top1_measured_rank=3/3 gqa2:s128:predicted_top1_measured_rank=2/3 mha4:s4:predicted_top1_measured_rank=3/3 mha4:s128:predicted_top1_measured_rank=3/3 real:s4:predicted_top1_measured_rank=3/3 real:s128:predicted_top1_measured_rank=1/3 evidence=/root/TileMega/docs/experiments/JOINT/raw/*/measure
S3-c FAIL gqa2:s4:predicted_top1_measured_rank=3/3 gqa2:s128:predicted_top1_measured_rank=2/3 mha4:s4:predicted_top1_measured_rank=3/3 mha4:s128:predicted_top1_measured_rank=3/3 real:s4:predicted_top1_measured_rank=3/3 real:s128:predicted_top1_measured_rank=1/3 catalog_top_3_percent=unresolved unmeasured_catalog_candidates_have_no_empirical_rank evidence=/root/TileMega/docs/experiments/JOINT/raw/*/measure
S3-d/f PASS gqa2:s4:control:cp_ns=228352,nodes=20,mean_ns=11417.600,queue_cp=0.170404,l2_l1=0.724194,l2_floor=1.237668 gqa2:s4:top3:cp_ns=89088,nodes=20,mean_ns=4454.400,queue_cp=0.942529,l2_l1=0.783217,l2_floor=1.295259 gqa2:s128:control:cp_ns=320512,nodes=20,mean_ns=16025.600,queue_cp=0.447284,l2_l1=0.800399,l2_floor=1.284345 gqa2:s128:top2:cp_ns=190464,nodes=19,mean_ns=10024.421,queue_cp=0.887097,l2_l1=0.761102,l2_floor=1.188172 mha4:s4:control:cp_ns=452608,nodes=40,mean_ns=11315.200,queue_cp=0.171946,l2_l1=0.727305,l2_floor=1.160351 mha4:s4:top3:cp_ns=185344,nodes=40,mean_ns=4633.600,queue_cp=0.906077,l2_l1=0.765472,l2_floor=1.303867 mha4:s128:control:cp_ns=645120,nodes=40,mean_ns=16128.000,queue_cp=0.426984,l2_l1=0.818002,l2_floor=1.297222 mha4:s128:top3:cp_ns=393216,nodes=40,mean_ns=9830.400,queue_cp=0.927083,l2_l1=0.699225,l2_floor=1.181641 real:s4:control:cp_ns=4661248,nodes=40,mean_ns=116531.200,queue_cp=0.367311,l2_l1=0.814461,l2_floor=1.013366 real:s4:top1:cp_ns=2136064,nodes=36,mean_ns=59335.111,queue_cp=1.758869,l2_l1=0.917369,l2_floor=1.098392 real:s128:control:cp_ns=4775936,nodes=37,mean_ns=129079.351,queue_cp=0.709262,l2_l1=0.910670,l2_floor=1.408240 real:s128:top1:cp_ns=4687872,nodes=40,mean_ns=117196.800,queue_cp=1.417431,l2_l1=0.949472,l2_floor=1.049468 evidence=/root/TileMega/docs/experiments/JOINT/raw/*/trace
S3-e PASS real:s4=0.873807[0.872940,0.874120] correctness=50/50 real:s128=1.009736[0.997030,1.039466] correctness=50/50 evidence=/root/TileMega/docs/experiments/JOINT/raw/real*
E4-conditional PASS not_applicable FORK5 rule=3 load_share=0.006 fixed_share=0.232 math_share=0.763 cells=4 EX-E4 excluded by frozen rule evidence=/root/TileMega/docs/experiments/PHASE/raw/trace
VERIFY5 gates=15 hard_failed=3
```

4. **Stop ledger.** No global stop condition occurred. Existing configurations remain correct; the motivating floor and mean-node magnitudes reproduce.

| Item | Affected scope | Cause and disposition | Unlock / estimated work |
|---|---|---|---|
| S1c-a/b | Full-catalog inner evaluation | Full simulation misses budget; cheap bounds miss rho. S3 continues with explicit capacity deferrals. | Cache/incrementally update readiness and grouping; benchmark event-heap and dense-edge costs, approximately 1–3 engineering days plus the unchanged 18-point rerun. |
| S3-c | Claim of empirical top-3% across the enumerated catalog | Only the reduced shortlist is compiled. Unmeasured candidates have no actual rank. | Validate arithmetic/resource admissibility, then measure the frozen domain: thousands of configurations per cell before six placements; a multi-day build/measurement campaign, not a retrospective gate change. |
| Rejected new real-width configurations | Only the failed arithmetic shapes | split8 s4 and split2 s128 miss one CPU-golden element while L0.5/L1/L2 agree. Fresh admissible shortlists replace them; failed logs remain. | Current selected configurations complete their own checks. Per-GEMM split admission and first-divergence localization are the next approximately 1–2 day extension. |
| Long-sequence CPU preparation | MHA SEQSCAN preparation only | Coordinate-pair materialization exhausted memory. Streamed exact relations and graph-only projection preserve checks; failed attempts remain. | Completed/retried evidence is in seqscan_plans and self_check. No GPU correctness failure is relabeled. |

5. **Explicit degradations.** The outer axis catalog is enumerated, but only three priority geometries plus L1/R4 seeds (and a logged numerical-admission expansion) are fully projected. The arithmetic lower bound is weak and unknown CP bound is zero; capacity deferrals are not dominance proofs. Top-k=3 includes ties. The 70%-tile-area/30%-operand-size phase extrapolator omits K-loop fixed cost and is not an oracle. Historical W>1 replay uses observed execution order as a FIFO approximation. sm_120 scripts are self-checked/compiled on 4090 only; target hardware results are pending as required. No prefetch implementation is claimed.

6. **Frozen diagnosis and phases.**

```text
FORK5 rule=3 load_share=0.006 fixed_share=0.232 math_share=0.763 cells=4
```

Each entry below is microseconds followed by fraction of summed node time. CP and all-node totals are different populations; neither is a sum of wall-clock kernel phases. Epilogue includes the separately exported executor tail. SIMT has no isolated prologue, so its load_wait is structurally zero. Event-hop columns from phase-only dumps are explicitly unavailable; corrected DAG/node phases remain usable.

**Initial fresh premise audit (configuration A).** This precedes instrumentation and uses digest-checked R4 binaries in new processes. The later confirmation control uses R3 B; task-weight floors can also vary with contention and timer quantization, so the initial 1.18–1.38 interval is not an invariant across cohorts.

| Cell | Placement | CP nodes | CP mean / p50 / p90 / max us | All-task mean / p50 / p90 / max us | Measured/floor |
|---|---|---|---|---|---|
| gqa2 s4 | 0 | 20 | 12.134 / 4.096 / 23.552 / 43.008 | 8.863 / 3.072 / 21.504 / 43.008 | 1.229050 |
| gqa2 s4 | 5 | 20 | 12.186 / 4.096 / 23.552 / 43.008 | 9.114 / 3.072 / 21.504 / 43.008 | 1.205882 |
| gqa2 s128 | 0 | 20 | 17.101 / 24.576 / 33.792 / 53.248 | 6.301 / 1.024 / 21.504 / 53.248 | 1.175000 |
| gqa2 s128 | 5 | 20 | 17.510 / 24.576 / 34.816 / 53.248 | 6.257 / 1.024 / 20.480 / 53.248 | 1.327211 |
| mha4 s4 | 0 | 40 | 12.186 / 4.096 / 23.552 / 41.984 | 7.772 / 1.024 / 21.504 / 41.984 | 1.218534 |
| mha4 s4 | 5 | 40 | 12.288 / 4.096 / 23.552 / 43.008 | 7.966 / 2.048 / 21.504 / 43.008 | 1.183333 |
| mha4 s128 | 0 | 40 | 17.178 / 24.576 / 33.792 / 53.248 | 4.859 / 1.024 / 20.480 / 53.248 | 1.185782 |
| mha4 s128 | 5 | 40 | 17.382 / 24.576 / 34.816 / 54.272 | 4.847 / 1.024 / 20.480 / 54.272 | 1.375552 |

**Original configuration, cp nodes.**

| Cell | Placement | setup us/share | load_wait us/share | mainloop us/share | epilogue us/share |
|---|---|---|---|---|---|
| gqa2 s4 | 0 | 14.336 / 0.0617 | 4.096 / 0.0176 | 186.368 / 0.8018 | 27.648 / 0.1189 |
| gqa2 s4 | 5 | 16.384 / 0.0717 | 0.000 / 0.0000 | 186.368 / 0.8161 | 25.600 / 0.1121 |
| gqa2 s128 | 0 | 16.384 / 0.0513 | 4.096 / 0.0128 | 230.400 / 0.7212 | 68.608 / 0.2147 |
| gqa2 s128 | 5 | 19.456 / 0.0605 | 5.120 / 0.0159 | 226.304 / 0.7038 | 70.656 / 0.2197 |
| mha4 s4 | 0 | 32.768 / 0.0702 | 7.168 / 0.0154 | 373.760 / 0.8004 | 53.248 / 0.1140 |
| mha4 s4 | 5 | 32.768 / 0.0717 | 1.024 / 0.0022 | 374.784 / 0.8206 | 48.128 / 0.1054 |
| mha4 s128 | 0 | 38.912 / 0.0609 | 13.312 / 0.0208 | 447.488 / 0.7003 | 139.264 / 0.2179 |
| mha4 s128 | 5 | 38.912 / 0.0614 | 6.144 / 0.0097 | 450.560 / 0.7108 | 138.240 / 0.2181 |
| real s4 | 0 | 37.888 / 0.0095 | 22.528 / 0.0056 | 3833.856 / 0.9566 | 113.664 / 0.0284 |
| real s4 | 5 | 40.960 / 0.0089 | 32.768 / 0.0071 | 4433.920 / 0.9601 | 110.592 / 0.0239 |
| real s128 | 0 | 43.008 / 0.0102 | 20.480 / 0.0048 | 3943.424 / 0.9324 | 222.208 / 0.0525 |
| real s128 | 5 | 43.008 / 0.0037 | 34.816 / 0.0030 | 11437.056 / 0.9721 | 249.856 / 0.0212 |

**Original configuration, all nodes.**

| Cell | Placement | setup us/share | load_wait us/share | mainloop us/share | epilogue us/share |
|---|---|---|---|---|---|
| gqa2 s4 | 0 | 149.504 / 0.0890 | 18.432 / 0.0110 | 1364.992 / 0.8128 | 146.432 / 0.0872 |
| gqa2 s4 | 5 | 162.816 / 0.0971 | 0.000 / 0.0000 | 1383.424 / 0.8248 | 131.072 / 0.0781 |
| gqa2 s128 | 0 | 2932.736 / 0.1138 | 37.888 / 0.0015 | 21779.456 / 0.8450 | 1025.024 / 0.0398 |
| gqa2 s128 | 5 | 3110.912 / 0.1203 | 27.648 / 0.0011 | 21618.688 / 0.8358 | 1110.016 / 0.0429 |
| mha4 s4 | 0 | 354.304 / 0.0940 | 40.960 / 0.0109 | 3069.952 / 0.8147 | 303.104 / 0.0804 |
| mha4 s4 | 5 | 387.072 / 0.1029 | 1.024 / 0.0003 | 3089.408 / 0.8214 | 283.648 / 0.0754 |
| mha4 s128 | 0 | 6822.912 / 0.1257 | 72.704 / 0.0013 | 45049.856 / 0.8303 | 2313.216 / 0.0426 |
| mha4 s128 | 5 | 6758.400 / 0.1244 | 47.104 / 0.0009 | 44990.464 / 0.8283 | 2520.064 / 0.0464 |
| real s4 | 0 | 2576.384 / 0.0107 | 1649.664 / 0.0068 | 234561.536 / 0.9704 | 2928.640 / 0.0121 |
| real s4 | 5 | 2867.200 / 0.0084 | 3418.112 / 0.0100 | 331249.664 / 0.9722 | 3170.304 / 0.0093 |
| real s128 | 0 | 31901.696 / 0.0491 | 1628.160 / 0.0025 | 593366.016 / 0.9139 | 22382.592 / 0.0345 |
| real s128 | 5 | 32376.832 / 0.0189 | 3678.208 / 0.0021 | 1655405.568 / 0.9650 | 23993.344 / 0.0140 |

**Selected configurations, CP nodes (supplemental snapshots; do not rewrite FORK5).**

| Cell | Config | setup share | load share | mainloop share | epilogue share |
|---|---|---|---|---|---|
| gqa2 s4 | 32x16x64s2k1_k1_r1 | 0.1625 | 0.0625 | 0.6750 | 0.1000 |
| gqa2 s128 | 32x16x32s2k1_k1_r5 | 0.1005 | 0.0317 | 0.7937 | 0.0741 |
| mha4 s4 | 32x16x64s2k1_k1_r1 | 0.1637 | 0.0643 | 0.6433 | 0.1287 |
| mha4 s128 | 32x16x16s2k1_k2_r5 | 0.0945 | 0.0131 | 0.8005 | 0.0919 |
| real s4 | 32x16x16s2k1_k1_r2 | 0.0157 | 0.0048 | 0.9434 | 0.0362 |
| real s128 | 64x128x16s2k1_k1_r2 | 0.0088 | 0.0046 | 0.9512 | 0.0353 |

Operand shape/split/bytes associations and quantized load distributions are in [operand_load.tsv](operand_load.tsv), [selected_operand_load.tsv](selected_operand_load.tsv), and [operand_split_probe.tsv](operand_split_probe.tsv), the admitted same-geometry split1/16 diagnostic. At 32x128x64 and the same 20480 first-tile bytes, GEMM mean load_wait is 3334.095 ns / 8262.421 cycles (split1) versus 2951.381 ns / 7369.500 cycles (split16). CP load share rises 1.040% to 7.741% because splitting changes the graph and task duration; this is a diagnostic snapshot, not a controlled end-to-end speedup claim. Baseline GEMM uses 8192 first-tile operand bytes; selected geometries change both footprint and occupancy. These associations do not identify a causal bytes-to-latency slope. clock64 boundaries provide the finer within-CTA check. The original real-s128 rotate phase snapshot is slow relative to the repeated timing cohort; extrapolating from a single snapshot remains a model limitation.

7. **Configuration, path and width answers.** Fresh ChainDP's uniform L1 seed is 32x16x16s2 split1. Its per-operator result uses three shapes and split4/8 in some GEMMs; see PHASE/chain_dp/plan_*.tsv. The old measured R4 configuration is 128x128x16s3 split1. All selected split choices are reported below; kappa/residency are encoded as `_kN_rN`. W remains 1 and ChainDP is unchanged.

| Cell | Arm / config / placement | CP us | Nodes | Mean / p50 / p90 / max us | queue_lb / CP | Body busy fraction |
|---|---|---|---|---|---|---|
| gqa2 s4 | control / 128x128x16s3k1_k1_natural / rotate | 228.352 | 20 | 11.418 / 4.096 / 21.504 / 38.912 | 0.1704 | 0.0320 |
| gqa2 s4 | top3 / 32x16x64s2k1_k1_r1 / eft | 89.088 | 20 | 4.454 / 3.584 / 7.168 / 11.264 | 0.9425 | 0.2381 |
| gqa2 s128 | control / 128x128x16s3k1_k1_natural / rotate | 320.512 | 20 | 16.026 / 13.312 / 31.744 / 48.128 | 0.4473 | 0.2448 |
| gqa2 s128 | top2 / 32x16x32s2k1_k1_r5 / eft | 190.464 | 19 | 10.024 / 9.216 / 17.408 / 32.768 | 0.8871 | 0.3648 |
| mha4 s4 | control / 128x128x16s3k1_k1_natural / rotate | 452.608 | 40 | 11.315 / 3.072 / 21.504 / 38.912 | 0.1719 | 0.0279 |
| mha4 s4 | top3 / 32x16x64s2k1_k1_r1 / eft | 185.344 | 40 | 4.634 / 3.584 / 7.168 / 11.264 | 0.9061 | 0.2641 |
| mha4 s128 | control / 128x128x16s3k1_k1_natural / rotate | 645.120 | 40 | 16.128 / 12.800 / 31.744 / 49.152 | 0.4270 | 0.2467 |
| mha4 s128 | top3 / 32x16x16s2k1_k2_r5 / eft | 393.216 | 40 | 9.830 / 7.168 / 21.504 / 32.768 | 0.9271 | 0.3807 |
| real s4 | control / 128x128x16s3k1_k1_natural / rotate | 4661.248 | 40 | 116.531 / 22.528 / 282.624 / 519.168 | 0.3673 | 0.2825 |
| real s4 | top1 / 32x16x16s2k1_k1_r2 / eft | 2136.064 | 36 | 59.335 / 50.688 / 238.592 / 240.640 | 1.7589 | 0.8120 |
| real s128 | control / 128x128x16s3k1_k1_natural / rotate | 4775.936 | 37 | 129.079 / 32.768 / 309.248 / 494.592 | 0.7093 | 0.4435 |
| real s128 | top1 / 64x128x16s2k1_k1_r2 / wavefront | 4687.872 | 40 | 117.197 / 37.888 / 240.640 / 510.976 | 1.4174 | 0.5460 |

Real-s4 CP falls 4661.248 to 2136.064 us but queue_lb becomes 3757.056 us; real-s128 CP falls only 4775.936 to 4687.872 us while queue_lb rises to 6644.736 us. Both selected real-width floors are queue-bound. The reference CP shrinks mostly through cheaper nodes; counts stay 20/40 except one 19-node path. queue_lb/CP moves toward one, and body-busy fractions increase. These are utilization proxies, not tensor-core saturation measurements. At seq4, M=128 pads 4 active rows into 128; M=32 reduces that waste, while smaller N exposes more independent tiles. K64 reduces mainloop iterations in the short-sequence winner. Compared with the same-cell L1-geometry top1 arm, selected ratios are 0.829630/0.948498/0.847684/0.944559 (all 95% CIs below one); the last changes kappa. Not all gain against the stale R4 geometry is attributable solely to changing the optimization objective. No measured need for per-stage kappa was established, so the global field is retained.

8. **End-to-end position.** All controls are fresh in the same rotated cohort. `own L1` uses the candidate binary; `control L1` makes geometry improvements visible separately. Floors come from each candidate's fresh traced task weights, not a universal hardware lower bound.

| Cell | Selected L2 ms | Selected/control [95% CI] | L2 / own L1 | L2 / control L1 | L2 / own floor |
|---|---|---|---|---|---|
| gqa2 s4 | 0.115392 | 0.409039 [0.405965,0.413588] | 0.783217 | 0.294906 | 1.295259 |
| gqa2 s128 | 0.226304 | 0.549580 [0.548132,0.550373] | 0.761102 | 0.439364 | 1.188172 |
| mha4 s4 | 0.241664 | 0.460194 [0.458923,0.462054] | 0.765472 | 0.334613 | 1.303867 |
| mha4 s128 | 0.464640 | 0.553799 [0.550798,0.559244] | 0.699225 | 0.452754 | 1.181641 |
| real s4 | 4.126720 | 0.873807 [0.872940,0.874120] | 0.917369 | 0.712022 | 1.098392 |
| real s128 | 6.973440 | 1.009736 [0.997030,1.039466] | 0.949472 | 0.920285 | 1.049468 |

9. **Simulator budget and prediction.** Full evaluation fails 1/10 ms budgets (140.260/138.348 ms maxima), whereas coarse evaluation costs at most 0.151/0.619 ms. Coarse/full Spearman on the historical 18 points is 0.561241/0.880289; actual top1 is coarse rank1. `SIMULATOR/r5/ranks.tsv` gives every k=1..5 timing/ranking tradeoff. Publication/consumer-wait fits are 1074.1284/1764.3682 ns; primitive hop is 1235.4118 ns. These are inferred marginal prices with actual publisher masks, not independent instruction latencies. Historical replay errors are p50 4.540%, p90 10.848%, max13.828%; changed-geometry prediction is a different test:

| Cell | Arm | Config / placement | Prediction ms | Measured ms | Measured rank / 3 |
|---|---|---|---|---|---|
| gqa2 s4 | top1 | 32x16x16s2k1_k1_r1 / eft | 0.134848 | 0.139264 | 3 |
| gqa2 s4 | top2 | 32x16x32s2k1_k1_r2 / eft | 0.134848 | 0.118784 | 2 |
| gqa2 s4 | top3 | 32x16x64s2k1_k1_r1 / eft | 0.134848 | 0.115392 | 1 |
| gqa2 s128 | top1 | 32x16x16s2k1_k1_r5 / eft | 0.216607 | 0.238368 | 2 |
| gqa2 s128 | top2 | 32x16x32s2k1_k1_r5 / eft | 0.216607 | 0.226304 | 1 |
| gqa2 s128 | top3 | 32x16x16s2k1_k1_r5 / rotate | 0.219272 | 0.301056 | 3 |
| mha4 s4 | top1 | 32x16x16s2k1_k1_r1 / eft | 0.273623 | 0.285568 | 3 |
| mha4 s4 | top2 | 32x16x32s2k1_k1_r2 / eft | 0.273623 | 0.248832 | 2 |
| mha4 s4 | top3 | 32x16x64s2k1_k1_r1 / eft | 0.273623 | 0.241664 | 1 |
| mha4 s128 | top1 | 32x16x16s2k1_k1_r5 / eft | 0.449429 | 0.492544 | 3 |
| mha4 s128 | top2 | 32x16x32s2k1_k1_r5 / eft | 0.449429 | 0.469856 | 2 |
| mha4 s128 | top3 | 32x16x16s2k1_k2_r5 / eft | 0.454967 | 0.464640 | 1 |
| real s4 | top1 | 32x16x16s2k1_k1_r2 / eft | 1.407002 | 4.126720 | 3 |
| real s4 | top2 | 32x16x16s2k1_k2_r2 / eft | 1.408889 | 4.073472 | 2 |
| real s4 | top3 | 32x16x16s2k1_k4_r2 / eft | 1.408889 | 4.051808 | 1 |
| real s128 | top1 | 64x128x16s2k1_k1_r2 / wavefront | 11.063491 | 6.973440 | 1 |
| real s128 | top2 | 64x128x16s2k1_k1_r2 / rotate | 11.065267 | 6.979584 | 2 |
| real s128 | top3 | 64x128x16s2k1_k2_r2 / wavefront | 11.283513 | 6.980320 | 3 |

**k tradeoff.** Times below are whole six-placement batch microseconds, not the single-Plan gate. The hybrid rho uses simulated scores where available and retained lower-bound scores otherwise; this mixed score is not a uniformly calibrated estimator. k=3 was fixed to match the three compiled finalists, with boundary ties. k=2 crosses 0.85 in this hybrid replay but still misses the time budget; changing k alone does not solve S1c.

| k | Hybrid rho / 18 | Actual top1 rank | Reference batch us | Real batch us | Simulated calibration points |
|---|---|---|---|---|---|
| 1 | 0.6388028895768834 | 1 | 130423.51 | 135914.176 | 9 |
| 2 | 0.8513931888544891 | 1 | 264432.114 | 264140.262 | 10 |
| 3 | 0.8348813209494325 | 1 | 535620.932 | 403279.352 | 12 |
| 4 | 0.8802889576883385 | 1 | 530402.513 | 674077.949 | 17 |
| 5 | 0.8802889576883385 | 1 | 646983.419 | 683941.622 | 18 |

Changed-geometry top-3 absolute relative prediction error: p50 13.3743%, maximum 65.9051%. These are fresh measurements, unlike the historical replay.

10. **Prompt corrections and deviations.** The actual R4 tile and current ChainDP output replace the stale F-128 example; the mean ~12 us and floor premise themselves reproduce. Historical 0.7129 ms belongs to legacy placement, so the control is explicitly fresh rotate + R3 B. Reference rotate cells determine the frozen fork; legacy/real-width are separately reported. Capacity reduction, numerical exclusions, FIFO replay approximation and unverified catalog percentile are explicit degradations, not altered gates. Graph-only projection, streaming/RLE storage and exact relation caching repair preparation cost without changing Plan semantics. RLE is lossless raw evidence: decode each task_dag.json's runs with dag_codec, then check original SHA256. No historical experiment file other than the authorized analyzer is rewritten; user dirty R4 files are preserved.

11. **Excluded scope.** No new synchronization protocol, acquire-fence pricing, window batching or chain-cost refinement; no default window; no EX-E5, EX-S4, EX-S5 implementation; no standalone EX-V1 migration. No Plan dialect semantic change. No skeleton, CLAUDE.md or AGENTS.md edits. ChainDP remains the L1 path/seed. EX-E4 is not activated by rule3; its first-prologue-only ideal reduction is about 0.6% of the original median path, while inner-mainloop stalls remain a separate question. Shared-memory prefetch/union-lifetime changes are not made. sm_120 uses new local exports, calibration and Plans, never transplanted resident tables.

12. **Unmet gates and concrete next changes.** S1c-a/b: replace repeated dense producer-consumer expansion with shared interval/event readiness in ExecutionSimulator.cpp (the adjacency sweeps that initialize unmet counts, propagate completion, and reconstruct paths) and JOINT/project.cpp (grouped graph copying). The event heap already exists; compacting shared predecessor sets is the next missing mechanism. Cache this prepared form and update only changed queues, then rerun the unchanged budget/ranking set. S3-c: calibrate the K-loop fixed term in search.py/project.cpp (current 0.7 area + 0.3 operand formula ties K16/32/64), include register/smem residency effects, freeze the candidate domain and measure enough of it to establish an empirical percentile. Numeric exclusions: locate the first divergent GEMM, then select split per GEMM while preserving GemmCombineTaskBody's BF16-before-residual rounding; no tolerance inflation. Real-s128 remains around parity with its control after numerical exclusions leave only one competitive measured geometry: refill the valid shortlist with 64x128 K32/K64 at split1, price their K-loop iteration overhead and residency, and collect a new independent confirmation cohort. This is a concrete unmeasured next experiment, not an improvement claim. Strong reference speedups do not resolve these model/coverage gaps.

13. **Next round priority.** EX-V1 real-width main benchmark first, because its arithmetic admissibility and geometry-cost errors differ materially from narrow references; Fuse pricing second, using measured setup/epilogue plus avoided intermediate traffic as an explicit bound; EX-E4 shared-memory step2 third, conditional on measuring exposed waits inside the already pipelined mainloop and explicitly revisiting §8.6; EX-S5 parameterized placement fourth, after the cost model and chosen geometry stabilize. The next direct mechanism for lowering task cost is reducing/repricing K-loop work and padded tensor-tile work, with stage-specific split choices constrained by numerical error. The measured mainloop still contains memory and waits; declaring arithmetic saturation would be unsupported. First-operand L2 prefetch alone cannot explain or remove the remaining mainloop cost.

14. **Final integrity and delivery.** The final source/artifact parent-child SASS stamp is under PHASE/sass_identity. The verifier checks both model byte streams, all stamped source hashes, and that the artifact child changed only identity files. sm_120 SELF_CHECK outputs are retained under PHASE/self_check and JOINT/self_check; full target execution is pending by design. Push status and the final artifact hash are reported in the delivery message; if remote access fails, format-patch is emitted to /tmp/round5-patches/.
