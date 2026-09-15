# R4 task-DAG reconstruction audit

✅ Verified offline from the 32 R3 placement traces and 12 R3 window traces. No historical file was changed. Input paths and SHA256 digests are in `manifest.tsv` and `inputs_sha256.tsv`. These are reanalyses of historical measurements, not new R4 kernel timings.

The three historical dump tables do not contain the unsimplified DAG: `schedule.tsv` contains offsets, `waits.tsv` has already lost local and lifted waits, and `events.tsv` identifies publication rows. Reconstruction therefore reads the matching generated `kDependencies0` table, expands its runtime windows, and checks each traced task's dependency slice. Trace `(worker, slot, stage, logical_task)` supplies materialized sigma.

All original analyzer metrics remain available with `_legacy` suffixes. The corrected zero-sync DAG bound includes all producer nodes even on zero-cost same-worker edges. Explicit executor edges are distinct: at W=1 the preceding slot must complete; at W>1 every slot at least W positions back must complete. These edges enter the separate `cp_plan_nosync_ns`, not the placement-independent task-DAG bound. Actual execution-order edges are used only for causal reconstruction.

Zero-globaltimer-duration nodes remain on tied longest paths (maximize node count, then break ties by task coordinates). This preserves meaningful node counts without assigning invented durations to sub-tick work. Cycles and unknown predecessors are rejected.

## A-a / A-b / A-c

44/44 satisfy measured >= max(corrected DAG bound, busiest-worker work), including all 32 required placement dumps. 44/44 causal interval partitions close exactly, with nonnegative gap and wait <= kernel span. gqa2 s4 A rotate and chain both have 20 nodes and 242688 ns task weight.

For the split, each predecessor-to-successor interval starts at the predecessor's run_end. Publish gets its measured portion first, then only the non-overlapping remainder of the successor wait and pre-run barrier is counted. The final node's publication tail is included. Root pre-run delay is excluded and the start anchor is the root run_begin. Thus `cp_chain_span_ns` and the reconstruction have the same endpoints. No negative values are clamped to make the partition pass.

## A-d: rotate under A/B/D/W

| model | seq | config | legacy DAG bound us | corrected DAG bound us | corrected reconstruction us | trace span us |
|---|---:|---|---:|---:|---:|---:|
| gqa2 | 128 | a | 89.088 | 346.112 | 454.656 | 455.680 |
| gqa2 | 4 | a | 242.688 | 242.688 | 286.720 | 287.744 |
| mha4 | 128 | a | 91.136 | 697.344 | 948.224 | 948.224 |
| mha4 | 4 | a | 244.736 | 488.448 | 572.416 | 572.416 |
| gqa2 | 128 | b | 90.112 | 349.184 | 441.344 | 441.344 |
| gqa2 | 4 | b | 242.688 | 242.688 | 282.624 | 282.624 |
| mha4 | 128 | b | 90.112 | 695.296 | 909.312 | 909.312 |
| mha4 | 4 | b | 245.760 | 491.520 | 567.296 | 568.320 |
| gqa2 | 128 | d | 115.712 | 351.232 | 462.848 | 462.848 |
| gqa2 | 4 | d | 250.880 | 250.880 | 289.792 | 289.792 |
| mha4 | 128 | d | 206.848 | 710.656 | 972.800 | 973.824 |
| mha4 | 4 | d | 253.952 | 507.904 | 586.752 | 586.752 |
| gqa2 | 128 | w | 112.640 | 342.016 | 466.944 | 467.968 |
| gqa2 | 4 | w | 244.736 | 244.736 | 286.720 | 286.720 |
| mha4 | 128 | w | 203.776 | 708.608 | 993.280 | 994.304 |
| mha4 | 4 | w | 247.808 | 493.568 | 579.584 | 579.584 |

The corrected bound does not consult worker assignment, wait counts, or W. Its remaining differences across traces are different observed node durations (including 1024-ns quantization and configuration-dependent tracing/CTA execution), not disappearance of DAG edges. The new regression test holds one trace fixed and varies W=1/2/4, obtaining the same DAG bound. No duration renormalization or placement-specific adjustment is applied.

## Window reassessment

| model | seq | W | legacy cp us | corrected cp us | legacy HOL us | corrected HOL us |
|---|---:|---:|---:|---:|---:|---:|
| gqa2 | 128 | 1 | 187.392 | 324.608 | 79889.408 | 22872.064 |
| gqa2 | 4 | 1 | 136.192 | 232.448 | 1274.880 | 483.328 |
| mha4 | 128 | 1 | 196.608 | 685.056 | 207584.256 | 47318.016 |
| mha4 | 4 | 1 | 136.192 | 463.872 | 7692.288 | 1936.384 |
| gqa2 | 128 | 2 | 338.944 | 342.016 | 86979.584 | 14131.200 |
| gqa2 | 4 | 2 | 245.760 | 246.784 | 1347.584 | 314.368 |
| mha4 | 128 | 2 | 679.936 | 684.032 | 217905.152 | 44335.104 |
| mha4 | 4 | 2 | 482.304 | 489.472 | 8433.664 | 1215.488 |
| gqa2 | 128 | 4 | 337.920 | 342.016 | 86464.512 | 2158.592 |
| gqa2 | 4 | 4 | 239.616 | 242.688 | 1360.896 | 62.464 |
| mha4 | 128 | 4 | 681.984 | 688.128 | 216441.856 | 23089.152 |
| mha4 | 4 | 4 | 483.328 | 486.400 | 8445.952 | 185.344 |

The corrected HOL estimate includes every semantic predecessor when testing future readiness. It counts only the part of a head stall after a later task actually becomes data-ready. Whole waits are no longer counted merely because some later slot has an empty, already-lifted wait interval. W=4 reduces this estimate in all four window cells; the old conclusion that the window reclaimed no HOL is not supported by the corrected analysis. HOL is summed across workers and must not be added directly to kernel elapsed time.

### Original paired timing is independent of the analysis formula

| model | seq | paired W2/A ratio median | ratio of medians | raw pairs |
|---|---:|---:|---:|---:|
| gqa2 | 4 | 1.035531 | 1.037915 | 25 |
| gqa2 | 128 | 1.044141 | 1.044843 | 25 |
| mha4 | 4 | 1.038869 | 1.038651 | 25 |
| mha4 | 128 | 1.047805 | 1.047619 | 25 |

The analyzer does not produce E2E_TIME. Its repair removes **0% of the raw paired timing regression**: 100% of that recorded end-to-end delta remains. This does not quantify how much is hardware noise versus scan/probe/fence overhead. A new paired experiment is necessary for that causal allocation; no physical-overhead percentage is invented here. The measurement artifact was in the inferred lower bound and HOL interpretation, not in the arithmetic that timed the kernel. W remains disabled by default.

## Limits

Only rotate and chain have the 32 historical placement dumps. E needs new A traces for the other candidates before a complete freeze. Exact-ISL runtime descriptors fail explicitly in this parser; they need an authoritative task-DAG export. No poll-derived fallback is presented as a corrected DAG.
