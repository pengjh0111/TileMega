# Round 5 Production Placement

✅ Correctness: **400/400 fresh BF16 processes**, two models, seq={4,128},
two compilation states, 50 rounds per cell. Entire state/product rotates each
round; warmup=5, repeat=11. Original mapping is 0, new mapping is 4.
Default remains 0. No tolerance changed. sm_120 was not run.

✅ Shared runtime projection handles split/attention-expanded stages.
`lib/Solver/RuntimeProjection.cpp:BalanceProjectedQueues` consumes its task
and dependency relations; `lib/Solver/BalancedPlacement.cpp` scores incoming
local edges under a baseline maximum queue cap and validates queue+DAG order.
`lib/Codegen/RuntimeTaskGraph.cpp` materializes already-expanded stage windows
for the host; the projection unit test checks its edge set against ISL.
`ModelHarness.cuh` consumes per-task owners for queue entries, local poll
elision, span accounting and shard ownership. No fence is removed.

✅ L-sched `mapping_mode="balanced"` requires `resident_only=true` on every
placement. Lowering preserves the flags in RuntimeVariantDesc, and launch
rejects a grid beyond actual L1/L2 occupancy. This is **resident-only**, not a
parameter-domain over-resident proof. Old controls 0/1/2/3 remain available.

## Main Result: Negative

First 25 paired processes are the primary performance sample; all 50 are
reported as sensitivity in `balanced_runtime/summary.json`. Bootstrap uses
20,000 resamples of paired rounds. Wilcoxon uses the two-sided normal
approximation with tie/continuity correction (p about 1.3e-5 in all four).

| BF16 model / seq | L2 baseline ms | balanced ms | paired ratio, 95% CI | predicted event delta ns |
|---|---:|---:|---|---:|
| gqa2 / 4 | .401408 | .658272 | 1.639908 [1.637304, 1.642060] | -140.613 |
| gqa2 / 128 | .558080 | 1.723392 | 3.087011 [3.084404, 3.089908] | -678.931 |
| mha4 / 4 | .797696 | 1.114048 | 1.396662 [1.394589, 1.400177] | -304.312 |
| mha4 / 128 | 1.137664 | 4.554752 | 4.002253 [3.983017, 4.004169] | -2967.57 |

✅ Counts reconcile against all **200/200 new-mapping process records**:
waits 524→256, 13012→11718, 1140→560, 27044→21388 respectively.
Both arms retain registers=212, CTAs/SM=2, TaskSmem=24576; all ptxas logs
are archived. Maximum queues remain 30,38,60,84. Full resource/schedule,
L0.5/L1/L2 measurements, hashes, generated sources and build commands are in
`balanced_runtime/`; event predictions are in `projected_runtime/`.

❌ A9.3 placement **performance-sign gate does not pass**: the prediction is
nonzero and correctly follows fewer polls, but end-to-end performance moves
the other way. No coefficients were changed to fit these measurements.
The new mapping is an explicit experimental control, not a selected optimum.

⚠️ Diagnosis: equal *task counts* do not guarantee equal work or shorter
dependency critical paths. This mapping co-locates dependencies but can
serialize expensive chains; one poll record can spin for variable time.
The current counts-times-rates event model cannot represent that timing.
Count agreement and unchanged resources rule out two confounders, but do
not establish how much of the gap comes from work imbalance versus waiting.
That attribution remains unmeasured; no claim of a proven root cause.

## Repairs and Limits

An added empty-dependency relation initially emitted untyped `{}`, rejected
by ISL. It now emits an explicitly typed empty map; the original projection
unit test's edge-free KV case passes. A new IR roundtrip test initially used
the wrong printed map syntax (`array<i64: 0>` instead of `[0]`); corrected
without changing its resident-constraint assertion. A compile-time variable
name typo was corrected before any GPU run.

⚠️ Full candidate DP selection is still not implemented. These plans are
explicit comparison states, not solver-selected placements. The manual
sm_120 runner exists but no actual sm_120 candidate manifest or result is
claimed. B1 Fusion and B3 symbolic DP are independent unfinished work.
