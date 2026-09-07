# Queue-driven κ ablation

Evidence status: ✅ measured on RTX 4090 (`sm_89`), BF16, 2026-09-07, after
logical-task events, kAll aggregation, and selective event publication. No
timing from the retired stage loop or either intermediate queue executor is
used.

`run.sh` compiles exactly `κ ∈ {0,1,2,4,8,16,32}`, rotates arm order over
25 fresh-process rounds, and requires every arm to match PyTorch/L1. It reports
within-round ratios, a 20,000-resample bootstrap interval, and a two-sided
Wilcoxon signed-rank test. `κ=0` is aggregate-only; positive κ groups that many
consecutive logical producer tasks. A stage publishes only the aggregate
and/or fine rows actually referenced by its consumers.

## Result

| model | κ | L2 median (ms) | paired change vs κ=0 | bootstrap 95% CI | p |
|---|---:|---:|---:|---:|---:|
| gqa2 | 0 | 0.618336 | — | — | — |
|  | **1** | **0.616448** | **−0.285%** | **[−0.481%, −0.031%]** | **3.36e-3** |
|  | 2 | 0.619520 | +0.191% | [+0.000%, +0.498%] | 1.63e-2 |
|  | 4 | 0.620736 | +0.497% | [+0.321%, +0.664%] | 1.84e-4 |
|  | 8 | 0.619712 | +0.331% | [−0.015%, +0.503%] | 1.43e-2 |
|  | 16 | 0.620288 | +0.300% | [+0.000%, +0.498%] | 1.46e-2 |
|  | 32 | 0.620544 | +0.489% | [+0.166%, +0.659%] | 7.09e-4 |
| mha4 | 0 | 1.257568 | — | — | — |
|  | **1** | **1.255296** | **−0.165%** | **[−0.401%, +0.000%]** | **4.47e-3** |
|  | 2 | 1.262624 | +0.407% | [+0.094%, +0.651%] | 9.62e-5 |
|  | 4 | 1.262592 | +0.326% | [+0.246%, +0.484%] | 3.42e-5 |
|  | 8 | 1.268736 | +0.809% | [+0.729%, +0.973%] | 1.31e-5 |
|  | 16 | 1.259712 | +0.155% | [+0.000%, +0.244%] | 1.15e-2 |
|  | 32 | 1.264640 | +0.570% | [+0.407%, +0.654%] | 1.47e-5 |

✅ Correctness is 25/25 for every model/κ pair: **350/350 fresh
processes**. L1 is the null control; its intervals mostly cross zero, and the
conclusion uses paired L2 contrasts rather than unpaired medians.

✅ The benefit side is now nonzero. `κ=1` is the measured argmin for both
models and beats aggregate-only by 0.285% / 0.165%. The effect is small: mha4's
bootstrap upper endpoint rounds to zero, although its signed-rank test rejects
the paired zero median. Larger groups lose the readiness benefit without
removing the per-stage event machinery.

## Decision

The reference-model default is changed from `κ=0` to `κ=1`. The common
winner means κ need not enter the DP state for these two models, but the old
claim that the argmin is structurally fixed is withdrawn. The generator keeps
all seven values because graph fan-in and producer imbalance can move the
optimum.
