# A9 structured event prices

✅ BF16, sm_89: 600/600 fresh-process correctness runs and 1200 four-arm
processes (12 cells × 25 paired rounds × 4 arms), warmup=5/repeat=11.
The cells are both reference models × seq={1,4,16,128,512,2048}, past=3.
No GPU timings overlapped with the A2 matrix or A12.2 calibration. This is a
new steady-state dataset; historical cold-start attribution is not reused.

## Recovery and provenance

The connection interruption left 1100 attribution rows: eleven **complete**
cells, ending at mha4/512 round24. `run_calibration.py --resume` first checked
the eight frozen binary and ptxas hashes, then reconstructed every existing
table row from its raw log and checked execution order before appending the
last cell. No four-arm pair straddles the interruption. `resume_prefix.json`
records the old table/command hashes. Raw data, compiler logs, command,
build hashes and before/after artifact checks are in `calibration_round5/`.
The target hashes there describe the files at measurement time, before the
subsequent explicit schema/calibration commits; they are not current hashes.

## Counts × rates, units and residuals

`fit_structured.py:74` fits three nonnegative coefficients per term over
twelve paired-cell medians. It enumerates NNLS faces, with no intercept,
regularizer, positive epsilon or fitted E2E ranking target. Full-rank checks
precede the fit. The measured coefficients are:

| term | ns/runtime stage | ns/max-worker task ref | ns/total work item |
|---|---:|---:|---:|
| notify | 2030.8367987160177 | **0** | 4.184597721499793 per task ref |
| poll | 824.1297704910431 | **0** | 0.5246761917316038 per wait entry |

The longest-queue feature is included but the constrained fit selects zero
for both coefficients. This dataset therefore provides **no calibrated
nonzero queue-length price** for placement. It does not establish that queue
length is free; the coefficients are effective exposed costs, not individual
instruction latencies. `fence` stays explicitly `not_calibrated`, so its
rebate is zero. No fraction of notify is assigned to it.

Leave-one-cell-out relative errors are in
`calibration_round5_fit/cross_validation.tsv`: notify −16.86% to +30.55%;
poll −37.18% to +95.06%. The structured fit still fails to predict some cells
well. Per A10, that residual is reported, not treated as the A9.3 gate.
No new BF16/FP32 ranking claim follows from fitting event costs.

`decomposition.tsv` contains each paired median, 95% bootstrap interval
(10000 resamples, seed0), exact two-sided Wilcoxon sign-randomization p,
and closure error. Selected steady-state terms, in microseconds:

| model / seq | notify median [95% CI] | wait median [95% CI] |
|---|---:|---:|
| gqa2 / 4 | 66.272 [64.768,66.848] | 24.576 [23.392,24.992] |
| gqa2 / 128 | 71.456 [70.912,71.680] | 23.840 [23.552,24.576] |
| mha4 / 4 | 114.752 [113.920,117.600] | 44.000 [42.176,46.080] |
| mha4 / 128 | 153.344 [153.024,153.664] | 40.960 [39.776,41.600] |

All eight displayed signed-rank p values are 5.960464477539063e-8.
Maximum per-round algebraic closure error is 0 ns. This algebraic identity
does not independently prove the absence of cross-binary resource effects;
registers/residency/grid and spill remain present in every raw row.

## Actual consumption and functional gate

`ModelDescription.h:48` stores runtime QPs **inside coupling_metrics**.
`RuntimeProjection.cpp:243` attaches the same exact verified-CG projection
used by A2, including split rewrite and ownership. `CostModel.cpp:475`
substitutes semantic parameter aliases and evaluates these QPs; mismatched
g, split, κ, grid, threads or rewritten stage count are rejected. `event_ns`
replaces only the synchronization choice: L1's barrier expression and FP64
sum order are retained when `l2_events` is OFF (the compile-time default).
Controls: `TILEMEGA_L2_EVENT_COST`, `CostModelOptions::l2_events` and `kappa`.
Fusion/fence-free producer QPs default to explicit zero until B supplies them.

`tilemega-event-price` imports BF16 CG, projects each κ separately, and
evaluates the **measured** target profile. It does not read generated CUDA
tables. Both direct-concrete and explicit-SubstituteParams paths produce
identical price bits: **36/36**. `verify_price.py` independently compares
task refs, waits, stage count and longest queue against all 300 full-arm
processes: **1200/1200 exact**. Both tool exits show zero remaining ISL refs.
Raw QPs, prices, hashes and the verification are in `calibration_round5_fit/`.

| model / seq | waits κ0 → κ1 → κ2 | event κ0 → κ1 (ns) | κ1−κ0 (ns) |
|---|---|---|---:|
| gqa2 / 4 | 244 → 500 → 392 | 86613.937611294285 → 86748.254716377574 | 134.31710508328979 |
| gqa2 / 128 | 4520 → 16292 → 11008 | 106499.71700098175 → 112676.20513004619 | 6176.4881290644407 |
| mha4 / 4 | 588 → 1076 → 888 | 173749.01778656972 → 174005.05976813473 | 256.04198156500934 |
| mha4 / 128 | 9168 → 33092 → 23296 | 225988.63031849652 → 238540.9835294834 | 12552.353210986883 |

✅ A9.3 passes in all twelve cells. κ0 is the runtime's whole-stage aggregate
special case, not the beginning of positive κ coarsening. The user approved
this interpretation after seeing the exact wait counts: κ0→1 increases waits
and price, κ1→2 decreases both. Runtime task refs do not change with κ.
No κ performance/argmin conclusion is inferred from these prices.

## Remaining boundary and corrections

The new target fields initially failed target-audit; all five configs now
declare missing structured/partial rates explicitly, without weakening the
schema audit. A test initially used a private QP constructor; it now uses the
public canonicalizing parser. The missing-rate unit test now constructs
missing data explicitly rather than depending on sm_89 remaining uncalibrated.
The synthetic unit rates never enter this report's acceptance values.

⚠️ `ChainDP.cpp:148` explicitly rejects L2 event mode until candidate-specific
counter transitions are integrated. A9 concrete pricing is not an L2 DP
implementation, A6 GEMM price acceptance, unified non-GEMM pricing, or B
acceptance. A6 is still open, so B has not started. sm_120's script exists but
was not run; no structured sm_120 coefficients were invented.
