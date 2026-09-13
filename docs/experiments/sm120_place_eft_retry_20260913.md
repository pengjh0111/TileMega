# sm_120 PLACE_EFT preparation repair, 2026-09-13

Verified: the original failure was a preparation error, not an incorrect
runtime guard. The wrapper copied a materialized sm_89 EFT table whose
worker/slot arrays required grid 256, while the RTX 5090 selected grid 340.
The guard and runtime grid selection remain unchanged.

The repaired preparation probes frozen control sources on sm_120 and solves
new plans from their measured resources, task graph and resident grid.
It uses an experiment-local BF16 target and the previously measured
SIMULATOR sm_120 hop coefficients, rather than sm_89 calibration/placement.
Prediction uses explicit modulo worker-SM assignment when no trace exists;
this is an approximation, not an observed hardware mapping.

Verified: driver rebuild, three CPU regression tests and SELF_CHECK passed.
The initial preparation retry rejected the uncalibrated repository BF16
profile. A local 41-repeat calibration completed with calibrated=true.
That flag is not a claim that every fitted model is statistically adequate:
the calibration log reports StreamK R-squared -11.8374, and the hop fit has
reduced chi-square 1936.5. Repository target defaults were not overwritten.

## Retry invocation

Implementation revision: `2918f55b`.

```bash
cd /root/TileMega
BUILD_DIR=/root/TileMega/build-cluster \
CUDACXX=/usr/local/cuda-12.8/bin/nvcc CUDA_VISIBLE_DEVICES=0 \
REALWIDTH=0 RUNS=25 CORRECTNESS_RUNS=50 \
OUT_DIR=/root/TileMega/docs/experiments/PLACE_EFT/raw_sm120_retry_20260913 \
bash docs/experiments/PLACE_EFT/run_sm120.sh
```

Verified on RTX 5090 (170 SMs): all four reference plans use grid 340;
the four control-source provenance comparisons are byte-identical. The
24-arm placement-statistics pass and 1200/1200 fresh-process correctness
checks passed (24 combinations, 50 independent invocations each).

## Paired performance: negative S2-b result

Verified from 600 paired timing processes: six arms per reference cell,
25 rounds per arm, with the script's original rotating invocation order.
The EFT/rotate paired ratios and bootstrap 95% confidence intervals are:

| Cell | Median ratio | 95% CI |
| --- | ---: | --- |
| gqa2, seq=4 | 1.0409 | [1.0389, 1.0436] |
| mha4, seq=4 | 1.0542 | [1.0524, 1.0561] |
| gqa2, seq=128 | 1.0528 | [1.0510, 1.0548] |
| mha4, seq=128 | 1.0468 | [1.0457, 1.0486] |

The pooled 100-pair ratio is 1.0491 [1.0473, 1.0510]. EFT is slower than
rotate in all four cells, so the original improvement gate fails 0/4.
Predicted EFT/rotate ratios are 0.9940, 0.9940, 0.9966 and 0.9961 in the
table's order: predicted and measured improvement directions disagree 4/4.
No parameters, expected values or acceptance thresholds were adjusted.

REALWIDTH is disabled; no real-width result or full repository verification
suite success is claimed. The original sm_120 failure and sm_89 records are
preserved; retry outputs use their own directory. Per-process measurements
are retained using collect_samples.py, without committing executables.

## Completion and evidence

Verified: the wrapper exited zero and wrote PASS. All 800 synchronization
decomposition invocations completed (32 combinations, 25 rounds each).
The safe full probes passed 200/200; the intentionally unsafe probes
reported MISMATCH 600/600 and are timing-only, not numerical acceptance.
The 25 full-probe processes per combination are not a separate 50-process
race guarantee. The formal 50-process-per-arm correctness results above
remain the correctness evidence.

The collected [samples](PLACE_EFT/raw_sm120_retry_20260913/samples.tsv)
contain exactly 2600 processes: 1200 correctness, 600 paired timing and
800 decomposition. The [summary](PLACE_EFT/raw_sm120_retry_20260913/summary.tsv),
[selective S2-b/d/e checks](PLACE_EFT/raw_sm120_retry_20260913/research_checks.txt),
[session](PLACE_EFT/raw_sm120_retry_20260913/serial_session.txt),
[calibration](PLACE_EFT/raw_sm120_retry_20260913/calibration.txt),
[plan solver](PLACE_EFT/raw_sm120_retry_20260913/prepare_solver.txt), and
[grid assignments](PLACE_EFT/raw_sm120_retry_20260913/eft_schedule.tsv)
are preserved. S2-b/d/e were recomputed selectively against retry data;
this is not a claim that the full cross-experiment verifier passed.
