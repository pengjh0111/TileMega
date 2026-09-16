# R5 hierarchical Plan evaluation

`PreparedPlanBounds` walks the task DAG once per configuration. Per-Plan
`EvaluatePlanBounds` validates task coverage and computes queue/work bounds
in O(tasks); same-worker dependency node weights remain in the prepared CP.
`RankPlans` simulates top-k (k=3), including exact cutoff ties rather than
arbitrarily discarding equal-bound placements. `ranks.tsv` gives k=1..5.
The EFT materialization requires an explicit solved table and is covered by
JOINT, not invented by this historical-template driver.

The complete event simulator remains over its original per-Plan budget.
Only coarse evaluation fits. Therefore S1c-a is FAIL, not a renamed coarse
budget PASS. R5 §9.3 permits the reduced, fully measured candidate pool for
EX-S3. Exact preparation, coarse, full and batch times are all retained.
The graph preparation time is never presented as free.

`evaluations.tsv` and `ranks.tsv` are fresh CPU executions; measured ranking
uses R2's explicitly requested 18-point calibration set. It is historical
validation, not a fresh GPU speedup. New candidate ordering is measured in
JOINT. Default solver/runtime behavior is unchanged until explicitly opting
into the new ranking API. Existing full-simulator tests and the new bounds
contract test pass. Raw test commands/output: `bounds_test.json`.

## Publication calibration and replay

`calibrate.py` uses R4 raw five-arm logs plus the corrected reconstructed paths.
The pooled publication estimate is 1074.128 ns per publishing producer, the
residual waiting-consumer estimate is 1764.368 ns, and the independently
calibrated transfer constant stays 1235.412 ns. Actual stage event flags override
the idealized minimal cross-consumer mask. No runtime synchronization changes.

`replay.cpp` replays 68 historical R3/R4 dumps with observed task times (no second
co-residency multiplier). The 24 W>1 traces use actual recorded execution order
as a FIFO projection, explicitly a degraded approximation. Error distributions
are in replay_errors.json and every input path and prediction is retained.
The first exploratory fit attributed queue residuals to near-zero cross-hop
counts, yielding an unidentifiable 25.5 us coefficient; its inputs are archived
as preliminary_* and it is not used. These observational fits remain inferred
parameters, not identified fence or poll instruction latencies.
