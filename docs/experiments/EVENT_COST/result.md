# Round 5 current status: A2 stopped; A9 event pricing not implemented

✅ Symbolic runtime counts and shared verified-CG projection seed are
implemented. Split1 task_refs/waits match 3000/3000 archived values; final
codegen refactor is byte-identical in 4/4 controls; CTest passes 27/27.

**Correctness stop:** one-process-per-cell capture using unchanged archived
BF16 binaries stopped after 94 PASS and 1 FAIL. gqa2 seq512/past0/split16 has
223287 L2-vs-L1 mismatches, max_abs=1.1054688. A static witness proves the
emitted split task order and runtime task order disagree on required rows.
This is not the closed BF16 criterion artifact. No subsequent GPU cell ran.

Implementation, exact scope, raw evidence, code locations, detours and the
user-approved kappa-direction interpretation are in
[A2 report](runtime_projection/result.md). A2 is not accepted, A9 is not
implemented, and B has not started. Counts matching an incorrect schedule
are insufficient evidence of semantic correctness.

# Historical T1 event pricing: rejected assumptions, not an acceptance report

Baseline: c8be09e. No L2 runtime optimization, numerical tolerance change,
cache-model replacement, or new GPU timing was made for this report.

## Steady-state calibration (T1.3)

✅ `calibrate.py:28` reprocesses OCCUPANCY `raw/attrib.tsv` (sm_89) and
`raw_sm120/attrib.tsv` (user-supplied sm_120 execution). It selects occ1 only,
checks all four arms in each of 25 rounds, and reads the original logs to
require cold=0/warmup=5/repeat=11 and identical task_refs/waits across arms.
The sm_120 data are not claimed as a new local hardware run. Source hashes
and exact coefficients are in `calibration/sm_*.json`.

Each cell uses the median of paired arm differences. The final coefficient
is a zero-intercept least-squares slope through all four cell medians.
Cross-validation fits three cells and predicts the omitted fourth.

| Target | notify, ns/runtime task_ref | poll, ns/runtime wait entry | fence |
|---|---:|---:|---|
| sm_89 | 14.0170996531 | 1.4387400428 | not_calibrated |
| sm_120 | 11.7653282731 | 1.5438009303 | not_calibrated |

✅ Recomputed leave-one-cell-out relative error `(prediction-observation)/observation`:

| Target | Cell | notify | wait |
|---|---|---:|---:|
| sm_89 | gqa2/4 | −94.8792% | −97.5955% |
| sm_89 | gqa2/128 | −4.4417% | −14.8543% |
| sm_89 | mha4/4 | −93.7623% | −96.5663% |
| sm_89 | mha4/128 | +32.1155% | +39.0872% |
| sm_120 | gqa2/4 | −96.0033% | −96.6235% |
| sm_120 | gqa2/128 | −3.1579% | +0.3514% |
| sm_120 | mha4/4 | −95.3424% | −96.4549% |
| sm_120 | mha4/128 | +40.4410% | +16.6853% |

These are **poor effective linear fits**, not validated device-primitive
latencies. The records in TargetSpec preserve units and fitting method;
`reason=measured` describes their measured source, not successful model
validation. All other target/dtype combinations explicitly have null rates
and `not_calibrated`. The four arms cannot separate the same-worker fence
rebate; it remains uncalibrated, not a guessed fraction of notify.

❌ Inference: parallel execution and fixed per-worker work may explain why
raw task count is a poor scalar predictor; this experiment does not isolate
that explanation. No new attribution to atomic fan-in is made.

## Actual CG metric-domain mismatch (T1.2 prerequisite)

✅ Reproducer, exit 2, full output in `metric_audit.txt`:

```
TILEMEGA_ISL_AUDIT=1 build-portable/tools/tilemega-event-cost /root/TileMega
```

`tools/tilemega-event-cost.cpp:35` imports the BF16 export as verified CG,
binds seq=4/past=3, and evaluates the stored four QPs, with SumDomain for
wait/fanout. The first gqa2 edge (stage 0 → 1) yields:

| Stored metric | Value |
|---|---:|
| sum(wait) | 512 |
| sum(fanout) | 16 |
| count | 4 |
| volume | 512 |

The relation is `{ [m=0,n] -> [p0] : 0<=n<=3 and 0<=p0<=127 }`.
The consumer has four N tiles. Its nominal M tile spans 128 producer rows,
but only four rows exist. This is **not** a rounding discrepancy in QP Eval.

Code evidence: `lib/Analysis/CouplingDerivation.cpp:490` explicitly documents
that the producer domain is applied only when deriving fanout; `ComputeMetrics`
uses `C.Card()` for wait but `C.IntersectRange(producer_domain).FanoutCard()`
for fanout. The comment records a prior barvinok bounded-counting failure
when the producer domain was folded into symbolic C. The offline affine
probe independently intersects C with actual task domains before scheduling
(`tools/tilemega-affine-probe.cpp:160`). Stored metrics therefore do not
automatically describe the same physical incidence graph on partial tiles.

The existing pointwise QP checks do not establish that stronger property.
This does not establish a runtime correctness failure: the runtime has its
own physical ownership/window bounds.

## Self-correction and current boundary

An uncommitted prototype multiplied `|image(C_kappa)|` by the fitted
notify coefficient. It was removed: the coefficient is per **runtime task
reference**, not per event group. Producer task count, per-edge event-group
count, shared producer events, split partial/combine projection, and lifted
runtime polls must be reconciled before such a multiplication has the
claimed meaning. Coarsening event groups does not by itself eliminate a
producer task's NotifyTask call.

The retained code is an **audit tool**, relation/QP primitives, and explicit
calibration records. It does not add an incomplete L2 price branch to
CostModel/ChainDP, change their L1 semantics, or advertise a nonzero κ price
as functional acceptance. Volume/fanout are inspected, not falsely claimed
to have been priced. Fusion and placement discounts are not implemented.

⚠️ T1.5 main gate **not passed / no valid prediction yet**. T1.1 cache
comparison, T1.4 symbolic intersections/DP, event-aware Fusion, and runtime
placement integration remain unimplemented. Correct physical-domain metrics
and a unit-consistent logical→runtime projection are prerequisites, not a
request to relax the gate. The large calibration residual also remains open.

Other corrections during implementation: the first calibration driver
incorrectly required unsafe no-wait arms to pass numerics; it now requires
the synchronized full arm and retains the unsafe-arm records. Target audit
then found omitted explicit-null schema fields on uncalibrated targets;
all five JSON schemas were completed rather than weakening the audit.
`target_audit.txt` records five targets, zero failures.
