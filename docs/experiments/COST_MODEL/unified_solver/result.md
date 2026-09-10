# A6 production solver control

✅ `tools/tilemega-solve.cpp:220` imports CG for the unified task-price
entry; its explicit `--legacy-task-cost` path preserves the historical FP32
generated-CUDA reader. BF16 imports its typed CG in both control arms.
Candidate enumeration includes all 770 archived BF16 rows with ptxas
resource records, even the 308 previously misclassified numerical failures;
only measured rank calculation is conditional on available PASS timings.

Two model-specific plan files (`plan_*.tsv`) list each GEMM's uniform,
split-only and per-op configurations. All four dtype/model old-versus-new
plan comparisons are byte-identical. The FP32 plans also match the archived
`SOLVER/raw` plans. General and separable objectives agree exactly for this
L1/historical-interface control. The separate new interface experiment uses
exact frontier DP and deliberately rejects the separable shortcut.

| dtype | model | old uniform ms | new uniform ms | old/new measured rank | old/new per-op ms |
|---|---|---:|---:|---:|---:|
| BF16 | gqa2 | .0890518 | .0922252 | 232/770, unchanged | .0588226 / .0619960 |
| BF16 | mha4 | .178104 | .184450 | 145/462, unchanged | .116037 / .122384 |
| FP32 | gqa2 | .0799104 | .0849473 | 14/1077, unchanged | .0775161 / .0825530 |
| FP32 | mha4 | .159928 | .170002 | 8/1077, unchanged | .159737 / .169810 |

✅ FP32 old and unified solver ranking checks pass. BF16 old and unified
checks both **FAIL** (exit1), since neither rank is inside the measured
top3% band (23/13). This is not a runtime failure and is not relabeled PASS.
The failure predates unified task pricing; raw old and new logs are retained.
Missing timings for the 308 numerical failures remain missing. Reported
residency is the solver's inference from archived candidate resources, not
a new megakernel occupancy measurement or a reopening of the closed T0 line.

The first unified executable SHA256 was
`e96d84d485d6495d7b6ffac7e882dc83c49806f4c750c92494313498f65a7bfc`;
it is frozen locally as `/tmp/tilemega-solve-a6-before-guards`.
The `f32_audited` repeat uses the explicit zero-reference tool footer and
`TILEMEGA_ISL_AUDIT=1`; it is pending at this report's checkpoint.

⚠️ Host evaluation remains expensive: the first unified uniform solve
took approximately 22.7/31.2s for FP32 and 28.5/52.4s for BF16, versus
millisecond legacy formulas. These are diagnostic wall times under concurrent
CPU verification, not paired performance measurements. Repeated QP
substitution in the current concrete evaluator is visible; no latency target
is claimed. This does not establish symbolic DP (B3) or L2 event transitions.

An initial build used an incorrect ScalarType namespace and a nonexistent
shared-storage alias. The compiler rejected both; the preserved build log
records it. The correction uses existing solver types and each scalar
TaskBody resource trait, not a new guessed shared-memory constant.
