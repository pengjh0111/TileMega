# Exact finite fibers and dependency rows

verified: the window, exact metadata and DW/PW fixture host checks pass with
coordinate screening, as recorded in `results/CI4_exact_fiber_row_proof.json`.
A direct constraint coupling only rejects a projection candidate; acceptance
still requires equality of the entire relation with the projected pullback.
Finite element boxes are proposed using a rational cover after removing local
divisions, and are counted as products only after exact equality. An accepted
independent coordinate is lifted directly into the count domain; it does not
require general Barvinok summation. Large printed polynomial partitions are
parsed piecewise by ISL and merged in a balanced tree. Disjoint addition is
used only after a support-disjointness proof; overlapping pieces retain ISL's
additive semantics. The tests include overlapping and fragmented sets.

inferred: a dependency table is globally exact when its complete bounded
consumer domain is covered and each distinct interval row has both inclusions
proved against its source row. The implementation performs these row proofs
and retains the original compact affine relation as the certified description,
avoiding an enormous consumer-row disjunction. The codec independently
reconstructs and compares intervals; this does not trust a mutable table merely
because its stored relation is exact. Existing table/codec/codegen tests pass.

Full MobileNetV1 fusion and B16 full-depth MoE prefill generation remain under
investigation. Host checks above do not claim those graph paths are complete.
