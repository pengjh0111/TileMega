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

verified: full MobileNetV1 fusion generates CUDA in
`runs/dm1-mbv1-fused-host-v17`; native execution and full-depth B16 MoE prefill
generation remain under investigation. Host checks above do not prove them.

verified: `runs/dm1-piece-screen-host-v2/results.json` passes exact metadata,
window accesses, dependency table/codec/codegen and DW/PW fixture checks. A
piece-bound signature rejects union-dependent candidate coordinates. Optional
pullback equality runs with an isolated ISL operation quota: quota exhaustion
rejects the optimization and retains exact counting, never accepts a proof or
relaxes dependencies. The caller's context/error/budget state is preserved.
Interval runs are parsed as small pieces and combined by balanced set unions.
The first implementation incorrectly called affine getters on map/unknown-div
spaces; host failures in `runs/dm1-piece-screen-host-v1` are preserved.

verified: the bound linear dependency path now proves its interval table first,
then tests whether every row has the same clamped affine window. Contiguous
endpoint equality against those proved rows certifies the window without
parsing a global pair-cardinality polynomial. Rows with holes keep the table.
Legacy symbolic binding is unchanged. Five targeted host checks pass in
`runs/dm1-linear-window-host-v1/results.json`; the first DW/PW invocation
omitted its fixture argument and failed before construction, and is retained.

verified: serialized dependency readers now validate canonical interval rows
directly against the fixed-consumer source map, proving both inclusions. They
do not repeat endpoint optimization or project away the fixed input coordinate.
Dimensions, bounds, maximal stride, zero padding and sorted maximal runs remain
checked. Seven targeted host checks pass in
`runs/dm1-direct-table-proof-host-v1/results.json`, including six new corruptions
and existing IR/runtime codec tamper rejection. The initial compile used an
unavailable ISL map API; its failed log is retained, and the repair preserves
the source tuple identity through the public API. No proof is skipped or
accepted on timeout. Native task arithmetic is unchanged.
