# A3: exact element reads and explicit task-point binding

✅ `test/unit/task_element_work_test.cpp:1` checks 48 synthetic cells and
648 production BF16 cells (gqa2: 144 RoPE + 72 attention; mha4: 288 + 144).
Attention K reads are checked by bidirectional set containment, including
the first/last token and first/last query head. Five rejected inputs leave
the caller-owned isl context at its entry reference count; final count is 0.

`lib/Analysis/Semantics.cpp` declares RoPE's partner/frequency reads and the
causal attention K read predicate. `lib/Analysis/TaskWork.cpp:29` interprets
these indexing relations generically and unions repeated tensor addresses
before barvinok counting. It does not switch on operator kind. The physical
set is distinct from rectangular coupling projection and nominal issued
collective work. `TILEMEGA_EXACT_ELEMENT_WORK=0` preserves the old physical
rectangle calculation; it does not change the GEMM nominal-work gate.

For a query at token s, width D and total S+past, the implemented attention
body reads D Q elements, D·(past+s+1) K elements, and D·(S+past) V elements.
V is not causally clipped because the current body still loads it for masked
positions. Grouped-head addressing preserves the within-head offset rather
than dividing the entire flattened column by the head-group factor.

## Correction exposed by the new test

✅ The QP `[S,past] -> {[s,h] -> 8+4*S+8*past+4*s: ...}` is task-position
dependent. `QuasiPolynomial::Eval` substitutes isl parameters only; adding
`s` to that parameter binding did not fix a task coordinate. The earlier
header incorrectly described both namespaces as bound. This had gone
unnoticed in position-independent work tests.

`QuasiPolynomial::BindCoordinates` now explicitly restricts all named task
coordinates, then the existing Eval/SubstituteParams path binds θ. Missing
coordinates throw. The old Eval semantics and historical FP64 gate are not
changed. `test/unit/isl_relation_test.cpp` independently checks a triangular
relation at i=5 and its missing-coordinate error exit.

## Validation and detours

✅ Full build and 30/30 CTest; policy check; target audit: 5 targets, 0 failures.
Generated CUDA is byte-identical in 4/4 comparisons (2 models × default or
variant plan) to the frozen compiler used by the previous byte gate. Raw
logs are in `element_work_evidence/` and `element_work_codegen/`.

The initial implementation used an unavailable ClosedForm subtraction
operator; this was rewritten using negative multiplication. Initial isl text
used a parenthesized coefficient around floor syntax, which the parser
rejected; literal coefficients with `floord` fix the representation. The
subsequent task-coordinate exception revealed the API issue above, not an
expected-value change. Both failed tests and corrected tests are retained.
During recovery, an initial build command selected the obsolete `build/`
directory (MLIR disabled) instead of `build-portable/`; an audit invocation
passed a JSON path where the tool requires the repository root. Both command
errors were corrected without changing the build or audit policy.

⚠️ This is analysis-only evidence, not a new GPU synchronization claim.
Exact element projection currently rejects an unprojected split result axis;
the production unsplit attention paths above are covered, reference split
attention and A7 chunk projection are not. A3's solver-side CG transport and
A6 price consumption remain open. The 4308-group historical GEMM work gate
does not substitute for A6's stage-price bit gate.
