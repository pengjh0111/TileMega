# Part 1: isl/barvinok dependency feasibility (ISL/barvinok migration, round 1)

Evidence labels: ✅ executed/observed; ⚠️ scope limitation; ❌ conjecture.

This experiment answers the task's Part 1 (dependency feasibility) and
produces the one piece of hard evidence Part 3's design depends on: whether
isl can represent the symbolic (θ, g) expressions TileMega's coupling table
actually needs. See `docs/DEPENDENCIES.md` for the full build recipe, version
pins, and the five required answers; this file is the experiment record.

## crosslink_probe.cpp — does isl/barvinok coexist with MLIR?

✅ Yes. Built and run both as a standalone binary and as the CMake target
`isl_crosslink_test` / ctest `isl_crosslink` (guarded by
`-DTILEMEGA_ENABLE_ISL=ON`). One process: builds an `mlir::ModuleOp`,
exercises MLIR's own bundled `mlir::presburger::IntegerRelation`, builds an
`isl_set`, computes its cardinality through barvinok's `isl_set_card`, and
checks a subset relation through `isl_set_is_subset`.

```
[mlir-presburger] { [i] : 0<=i<10 } isIntegerEmpty=0
[isl+barvinok] card([n]->{[i]:0<=i<n}) = [n] -> { n : n > 0 }
[isl] {[i]:0<=i<5} subset of {[i]:0<=i<10} = 1
RESULT mlir_ok=1 isl_barvinok_ok=1
```

`ctest -R isl_crosslink`: 1/1 passed.

## parametric_div_probe.c — can isl represent a *symbolic* tile-size divisor?

This is the question that actually decides how Part 3's migration has to be
shaped, and it was not asked by the task text explicitly — it surfaced while
trying to build the very first isl expression TileMega would need
(`floordiv(m, Tm)` with `Tm` symbolic, exactly §2.7's `wq(floordiv(m,Tm))`).

**Attempt 1: `Tm` as a genuine isl parameter (matching how `ClosedForm`
represents it today — symbolic in both θ and g until `Eval()`).**

```c
isl_aff *m  = isl_aff_var_on_domain(ls, isl_dim_set, 0);      /* m */
isl_aff *tm = isl_aff_param_on_domain_space_id(space, tm_id); /* Tm, a param */
isl_aff *quotient = isl_aff_div(m, tm);
```

Result: isl itself rejects this at the C API level, not just in the text
parser used for the smoke tests above —

```
third_party/barvinok/isl/isl_aff.c:3502: second argument should be a constant
```

`isl_aff_div`'s divisor argument must be a literal constant. This is not a
syntax limitation of the `iscc` text parser (which was tried first and also
rejected symbolic divisors, e.g. `[m/Tm]` and `m = Tm*q`); it is a property of
`isl_aff` itself: an isl affine expression's `floor`/`div` node stores a
*rational constant* denominator, so a genuinely parametric divisor is not
representable as a single `isl_aff` at all — it is not a Presburger-affine
object in isl's data model, independent of how it is spelled.

**Attempt 2: `Tm` as a literal (128), `S` (the token count) as the isl
parameter.**

```
$ iscc <<< 'card [S] -> { [m] : 0 <= m < S and exists (q = [m/128] : 128 q <= m) };'
[S] -> { S : S > 0 }
$ iscc <<< 'card [S] -> { [m] : 0 <= m < (S + 127)/128 };'
[S] -> { floor((127 + S)/128) : S > 0 }
```

Both work immediately and give the expected closed forms (`S`, and
`ceildiv(S,128)` in the second form).

**Conclusion, and why it is not actually a new constraint.** isl can only
build affine expressions whose divisors are literal constants; a coupling's
tile sizes (`Tm`, `Tn`, `Tkv`, ...) must therefore be bound to literals
*before* isl ever sees the expression, while the workload-dependent dimensions
(`S`, `H`, `n_h`, ...) stay as genuine isl parameters. This is not a gap the
migration introduces — it is the same boundary V-F already established for
CuTe's `RightInverse` ("`g` 固定了 tile 内的 W → 特化后用 CuTe 静态
`RightInverse`", skeleton §3.5), now shown to hold identically for isl. It
also matches what the frontend already does in practice:
`lib/Frontend/Frontend.cpp` sets `tilemega.g` to a literal dictionary
(`token_tile=1, hidden_tile=128, head_tile=1, head_dim_tile=128`) at import
time — `g` is not actually symbolic in the CG module today, only `ClosedForm`
keeps it unevaluated syntactically until `Codegen.cpp` calls
`.Eval(theta, granularity)`. An isl-backed representation has to do that
substitution earlier: when `DeriveCoupling`/`ComputeMetrics` build the
`isl_map`/`isl_pw_qpolynomial`, `g`'s concrete values must already be
substituted in (read from the module's `tilemega.g` attribute or the
`ParamBinding` passed to `CouplingDerivation`), while `theta` stays as real
isl parameters carried all the way to the generated code's runtime binding
(invariant I1). This is a concrete, buildable design, not an open question —
it is recorded here so the actual Part 3 rewrite does not have to
re-discover it.

## What Part 1 leaves open for Part 3

- The literal-`g` / parametric-`theta` boundary above is now proven, not
  assumed; Part 3's `DeriveCoupling`/`ComputeMetrics` rewrite should build
  `isl_map`s with `g` pre-substituted from `ParamBinding` and `theta` as isl
  parameters, consistent with how `tilemega.g` is already stored as literals
  in the CG module.
- Everything downstream of this boundary (§3.5's acceptance: cross-validating
  §2.7 against the isl path, demonstrating Coarsen, constructing a genuine
  piecewise-quasi-polynomial use case, deleting `ClosedForm`/`AffineRelation`,
  migrating the dialect attributes and verifiers, rewriting the five affected
  unit tests) is **not done in this round**. It is a substantially larger
  body of work than Part 1, and is recorded as open in
  `TileMega_skeleton.md` §1.5.1 rather than attempted under time pressure and
  left half-correct.
