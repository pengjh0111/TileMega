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

---

# Part 3: solving-authority migration (round 2)

The sections above answered "can we depend on isl at all". These answer
"is the migrated derivation right", i.e. the task's Part 3 acceptance items.

## (a) Cross-validation: §2.7 re-derived through isl

✅ `table27_test` re-derives the whole coupling table with `C` as an isl_map
and every metric as a barvinok count, and compares against
`docs/experiments/P3/table27.md`'s corrected baseline. Every row's `wait`
matches, and every row's `fanout` matches except one, which is a genuine
finding rather than a representation difference:

| §2.7 row | quantity | table | isl re-derivation | verdict |
|---|---|---:|---:|---|
| 1 | fanout (Wq+Wk+Wv) | 48 | 32 + 8 + 8 = 48 | match |
| 2 | wait / fanout | 1 / 1 | 1 / 1 | match |
| **3** | **fanout** | **1** | **128** | **table is wrong, see below** |
| 3 | wait, Tier | 1, 1 | 1, 1 | match |
| 6 | wait | ⌈L_s/Tkv⌉ | 8 at L_s=1024 | match |
| 7 | wait / fanout | Tm×32 / 32 | 4096 / 32 | match |
| 9 | wait / fanout | 32 / 1 | 32 / 1 | match |
| 10 | fanout (Wgate+Wup) | 224 | 112 + 112 = 224 | match |
| 11 | wait (both operands) | 2 | 1 + 1 = 2 | match |
| 12 | wait / fanout | 112 / 32 | 112 / 32 | match |
| 13 | wait / fanout | 1 / 1 | 1 / 1 | match |

**Row 3's tabulated fanout of 1 is wrong.** `kvappend_k`/`kvappend_v` tile the
cache row axis by 1 — one task per row — while `rope_k` produces rows in
Tm = 128-row blocks. So one producer block is needed by 128 consumer
row-tasks: `fanout(p0) = |C⁻¹(p0)| = 128`, not 1. `wait` is unaffected (a row
still needs exactly one block).

The pre-migration implementation also reported 1, which is why this was never
caught: its fanout was a heuristic — *"a consumer coordinate that occurs in C
is pinned by y and contributes a factor of 1; one that does not is free and
contributes its whole range"* — which silently assumes every occurrence is
recovered bijectively from the producer coordinate. That is true for the
identity occurrence (`hh ↦ p1`) and false for the floordiv occurrence
(`row ↦ ⌊row/Tm⌋`, many-to-one). isl computes the real inverse-image
cardinality, so the assumption has nowhere to hide. The skeleton's §2.7 table
and `docs/experiments/P3/table27.md` are corrected accordingly; the
expectation was not moved to fit the implementation in either direction.

## (b) Coarsen: the operation the migration exists to unlock

✅ `C_κ = ⌊·/κ⌋ ∘ C` is `isl_map_apply_range` with a floor map. `AffineRelation`
had no image, preimage or composition operator at all, so this was not
merely unimplemented before — it was inexpressible. Raw data:
`raw/coarsen.txt`, reproduced by `run.sh`.

`wait` divides by κ exactly, on both producer axes and on both a
2-D and a 1-D producer coordinate:

| edge | κ | wait |
|---|---|---:|
| attn_combine→wo | {1,1} | 4096 |
| attn_combine→wo | {1,2} | 2048 |
| attn_combine→wo | {1,4} | 1024 |
| attn_combine→wo | {2,1} | 2048 |
| attn_combine→wo | {4,1} | 1024 |
| attn_combine→wo | {4,4} | 256 |
| silu→wdown | {1,1} / {1,2} / {1,4} | 112 / 56 / 28 |
| rmsnorm1→wq | {1} / {2} / {4} | 1 / 1 / 1 (already minimal) |

Two algebraic laws are asserted in `coupling_types_test` as well as measured
here: κ=1 is the identity, and `⌊⌊·/2⌋/2⌋ == ⌊·/4⌋`. **Both were false on the
first implementation**, and the composition law is what caught it: the fresh
output names `Coarsen` generated (`q0, q1, …`) collided with the range names
of an already-coarsened relation, and isl reads the resulting
`q1 = floord(q1, 2)` as a constraint on a single variable whose only solution
is 0 — silently collapsing that coordinate to a point instead of halving it.
Fixed by choosing a collision-free prefix and renaming the range dims back
afterwards.

### P4.6's flagged question: does isl blow up on parameterised division?

⚠️ **Measured answer: no explosion for these patterns.** The skeleton flagged
`[!] 待验证：ISL 对含参数化整除的映射是否表达式爆炸`. Across the sweep above,
both the relation and the resulting quasi-polynomial stay at **one piece**,
and their isl text stays flat in κ:

| κ | C_κ text (chars) | C_κ pieces | wait text | wait pieces |
|---|---:|---:|---:|---:|
| {1,1} | 100 | 1 | 49 | 1 |
| {1,2} | 100 | 1 | 49 | 1 |
| {1,4} | 99 | 1 | 49 | 1 |
| {4,4} | 96 | 1 | 48 | 1 |

Leaving S symbolic costs a constant ~15 characters (115 vs 100) and still one
piece. So κ need not be restricted to powers of two on expression-size
grounds. ⚠️ This is measured for one decoder layer's edges at κ ≤ 4 and a
single coarsening axis at a time; deeper nesting is untested.

## (c) End-to-end: no regression

✅ Both accepted models produce byte-identical output to the pre-migration
generator, with the same bit hashes:

| model | L0.5 hash | L1 | L2 | fresh processes |
|---|---|---|---|---|
| 2-layer GQA | `5245714bc5d3ab4d` | bitwise equal | bitwise equal | 50/50 |
| 4-layer MHA | `fd15fa2e89cdb915` | bitwise equal | bitwise equal | 1/1 |

Generated vs handwritten L0.5 also still matches bitwise. The migration
changed how the Analysis layer computes and represents couplings; it did not
change a single generated instruction.

## (d) A case where the quasi-polynomial is load-bearing

✅ The skeleton's §1.5.1 previously recorded that *"当前测试集内两者重合，尚未构造出
需要真正拟多项式的反例"*. Here is one. Raw data: `raw/quasipoly.txt`.

`MisalignedTileModel(producer_tile, consumer_tile)` has a producer tiling the
row axis by one size and a consumer reading it tiled by another. When the
tiles do not divide one another, the number of producer blocks a consumer
block straddles varies periodically with the consumer coordinate:

| producer_tile | consumer_tile | wait | one scalar? |
|---|---|---|---|
| 96 | 160 | `(2 - ⌊(1+r)/3⌋) + ⌊(2+r)/3⌋` | **no** — 2 or 3, period 3 |
| 100 | 128 | `(2 - ⌊7r/25⌋) + ⌊(6+7r)/25⌋` | **no** — 2 or 3, period 25 |
| 96 | 192 | `2` | yes (aligned control) |
| 128 | 128 | `1` | yes (identical tiles) |

Two things matter here, and the aligned controls in the last two rows are
what separate them:

1. **The edge is derived exactly** (`exact`, no relaxation, Tier 0). Producer
   block `p` is coupled precisely when its interval overlaps the read
   interval — `p·tile < base + span` and `base < p·tile + tile` — which is
   affine, so isl holds it directly. The pre-migration path could express
   neither that two-sided condition nor a count over it, so it relaxed this
   entire class of edge to the full producer axis and raised the tier. This
   is a precision *gain*, and it changes nothing previously covered: the
   exact-quotient and single-element rules still fire first, and no reference
   model reached this branch before.
2. **The result has no single scalar value**, and `Eval` says so
   (`max 3 != min 2`) rather than silently returning one of them. A `wait`
   field of scalar type — which is what `ClosedForm` gave the metrics before
   this migration — cannot represent this edge at all without being wrong.

## Residual limitation found while doing this

⚠️ Bounding the producer (range) tuple inside `C` itself drives barvinok into
`unexpected missing (bounded) solution` (`basis_reduction_tab.c:210`) on the
*wait* side, for a relation that carries both a genuine isl parameter and an
inequality-range-derived producer coordinate. Leaving the range unbounded
instead makes *fanout* report a spurious `min = 0`, because `isl_map_card`'s
piecewise decomposition of the reversed map keeps a tail that is only
reachable for other parameter values and evaluates to 0 there. The resolution
is to apply the producer box **only to the reversed map**, inside the fanout
computation — each direction then stays in the regime its own counting
problem is tractable in. This is documented at `ProducerRangeBoxText` and
recorded in `TileMega_skeleton.md` §1.5.1.
