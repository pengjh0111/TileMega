# Tier orthogonalization: five attributes, tier as a derived summary

`§2.4` gives one number per edge. That number answers several unrelated
questions at once, and the answers move independently, so the summary loses
information the scheduler needs. This records the decomposition, what it is
checked against, and what it does *not* change.

## The five attributes

Declared in `include/tilemega/Analysis/CouplingDerivation.h`, assigned in
`CouplingDerivation::Derive`, carried in the IR as
`#tilemega.coupling_attrs<...>` on `tilemega.coupling`.

| attribute | values | what the derivation reads |
|---|---|---|
| `relation_kind` | `affine`, `layout_mediated`, `data_dependent` | a declared `layout_id` on either endpoint or the operand; `AccessRelation::data_dependent` |
| `extent_kind` | `static_literal`, `symbolic_static`, `runtime_dynamic` | `OperatorNode::HasRuntimeTaskSpace()`; otherwise whether any extent/origin/tile still names a symbol |
| `exactness` | `exact`, `relaxed` | `CouplingDetail::exact` (was the projection widened under I2) |
| `runtime_requirement` | `none`, `prefix_sum`, `tensor_values` | read off the *source* of the uncertainty, not off another attribute |
| `countability` | `constant`, `quasipoly`, `piecewise_quasipoly`, `uncountable` | the printed form of `wait`: `;` (several cells) or `floor` (periodic coefficient) → piecewise; a literal value → constant |

`extent_kind` is deliberately independent of the derivation-time `known`
binding. I1 is a statement about what L-sem says, not about which symbols
happened to be bound when an edge was derived, so an extent written as `S`
is `symbolic_static` even when `known` binds `S = 1536`.

## Tier as a function of the five

```
data_dependent or uncountable            -> 3
runtime_dynamic or relaxed               -> 2
layout_mediated                          -> 1
otherwise                                -> 0
```

`DeriveTier` is the only place a tier is produced; `CouplingEdge::tier` is
always `DeriveTier(attributes)`, asserted for every §2.7 edge in
`table27_test`. `CouplingOp::verify` recomputes it from the IR attribute and
rejects an edge whose written tier does not follow (`cg_attr_test`, two
negative cases: an exact static affine edge relabelled Tier 2, and a runtime
extent relabelled Tier 0 — both produce a diagnostic naming the expected
tier).

## ✅ Verified: no derived conclusion changes

`./build-portable/tools/tilemega-derive {llama,decode,llama4,mlp,mha,gather}`
regenerated all six `derived-*.md`. The `tier` column is **byte-identical to
the committed files for all six models** (diff of column 10 only, empty for
each). All 13 ctest tests pass.

Distribution over the six models (221 edges):

| tier | attributes | edges |
|---|---|---|
| 0 | `affine + symbolic_static + exact + none + constant` | 136 |
| 1 | `layout_mediated + symbolic_static + exact + none + constant` | 16 |
| 2 | `affine + runtime_dynamic + exact + prefix_sum + constant` | 8 |
| 2 | `affine + runtime_dynamic + exact + prefix_sum + piecewise_quasipoly` | 8 |
| 2 | `layout_mediated + runtime_dynamic + exact + prefix_sum + constant` | 16 |
| 3 | `data_dependent + symbolic_static + relaxed + tensor_values + uncountable` | 1 |

## What the summary was hiding

**Tier 2 is three different things.** In one Llama layer:

| edge | attributes |
|---|---|
| `rope_q -> attn_chunk` | `affine + runtime_dynamic + exact + prefix_sum + constant` |
| `kvappend_k -> attn_chunk` | `layout_mediated + runtime_dynamic + exact + prefix_sum + constant` |
| `attn_chunk -> attn_combine` | `affine + runtime_dynamic + exact + prefix_sum + piecewise_quasipoly` |

All four Tier-2 edges in §2.7 are **exactly** derived. The tier cannot say
that: §2.4 assigns Tier 2 both to a runtime extent (one prefix sum at launch,
the index map still closed form) and to an over-approximated projection
(nothing needed at run time, but the relation is no longer the true one).
`table27_test` asserts both facts — three distinct tuples under Tier 2, every
one of them `exact`.

**Tier 0 does not mean "one number".** The misaligned-tiling edge
(`coupling_types_test`, producer tiles rows by 96, consumer by 160) comes out

```
affine + symbolic_static + exact + none + piecewise_quasipoly
```

— tier 0 by the rule above, and correctly so: nothing about it is ragged,
relaxed or data dependent. Its `wait` is nevertheless genuinely piecewise
with period `lcm(96,160)/160 = 3`, and `QuasiPolynomial::Eval` refuses to
collapse it. Before the split, "tier 0" was routinely read as "wait is a
constant"; `countability` is what makes that reading impossible to write down.

## ⚠️ Finding: the committed `derived-*.md` were stale

The six files had not been regenerated since `b1dbc38`, i.e. since before the
isl/barvinok migration. Their `C`, `wait` and `fanout` columns were still in
the pre-migration AST syntax (`(m,n) -> {rmsnorm1(m)}`); the current tool
emits isl text. Regenerating them here therefore changes those columns as
well as adding the `attributes` column. That drift is pre-existing and
unrelated to this change — the evidence that this change is inert is the
`tier` column, which the stale files and the fresh ones agree on edge by edge.

Related rot, **not** fixed here: `test/Dialect/CouplingGraph/*.mlir` still
reference `#tilemega.closed_form`, an attribute the migration deleted, so all
three fail to parse. They are not wired into ctest (no lit/FileCheck in this
build), which is why nothing caught it. The new verifier rule is covered by a
C++ negative test in `cg_attr_test` instead, so it does not depend on the
dead lit files being revived first.

## ❌ Inferred, not verified

`runtime_requirement` never disagrees with the other attributes on any model
in this repository: every `runtime_dynamic` edge wants a prefix sum, every
`data_dependent` edge wants the index tensor. It is kept as its own attribute
because it answers a different question (what the launch path must supply),
not because a counterexample has been observed here. The verifier deliberately
does not enforce a correlation between the two.
