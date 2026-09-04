# WIRING — the derived coupling relation reaches the generator

Skeleton §1.5.1 / §0.2.  Before this round `lib/Frontend/Frontend.cpp` wrote a
constant into every `CouplingMapAttr`:

```cpp
analysis::CouplingRelation fixedRelation(llvm::StringRef /*kind*/) {
  return analysis::CouplingRelation::FromIslText("{ [0] -> [0] }");
}
```

`CouplingDerivation`'s input is an `OperatorGraph`; the frontend emitted a
per-`call_function` stage list and never built that abstraction, so
`DeriveCoupling` had no production caller.  This document is the evidence that
it now has one, and that what it produces is the analysis layer's own answer.

Reproduce with:

```
export PATH=/usr/local/cuda/bin:$PATH
ninja -C build-portable tilemega-wiring tilemega-derive wiring_coupling_test
./build-portable/tools/tilemega-wiring docs/experiments/E2E_GEN/raw/export_bridge.json reference
./build-portable/tools/tilemega-wiring docs/experiments/E2E_GEN/raw/export_bridge.json launch
./build-portable/wiring_coupling_test
```

Raw output is in `raw/`.  The fixture is the exported two-layer GQA model:
`H=512, n_h=4, n_kv=2, G=2, d=128, I=1024, Tm=Tn=Tkv=128`; `S` and `past` stay
isl parameters (I1).

## What the frontend now emits

✅ verified.  `FX -> ModelPlan -> L-sem -> Instantiate(g) -> CouplingDerivation`,
one `tilemega.task_space` per L-task node and one `tilemega.coupling` per
derived edge:

| fixture | granularity | task spaces | edges | Tier 0 | Tier 1 | Tier 2 | `{ [0] -> [0] }` |
|---|---|---|---|---|---|---|---|
| E2E_GEN (2 layers, GQA) | reference | 36 | 44 | 32 | 4 | 8 | 0 |
| E2E_GEN (2 layers, GQA) | launch | 34 | 42 | 34 | 8 | 0 | 0 |
| P3_GENERALIZATION (4 layers) | reference | 72 | 90 | 66 | 8 | 16 | 0 |
| P3_GENERALIZATION (4 layers) | launch | 68 | 86 | 70 | 16 | 0 | 0 |

`tilemega-opt` verify exits 0 on both, so the event-tensor-shape = `image(C_κ)`
check is live rather than trivially satisfied by a one-point relation.

Launch granularity carries no Tier 2 edge because attention is not split at
`Tkv` there: without a chunk axis there is no runtime extent, and the two
attention edges per layer collapse into one.  That is the granularity the
generated megakernel actually launches at; the reference granularity below is
§2.7's, used only for the comparison.

The degraded path is unchanged in shape and no longer a placeholder: the
`aten.imaginary.default` fixture yields 2 `generic` task spaces and 1 coupling
at **Tier 3**, because `LiftGenericSemantics` marks the operand maps
data-dependent.  `GenericSemantics`' full-range read is a sound cover but
prints as an exact affine edge, and no rule established an index for that
operator, so claiming Tier 0 would be an exactness the frontend never checked.

## Non-circular check: the frontend reproduces the analysis layer

Pinning the derived isl strings as expectations would be circular.  Instead
`test/unit/wiring_coupling_test.cpp` derives twice and compares:

- **left** — the exported FX graph, lifted, instantiated at `ReferenceGranularity`;
- **right** — `analysis::LlamaStackSem(shape, 2)` built at the *fixture's own*
  dimensions and instantiated at the same granularity.

44 edges each, compared field by field on `C`, event shape, wait, fanout,
volume, count, tier, attributes, guard and relaxation — 440 cells.

✅ verified: **4 differences out of 440, all one naming fact.**  The reference
names the second RMSNorm's parallel dim `i` where the semantic lifting names it
`m`, so edges 14 and 37 (`add -> norm`) differ in `C` and `wait` by that
identifier and nothing else.  `semantic_lifting_test` already pins ten records
of the same fact at the L-sem level; these four are it reaching the derived
relation.  The test asserts the difference list *equals* that expected set, so
a real divergence cannot hide behind it.

## The 14 §2.7 rows, item by item

Frontend edge numbers are from `raw/wiring-gqa2-reference.md`, layer 0.  Cells
are evaluated at the fixture's dimensions, so where §2.7's own cell is a
formula the two columns are compared as formulas.

| §2.7 | frontend edge(s) | `C` | wait | fanout | Tier | verdict |
|---|---|---|---|---|---|---|
| 1 | 1, 2, 3 (`norm -> q/k/v proj`) | `(m,n) -> {norm(m)}` | 1 | 4+2+2 = 8 = `n_h + 2·n_kv` | 0 | match |
| 2 | 4, 5 (`proj -> rope`) | `(m,hh) -> {proj(m,hh)}` | 1 | 1 | 0 | match |
| 3 | 6 (`rope_k -> append_k`) | `(row,hh) -> {rope_k(floordiv(row,Tm),hh)}` | 1 | **128 = Tm** | 1 | match, including note (f)'s correction |
| 4 | 9, 10 (`append -> attn`) | guarded `j == floordiv(past,Tkv)` | `Tkv` = 128 | **`2·S` = `G·S`** | 2 | `C`/wait/Tier match; **fanout differs — see (g)** |
| 5 | 8 (`rope_q -> attn`) | `(s,h,j) -> {rope_q(floordiv(s,Tm),h)}` | 1 | **`128·ceildiv(L_s,Tkv)` = `Tm·ceildiv(L_s,Tkv)`** | 2 | `C`/wait/Tier match; **fanout differs — see (g)** |
| 6 | 11 (`attn -> attn.combine`) | `(s,h) -> {attn(s,h,j) : j < ceildiv(L_s,Tkv)}` | `ceildiv(L_s,Tkv)` | 1 | 2 | match |
| 7 | 12 (`attn.combine -> wo`) | `(m,n) -> {(s,h) : Tm·m <= s < Tm·m+Tm}` | 512 = `Tm·n_h` | 4 = `ceildiv(H,Tn)` | 0 | match |
| 8 | 13 (`wo -> add1`) | `(m,n) -> {wo(m,n)}` | 1 | 1 | 0 | match |
| 9 | 14 (`add1 -> norm2`) | `(i) -> {add1(i,n) : n < ceildiv(H,Tn)}` | 4 = `ceildiv(H,Tn)` | 1 | 0 | match (the `i`/`m` naming difference above) |
| 10 | 15, 16 (`norm2 -> gate/up`) | `(m,n) -> {norm2(m)}` | 1 | 8+8 = 16 = `2·ceildiv(I,Tn)` | 0 | match |
| 11 | 17, 18 (`gate,up -> act`) | two producer edges | 1+1 = 2 | 1 | 0 | match — note (a) |
| 12 | 19 (`act -> wdown`) | `(m,n) -> {act(m,nk) : nk < ceildiv(I,Tn)}` | 8 = `ceildiv(I,Tn)` | 4 = `ceildiv(H,Tn)` | 0 | match |
| 13 | 20 (`wdown -> add2`) | `(m,n) -> {wdown(m,n)}` | 1 | 1 | 0 | match |
| 14 (note d) | 21 (`add1 -> add2`) | `(m,n) -> {add1(m,n)}` | 1 | 1 | 0 | match |

Edge 22 (`l0.s14.add -> l1.s15.norm`) is the inter-layer coupling, present
because the fixture stacks two layers; §2.7 tabulates one layer.

12 of 14 rows match outright.  The two that do not differ in the fanout cell
only, and the difference is in the document.

## (g) Rows 4 and 5 — the table's fanout, again

⚠️ The table's cells are `S` (row 4) and `ceildiv(L_s,Tkv)` (row 5).  The
frontend derives `G·S` and `Tm·ceildiv(L_s,Tkv)`.

This is not the wiring disagreeing with the analysis layer.  Running the
analysis layer's *own* reference derivation at §2.7's dimensions
(`raw/derived-llama.md`, `tilemega-derive llama`) gives edge 9 fanout `4 * S`
(`G = 4` there) and edge 8 fanout `128 * floor((127 + L_s)/128)` — the same two
shapes.  Both sides agree; the table does not.

✅ verified independently of TileMega: both cells follow by counting from the
table's **own `C` column**, with no isl involved.  At `S=7, n_h=8, n_kv=2, G=4,
Tm=3, Tkv=2, past=5, L_s=12`, enumerating `|{c : p ∈ C(c)}|` directly gives

- row 5: 18 for an interior producer block and 6 for the ragged tail, against
  `Tm·ceildiv(L_s,Tkv) = 18` and the table's `ceildiv(L_s,Tkv) = 6`;
- row 4: 28 uniformly, against `G·S = 28` and the table's `S = 7`.

The tail value 6 is why the table's cell looks plausible: it is the fanout of a
*partial* producer block holding one row, i.e. the per-consumer-coordinate
factor rather than the inverse-image cardinality.

The mechanism is exactly note (f)'s.  Row 5's producer `rope_q` emits `Tm`-row
blocks while `attn_chunk` consumes single tokens, so one block feeds `Tm`
tokens × `ceildiv(L_s,Tkv)` chunks; the table wrote the chunk factor and
dropped the token factor.  Row 4's producer `kvappend_k` emits one (row, kv
head) task while the consumer ranges over the `G` query heads sharing that kv
head; the table wrote the token factor and dropped the GQA group factor.  In
both cases the cell was written against the coarser model in which producer and
consumer are the same granularity — consistent with the table's prose, not with
the `C` the table itself states.

Per the standing rule, the expectation is not moved to fit the implementation:
the correction is made because the table is inconsistent with its own `C`
column under an independent count, and both values are kept visible.
`table27.md` gains note (g); §2.7 of the skeleton needs the same correction
(tracked in Part 6).

## End-to-end non-regression

✅ verified.  `docs/experiments/E2E_GEN/run.sh` re-run on the wired frontend,
RTX 4090 sm_89, 128 SMs, nvcc 12.8.  `raw/baseline-e2e_gen/` holds the
pre-wiring artefacts for comparison.

| | before wiring | after wiring |
|---|---|---|
| generated vs handwritten L0.5 | `5245714bc5d3ab4d`, bitwise PASS | `5245714bc5d3ab4d`, bitwise PASS |
| L0.5 vs PyTorch L0 | mismatch 0, max_abs `1.5497208e-06`, max_rel `0.0010710589` | **identical** |
| L1 vs L0.5 | mismatch 0, max_abs 0 | mismatch 0, max_abs 0 |
| L2 vs L1 | mismatch 0, max_abs 0 | mismatch 0, max_abs 0 |
| fresh processes | 50/50 pass | **50/50 pass** |
| import summary | tasks=179 couplings=222 | tasks=34 couplings=42 |

The numerical ladder is untouched: the L0.5-vs-L0 error is the same to every
digit, and every generated output still hashes to the pre-wiring value.  Only
the IR the generator reads changed.

Kernel times moved, and the honest reading is that they do **not** measure this
change:

| median over 50 processes | before | after |
|---|---:|---:|
| L0.5 | 1.103872 ms | 1.010976 ms |
| L1 | 1.095808 ms | 0.998400 ms |
| L2 | 1.110992 ms | 1.030144 ms |
| L2 / L1 | 1.013560× | **1.031795×** |

All three absolute times fell by about 9% — a machine-level shift, not
something the frontend can cause — so only the ratio is comparable, and the
ratio got *worse*.  ⚠️ That is not yet attributable: Codegen still emits
`Map::kAll` for every edge, so L2's synchronization structure is bit-for-bit
the same program it was before the wiring, and the derived `C` now in the IR
does not reach the wait path at all.  The re-measurement with attribution is
Part 3.1 and belongs after Part 2, not here.  What this round establishes is
only that wiring the analysis layer in cost nothing in correctness.

## What is still degraded

- Codegen still emits `kIdentity`/`kAll` only (`lib/Codegen/Codegen.cpp:176`),
  so the exact relation now present in the IR is not yet used to size a wait
  set.  That is Part 2 and it is the reason the L2 measurement has not been
  re-run yet.
