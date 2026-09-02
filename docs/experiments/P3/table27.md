# P3.2 / P3.3 — §2.7 coupling table, derived vs tabulated

Reproduce with:

```
ninja -C build-portable tilemega-derive table27_test
./build-portable/tools/tilemega-derive llama     # docs/experiments/P3/derived-llama.md
./build-portable/tools/tilemega-derive decode    # S = 1 instantiation
./build-portable/table27_test                    # the assertions below, machine-checked
```

The derivation is `C = W_p^-1 o R_c` computed in closed form: every producer's
`W` is a tiling, so `W^-1` is exactly `floor(./g_p)` per tiled axis and the
projection of a read interval through it has three outcomes (exact quotient,
exact `floordiv` for a single element, or an explicit relaxation).  No
expectation below was moved to match what came out; the three places where the
derivation and the table differ are stated as differences.

Instantiation for the numeric cells: `H=4096, n_h=32, n_kv=8, G=4, d=128,
I=14336, Tm=Tn=Tkv=128`, and for evaluation `S=512, L_s=1024, past=512`.

## Row by row

| §2.7 | derived edge(s) | `C` | image | wait | fanout | Tier | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `rmsnorm1 -> wq/wk/wv` | `(m,n) -> {rmsnorm1(m)}` | `[ceildiv(S,Tm)]` | 1 | 32+8+8 = **48** | 0 | match |
| 2 | `wq -> rope_q` | `(m,hh) -> {wq(m,hh)}` | `[ceildiv(S,Tm) x n_h]` | 1 | 1 | 0 | match |
| 3 | `rope_k -> kvappend_k` | `(row,hh) -> {rope_k(floordiv(row,Tm),hh)}` | `[S x n_kv]` | 1 | 1 | **1** | wait/fanout/Tier match; `C` differs — see (b) |
| 4 | `kvappend_k/v -> attn_chunk` | guarded to `j == floordiv(past,Tkv)` | ragged | `Tkv`, **1 at S=1** | `S` | **2** | match under the decode instantiation — see (c) |
| 5 | `rope_q -> attn_chunk` | `(s,h,j) -> {rope_q(floordiv(s,Tm),h)}` | `[S x n_h]` | 1 | `ceildiv(L_s,Tkv)` | **2** | match |
| 6 | `attn_chunk -> attn_combine` | `(s,h) -> {attn_chunk(s,h,j) : j < ceildiv(L_s,Tkv)}` | `[S x n_h]` | `ceildiv(L_s,Tkv)` | 1 | **2** | match |
| 7 | `attn_combine -> wo` | `(m,n) -> {(s,h) : Tm*m <= s < Tm*m+Tm, 0 <= h < n_h}` | `[ceildiv(S,Tm)]` | `Tm * n_h` = 4096 | 32 | 0 | match |
| 8 | `wo -> add1` | `(m,n) -> {wo(m,n)}` | `[ceildiv(S,Tm) x ceildiv(H,Tn)]` | 1 | 1 | 0 | match |
| 9 | `add1 -> rmsnorm2` | `(i) -> {add1(i,n) : n < ceildiv(H,Tn)}` | `[ceildiv(S,Tm)]` | 32 | 1 | 0 | match |
| 10 | `rmsnorm2 -> wgate/wup` | `(m,n) -> {rmsnorm2(m)}` | `[ceildiv(S,Tm)]` | 1 | 112+112 = **224** | 0 | match |
| 11 | `wgate,wup -> silu` | two producer edges | `[ceildiv(S,Tm) x ceildiv(I,Tn)]` | 1+1 = **2** | 1 | 0 | match — see (a) |
| 12 | `silu -> wdown` | `(m,n) -> {silu(m,nk) : nk < ceildiv(I,Tn)}` | `[ceildiv(S,Tm)]` | 112 | 32 | 0 | match |
| 13 | `wdown -> add2` | `(m,n) -> {wdown(m,n)}` | `[ceildiv(S,Tm) x ceildiv(H,Tn)]` | 1 | 1 | 0 | match |

All 13 rows are auto-derived.  Aggregate claims also hold: no Tier 3 anywhere in
the dense model, ragged confined to the attention edges (rows 4-6), and 15 of
the 21 derived edges are Tier 0 (the table's "11/13" counts its own grouped
rows, of which 11 are Tier 0).

## Differences, stated rather than absorbed

**(a) Granularity of a "row".**  The table groups by consumer operator; the
derivation emits one edge per `(consumer, operand)` pair.  Rows 1, 10 and 11 are
therefore 3, 2 and 2 derived edges, and the table's cell is the sum over them
(48 = 32+8+8, 224 = 112+112, wait 2 = 1+1).  This is a presentation difference:
per-operand is what event synthesis needs, since each operand is a separate
`C_kappa`.

**(b) Row 3's `C`.**  The table writes `(m,hh) -> (m,hh)`, reusing `m` for both
the token block index and the KV cache row.  The append is modelled at its real
granularity — one cache row per task, which is what the decode path does — so
its consumer coordinate is a row and the projection into RoPE's row-block space
is `floordiv(row, Tm)`.  The two agree exactly when the append is tiled at `Tm`
rows.  The derived form is the more precise one; the table cell is the special
case.

**(c) Row 4's wait.**  The derivation returns `min(Tkv, S)` in the form
`Tkv` for symbolic `S`, and **1** for the decode instantiation `S = 1`
(`derived-decode.md`, edge 9).  The table's `wait = 1` is the decode case: a
single appended row can only be inside one KV chunk.  For a prefill pass with
`S > Tkv` a chunk genuinely waits on more than one append task, so the
parameterized form is the correct generalization of the cell, per F-14.

**(d) One edge the table omits.**  `add1 -> add2` (the second residual reading
the first residual's output) is a real 14th coupling.  It is asserted in
`table27_test` so it cannot silently disappear.

**(e) Where a tile choice is load-bearing.**  The QKV projections are given a
column tile of `d` (one head) rather than the generic `Tn`.  With a column tile
unrelated to `d`, RoPE's per-head read is not tile aligned and the derivation
**relaxes** — correctly, since a projection tile spanning parts of two heads
really does couple a RoPE task to more than one projection task.  §2.7's rows 2
and 3 hold because it sets `Tn = d = 128`.  This is recorded because it is a
constraint the solver (L2) will have to respect, not a free choice.

## I1: split-K is a reparameterization

`table27_test` takes the **derived** row 7 and applies `PartitionRange("h",
"kc", "Kc")`:

- `StructureKey()` unchanged, `SameStructure` true — `C` is not re-derived;
- the parameter list grows by exactly one entry, `Kc`;
- `wait` becomes `ceildiv(Tm * n_h, Kc)` = 1024 at `Kc = 4`.

## Tier 3 degradation

`derived-gather.md`: a gather whose index tensor is runtime data relaxes every
tiled producer axis to its full range, giving `C` = the whole producer task
space, `wait = fanout = |T_p|`, empty image, `exact = false`, reason
`data-dependent index`, Tier 3.  That is an operator-level barrier — the
conservative widening I2 permits — and not a fabricated affine map.

## Other models

`derived-llama4.md` (4 stacked layers), `derived-mlp.md` (3 MLP blocks, no
attention/KV/RoPE) and `derived-mha.md` (2 layers, `n_kv = n_h`, ungated MLP)
are derived by the same code with no model-specific branch.  The MLP stack
contains no Tier 1 or Tier 2 edge at all, which is the expected outcome: no
declared layout and no runtime extent.
