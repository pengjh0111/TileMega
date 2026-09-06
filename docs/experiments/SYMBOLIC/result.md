# Part 1 — derived metrics as functions of theta, not integers at S_min

Reproduce:

```
cmake --build build-portable --target tilemega-symbolic-probe
./build-portable/tools/tilemega-symbolic-probe > raw/probe.txt
```

## What was wrong

`lib/Frontend/Frontend.cpp` built its `known` binding by pinning **every**
symbolic dimension to the bottom of its declared range:

```cpp
for (auto const& symbol : symbolic.dimensions)
  known.Bind(symbol, symbolic.ranges.at(symbol).minimum);
```

isl genuinely requires literal divisors (`isl_aff_div` rejects a parametric
one), so tile sizes and the GQA group factor must be numbers. The workload
dimensions did not have to be, and binding them collapsed `wait`, `fanout`,
`volume` and `count` from quasi-polynomials in `S` / `past` / `L_s` into one
integer measured at the smallest instantiation. P5.1's whole definition —
"a cost function parameterized by theta, a DP that emits piecewise
quasi-polynomials, solved for the interval boundaries" — cannot be built on an
integer measured at `S_min`.

Only the granularity is bound now. `floorBinding` survives in exactly one
place: an event tensor is a real allocation, so its *shape* is still resolved
at the smallest instantiation and any axis that is not constant there is
emitted `kDynamic`.

## The metrics are now parametric on the production path

✅ verified. The imported CG for the two-layer GQA model carries, e.g.

```
wait = #tilemega.metric<"[s11] -> { [m, n] -> 128 : ... }">
```

where before the parameter had been substituted away. `tilemega-wiring` prints
the same for every edge (`[S] -> ...`, `[S, past] -> ...`).

## (a) The symbolic metric equals a fresh derivation at the point

Pinning the symbolic text as an expectation would be circular, so the probe
derives twice: once with `S`/`past`/`L_s` free, once at each concrete point,
and asks `SemanticallyEqual` — which compares the two as *functions* of
whatever task coordinate remains, rather than collapsing both to a scalar.

| | |
|---|---|
| model | one Llama decoder layer at §2.7's theta (H=4096, n_h=32, n_kv=8, d=128, I=14336, Tm=Tn=Tkv=128) |
| edges | 21 |
| sequence lengths | `S ∈ {1, 4, 128, 512, 2048}`, `past = 3` |
| metrics compared | `wait`, `fanout`, `volume`, `count` |
| **agreements** | ✅ **420 / 420** |

## (b) What the floor binding was claiming

The same table, derived at `S = 1, past = 0` — what the frontend used to store
— against the truth at `S = 4096` (`raw/probe.txt`):

| | count |
|---|---:|
| metrics the floor binding got **wrong** | **27** |
| metrics that happen to be genuinely constant | 36 |
| metrics with no single value (position-dependent) | 0 |

The errors are not small and not uniform:

| edge | metric | floor | true at S=4096 | under-estimate |
|---|---|---:|---:|---:|
| `attn_chunk -> attn_combine` | **wait** | 1 | 32 | **32×** |
| `rope_q -> attn_chunk` | fanout | 1 | 4096 | 4096× |
| `kvappend_k -> attn_chunk` | fanout | 4 | 16384 | 4096× |
| `rope_k -> kvappend_k` | fanout | 1 | 128 | 128× |
| `rope_q -> attn_chunk` | count | 32 | 4194304 | 131 000× |
| every `count` on a row-tiled edge | count | — | — | 32× |

The first row is the one §2.7 names explicitly: `wait = ⌈L_s/Tkv⌉` on the
attention combine. `T_sync = |image(C_κ)| × latency` reads exactly that
quantity, so at a 4096-token context the old constant under-charged the
synchronization of that edge by **32×**, and the error grows linearly with the
KV length. Every `count` on a row-tiled edge is off by `⌈S/Tm⌉`, i.e. by 32×
at this length, because the floor binding leaves exactly one row tile.

## The window re-fit is still needed

`div/scale/offset/count` in `StageDependency` are `constexpr` integers in the
generated table, so they cannot become quasi-polynomials without adding runtime
fields to the emitted table and evaluating them per launch. The three-point
re-fit at `S ∈ {256, 384, 512}` therefore stays exactly as it was: it is the
mechanism that keeps a window admissible only where it holds at every probed
length, and degrades the pair to `kAll` (a superset) otherwise. ⚠️ Making the
window itself symbolic is a separate change and is **not** made here; it is the
natural companion to P5.1's runtime interval selection, which already carries
per-variant tables.

## Non-regression

✅ verified, same session, RTX 4090 sm_89, BF16 with the structured ownership
plan:

| | gqa2 | mha4 |
|---|---|---|
| fresh processes | **50 / 50** | **50 / 50** |
| distinct L2 output hash | `4c544373c0add101` | `853cf67220dbad5f` |
| dependency mix | 10 `kAll` / 7 `kIdentity` / 21 `kWindow` | — |

Both hashes are bit-identical to the ones the ORACLE sweep recorded before this
change, and the dependency mix is unchanged. Making the metrics parametric
changed what the IR carries, not what the kernel computes. All 24 ctest cases
pass.

## What is not done here

The cost model does not yet *read* these quasi-polynomials. `T_sync` is still
the calibrated grid-barrier curve charged once per stage, and
`ModelDescription::FromGeneratedCuda` parses only the producer/consumer pair
out of the dependency table. That was not a defect worth fixing before this
change — there was nothing but a wrong constant to read — and it is now
possible for the first time. ⚠️ Wiring it in changes what the cost model
predicts and therefore has to be re-validated against the oracle, so it belongs
with the cost-model work rather than here.

## Pre-existing, unrelated

`tilemega-compile` prints `isl_ctx not freed as some objects still reference
it` at exit. ✅ It is present in all 1540 codegen logs of the previous round,
i.e. it predates this change; it is a shutdown-time leak in the tool, not a
correctness issue, and it is recorded rather than fixed here.
