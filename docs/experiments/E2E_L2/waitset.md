# Part 2 — the generated dependency table, and what its wait sets are worth

Reproduce with:

```
ninja -C build-portable
./build-portable/tools/tilemega-compile docs/experiments/E2E_GEN/raw/export_bridge.json /tmp/gen_e2e.cu
nvcc /tmp/gen_e2e.cu -std=c++17 -O2 -arch=native -DTILEMEGA_EVENT_KAPPA=8 \
  -Iinclude -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include \
  -Ithird_party/cutlass/test build-portable/libtilemega.a -lcudart -o /tmp/gen_e2e_k8
TILEMEGA_WAIT_PROFILE=4,512 /tmp/gen_e2e_k8 docs/experiments/E2E/fixture
./build-portable/tools/tilemega-shape-probe          # per-edge fitted form
./build-portable/tools/tilemega-identity-probe .     # kIdentity admissibility
```

## What the generator now emits

`StageDependency` carries `{producer, consumer, map, div, scale, offset,
count}`; a consumer CTA waits on producer tasks
`[(task / div) * scale + offset, ... + count)` rather than on the producer's
whole launch axis. Over the 2-layer GQA model's 38 stage pairs (✅ verified,
`raw/dependency_table_forms.txt`):

| map | pairs | before this round |
|---|---:|---:|
| `kAll` | 20 | 38 |
| `kIdentity` | 3 | 0 |
| `kWindow` | 15 | 0 |

The CG carries the same thing per coupling as the `wait_map` attribute (42
couplings: 20 `all`, 7 `identity`, 15 `window(...)`); the generator joins them
per stage pair, and one relaxed or disagreeing member forces the pair back to
`kAll`.

## The six shapes of §2.7, and which one each of the 14 edges used

`tilemega-shape-probe` fits each derived edge (`raw/edge_forms.txt`). Layer 0
of the exported model, which is one instance of the §2.7 table:

| §2.7 | edge | fitted form | shape |
|---|---|---|---|
| 1 | `norm1 -> wq / wk / wv` | `window(4,1,0,1)`, `window(2,1,0,1)` x2 | projection `(m,n) -> m` |
| 2 | `wq -> rope_q` | `identity` | identity `c -> c` |
| 3 | `rope_k -> kvappend_k` | `all` | floordiv — **not narrowed**, see below |
| 4 | `kvappend -> attn_chunk` | `all` | ragged, runtime extent — **not narrowed** |
| 5 | `rope_q -> attn_chunk` | `all` | floordiv — **not narrowed** |
| 6 | `attn_chunk -> attn_combine` | `window(1,6,0,6)` | one-to-many interval, ragged extent |
| 7 | `attn_combine -> wo` | `window(128,16,0,16)` | one-to-many interval |
| 8 | `wo -> add1` | `identity` | identity |
| 9 | `add1 -> norm2` | `window(1,128,0,128)` | one-to-many interval |
| 10 | `norm2 -> wgate / wup` | `window(256,1,0,1)` x2 | projection |
| 11 | `wgate, wup -> silu` | `identity` x2 | multi-producer set, wait = 2 |
| 12 | `silu -> wdown` | `window(128,256,0,256)` | one-to-many interval |
| 13 | `wdown -> add2` | `identity` | identity |
| 14 | `add1 -> add2` (residual) | `identity` | identity |

So five of the six shapes are carried exactly by the affine-interval encoding.
The sixth — rows 3, 4 and 5 — is **not** narrowed, and the reason is ownership,
not derivation: RoPE, KVAppend, activation and the split-K combiner declare
`kElementChunk`, i.e. the CTA owns a grid-stride slice of a flat element range,
so `blockIdx.x` names a different set of tasks in the two stages and no
per-CTA window is admissible. Those edges stay `kAll` by construction. Row 4's
ragged extent is therefore never exercised as a window in this model; the
encoding supports it (`window(1,6,0,6)` on row 6 is fitted against a runtime
`ceildiv(L_s,Tkv)` extent), but row 4's ends disagree on ownership first.

## Correctness

150 fresh processes, 50 at each of kappa = 0, 1, 8 (`scripts/gpu_stat_run.sh`,
`raw/part2_fresh_processes.txt`): ✅ 150/150 pass, `l2_vs_l1_mismatch=0` in
every one, one distinct L2 output hash `5245714bc5d3ab4d` equal to L1's, and
the L0 (PyTorch) error unchanged at `max_abs=1.5497208e-06`,
`max_rel=0.0010710589` — the same values as before the window existed. All 22
ctest cases pass.

Two things had to be fixed for that to be true, both recorded rather than
quietly absorbed:

* **Grid-stride placement (found by review, not by a failing test).** A stage
  whose task count exceeds the resident grid is run grid-strided, so CTA `b`
  owns tasks `b, b+grid, ...` and task `t` is published by CTA `t % grid`. The
  first version of `WaitDependencies` used `PlacedBlock()` as *the* consumer
  task and clamped the producer window to `min(count, grid)` — under-waiting on
  both sides. It cannot fire on this fixture (every stage is narrower than the
  grid at `seq = 4`), so the 150/150 above would not have caught it. The wait is
  now the union over the tasks the CTA owns, with the producer interval mapped
  back onto CTA indices modulo the grid.
* **Split-K.** `BuildModel` rewrites a consumer's producer to the *combine*
  stage, whose ownership is `kElementChunk`; such edges are forced back to
  `kAll` because `blockIdx.x` no longer names the tile the window was fitted
  against.

## What the wait sets are actually worth (attribution for Part 3.1)

`TILEMEGA_WAIT_PROFILE=<seq list>` runs a device-side counting kernel that
reports, per edge, the number of event polls the L2 path performs and what the
same edge would cost as `kAll` (`raw/waitset_profile.txt`). It is a counting
what-if: substituting `seq` answers "how wide would this wait set be at that
length" without a fixture at that length; nothing is dispatched.

Total polls over all 38 edges, 2-layer GQA:

| kappa | seq=1 | seq=4 (the fixture) | seq=128 | seq=512 |
|---|---|---|---|---|
| 1 | 1.103x | **1.024x** | 1.000x | 1.000x |
| 8 | 1.000x | 1.000x | 1.000x | 1.000x |
| 32 | 1.000x | 1.000x | 1.000x | 1.000x |

**The narrowing is between 1.00x and 1.10x.** That is the honest number, and it
is why L2 is still not faster than L1. The reason is not that the windows are
loose — they are exact — but where the poll mass sits. At `seq = 512`,
`kappa = 1`:

| ownership of the two ends | map | edges | share of poll mass |
|---|---|---:|---:|
| chunk -> tile | `kAll` | 8 | **67.6%** |
| chunk -> chunk | `kAll` | 2 | **22.3%** |
| tile -> chunk | `kAll` | 10 | 4.2% |
| tile -> tile | `kWindow` | 15 | 5.9% |
| tile -> tile | `kIdentity` | 3 | 0.03% |

89.9% of the polls are on edges with a `kElementChunk` end, where a per-CTA
window is inadmissible. The 15 window edges — everything Part 2 narrows — carry
5.9% of the mass, so **5.9% is the ceiling on what an exact window can remove
here even if it removed all of it.** The three identity edges do narrow 4x
each, and carry 0.03%.

Within that 5.9% the windows currently remove nothing, for a reason worth
recording separately: every fitted window on a tile-row edge has
`count = scale = Tm = 128`, which on this GPU equals `gridDim.x = 128`. Under
the grid-stride placement a contiguous task interval one stride wide covers
*every* CTA, so the window maps back to the whole launch axis. A blocked
placement (`task -> task / ceil(count/grid)`) would preserve it; the same
counting model over the same edges gives 1.047x at `kappa = 1` and 1.082x at
`kappa = 32` (`raw/placement_whatif.txt`, ❌ inferred — a counting model, not a
run). Even that is bounded by the 5.9%.

The actionable statement is therefore not "the derivation is too weak" but:
the wait set of this model is dominated by element-chunk placement, which is a
§2.3 **Place** decision, and no amount of exactness in `C` moves it.

## What `kIdentity` is worth (Part 3.3)

`tools/tilemega-identity-probe` now asks the production path — the derived isl
`C`, and `LiftedOp::ownership` cross-checked against `OwnershipOf`, the table
the TaskBodies themselves return from `Ownership` (TaskBase.h). The previous
probe answered the ownership half from a hand-written table keyed on operator
name prefixes; a name is not a declaration and nothing kept it in step.

| | gqa2 | mha4 |
|---|---:|---:|
| derived edges | 42 | 86 |
| ranks differ (no identity possible) | 13 | 27 |
| `C ⊆ identity` | 9 | 19 |
| …and both ends `kTilePerBlock` (`kidentity_admissible`) | **7** | **15** |
| …both ends `kElementChunk` (not claimed — see below) | 2 | 4 |
| lifted-vs-TaskBody ownership disagreements | **0** | **0** |

Two `kElementChunk` stages would also share a CTA→task map, but only if they
linearize the same number of elements over the same grid, which the CG task
space does not record. They are counted separately rather than admitted.

These are not comparable to the old table (1 and 4 admissible): that probe ran
on the analytic `ReferenceModels` graphs (21 and 72 edges), this one on the
frontend's, and the criterion changed from "same ownership string" to "both
`kTilePerBlock`". The old numbers are superseded, not reproduced.

## Self-correction

An earlier draft of this measurement reported a contradiction between a run
with no `TILEMEGA_WAIT_PROFILE` override (`polls=1476 relaxed=1512`) and the
`seq=1` bucket of a sweep (`polls=350 relaxed=386`) and held the sweep back.
There was no contradiction: the fixture is `seq = 4`, not 1, and the two runs
were built at different `kappa`. At matching `kappa = 1` and `seq = 4` the
sweep reproduces `1476 / 1512` exactly.
