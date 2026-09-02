# L2 fine-grained events vs. L1 global barrier (P3.5 acceptance)

Evidence labels: ✅ executed/observed; ⚠️ scope limitation; ❌ conjecture.

`run.sh` here does not re-run the frontend/codegen pipeline itself; it reads
the L2 numbers off the same generated binaries produced by
`docs/experiments/E2E_GEN/run.sh` (2-layer GQA) and
`docs/experiments/P3_GENERALIZATION/run.sh` (4-layer MHA), both of which now
emit an `l2_ms`/`l2_over_l1` line and an `l2` hash alongside `l05`/`l1`. Raw
logs are copied into `raw/` under this directory; the source-of-truth raw data
for the GQA model (including its 50-fresh-process logs) stays in
`E2E_GEN/raw/` where the pipeline that produced it lives. The MHA pipeline
runs its binary only once, so `run.sh` does that model's fresh-process sweep
itself (25 runs) and writes it under `raw/mha4_fresh_processes/`.


## What L2 is here

`CouplingGraphToCUDA::Lower` collects the producer/consumer stage pairs of
every coupling whose producer stage precedes its consumer, transitively
reduces that DAG, and emits one `StageDependency` per surviving pair.
`tilemega_l2_kernel` waits on those specific producer-stage event counters
(`WaitDependencies` in `ModelHarness.cuh`) instead of one grid-wide barrier
per stage (`GridBarrier`, which L1 uses unconditionally). Both paths run
inside the same compiled binary and are selected by the harness at the top
level, so L1 and L2 share every TaskBody, every table, and every numeric code
path — only the synchronization primitive differs.

Every dependency still uses `StageDependency::Map::kAll`: a consumer CTA
waits for the producer stage's active CTAs to arrive, not just the producer
CTAs the coupling `C` actually identifies. `kAll` is the I2-safe relaxation
(`C' ⊇ C`). What changed this round is *why* `kIdentity` is not emitted —
see "The `kIdentity` ceiling" below, which measures it rather than assuming
it.

## Correctness

| Model | L2 vs L1 | Fresh processes |
|---|---|---|
| 2-layer GQA (179 task / 222 coupling / 30 stage) | 0 mismatch, max_abs `0`, hash `5245714bc5d3ab4d` for l05/l1/l2 | ✅ 50/50, 0 hang, 0 error |
| 4-layer MHA (355 task / 444 coupling / 60 stage / 11 guard, `kv_heads == heads`) | 0 mismatch, max_abs `0`, hash `fd15fa2e89cdb915` for l05/l1/l2 | ✅ 25/25, 0 hang, 0 error |

L2 is bitwise identical to L1 on both models, and both hashes are unchanged
from before the ISL migration.

### §8.2 monotonic counters, actually exercised

`needed = num_triggers × iteration_num` (`StageArrivalTarget`) was previously
implemented but never run in the situation it exists for. The harness now
runs L2 a second time on the **same event memory**, resetting only the
buffers (`ResetBuffersOnly`) and passing `iteration = 1`. If the target were
fixed and the counters cleared between iterations, a CTA still finishing
iteration *i* could be counted as an early arrival for iteration *i+1*;
because the target scales with the iteration instead, that cannot happen.

Reported as `E2E_ITER l2_iter1_vs_iter0_mismatch=…`: 0 on every one of the
50 GQA and 25 MHA fresh processes. ✅

## Performance

Medians over fresh processes — 50 runs (2-layer GQA, from `E2E_GEN`), 25 runs
(4-layer MHA, from this directory's `run.sh`):

| Model | l1 median | l2 median | l2/l1 | previously |
|---|---:|---:|---:|---:|
| 2-layer GQA | 1.095808 ms | 1.110992 ms | **1.0136×** | 1.182× |
| 4-layer MHA | 2.183296 ms | 2.215936 ms | **1.0155×** | 1.355× |

(`raw/gqa2_l2_timing_median.txt`, `raw/mha4_timing_median.txt`. Repeat sweeps
of the MHA model land between 1.013× and 1.016×, so the ratio is stable to
about ±0.2 points, well inside "still slower".)

**L2 is still slower than L1.** The gap closed from 18–36% to 1.4%, but it
did not change sign, and the honest reading is that it is not going to under
this execution model. The attribution below is what the measurements
actually support.

### How the 18–36% was attributed

Each step was measured, not reasoned about:

1. **Per-CTA event fan-out** (one event per producer CTA rather than per
   stage) — no improvement. Rejected.
2. **Consumers polling `arrivals` directly** instead of a published `epoch` —
   no improvement; that is read-write sharing of one line across the grid.
   Rejected. (Both are recorded in `WaitDependencies`' comment so they are
   not retried.)
3. **Waits disabled entirely**: L2 fell to 0.99× L1 with 5376 mismatches.
   That is the load-bearing measurement: the compute paths are identical and
   the whole gap is synchronization, not code quality.
4. **Backoff sweep** (64/256/1024 iterations): ratio stable at ~1.187. Not
   the spin policy.
5. **The dependency-table scan**: `WaitDependencies` read all 55 dependencies
   from global memory, per stage, per CTA, in thread 0. Sorting the table by
   consumer and emitting `kDependencyOffsets[stage_count+1]` so a consumer
   reads only its own slice took the ratio from **1.19× to 1.021×**.
6. **Transitive reduction** of the stage DAG: 55 → 31 edges (GQA) and 63 (MHA,
   from a larger raw set). Sound because a stage's epoch is published only
   after every active CTA of that stage has cleared its own waits, so
   `p → q → c` already implies `p` for `c`; the happens-before edge survives
   through `q` and I2 is untouched. Worth ~0.5%.
7. **Skipping waits and fences for CTAs that own nothing in a stage**: worth
   ~0.3%. Sound because every dispatched TaskBody guards on exactly the bound
   `ActiveBlocks` reports — which is now the TaskBody's own declaration
   (`TaskOwnership`) rather than a switch restating each guard.

### Why the residual 1.4% is structural

The megakernel's stage loop is **sequential per CTA**: every CTA walks stages
`0 … stage_count-1` in order. L2 can therefore only ever *remove waits*; it
can never let a CTA run a later stage first. On a stage DAG that is
essentially a chain — which is what a transformer decoder layer lowers to —
there are almost no waits left to remove after transitive reduction, and what
remains is one epoch poll per incoming edge versus L1's one arrival counter
per stage. L2 pays a slightly larger constant for the same ordering.

A per-edge event graph beats a barrier when consumers can start out of order.
Getting there needs either out-of-order stage execution (a task queue rather
than a stage loop) or `kIdentity` — and `kIdentity`'s ceiling is measured
below and is zero on these models.

## The `kIdentity` ceiling

`docs/experiments/E2E_L2/identity_probe.cpp` asks the migrated derivation the
question directly, using `isl_map_is_subset` — an operator that did not exist
before the ISL migration, so this could not have been measured last round.
An edge admits `kIdentity` only if consumer CTA `b` depends on producer CTA
`b` alone, which needs two things: `C ⊆ identity`, and the two stages sharing
a CTA→task map.

| | decoder layer | 4-layer MHA |
|---|---:|---:|
| derived edges | 21 | 72 |
| ranks differ (no identity possible) | 10 | 36 |
| `C ⊆ identity` | **7** | **20** |
| …and both ends share an ownership kind (`kidentity_admissible`) | **1** | **4** |
| …and not already implied by a longer path | **0** | **0** |

Every row but the last is printed by the probe itself
(`SUMMARY … identity_candidates=… kidentity_admissible=…`); the last row is
read off the edge list, since transitive reduction happens in the generator
on stage pairs rather than on operator edges.

The seven identity couplings in the decoder layer are `wq→rope_q`,
`wk→rope_k`, `wo→add1`, `wgate→silu`, `wup→silu`, `wdown→add2` and
`add1→add2`. Six of them cross an ownership kind: the producer is a GEMM
(`kTilePerBlock` — CTA `b` owns N-tile `b`) and the consumer is RoPE or
elementwise (`kElementChunk` — CTA `b` owns a grid-stride slice of a flat
element range, a function of `gridDim`, not of the task space). The same
`blockIdx.x` does not name the same data on the two sides, so an identity
*task* coupling is not an identity *CTA* coupling. Only `add1→add2` has both
ends in `kElementChunk` — and that edge is transitively implied by
`add1→rmsnorm2→wgate/wup→silu→wdown→add2`, so the reduction already dropped
it.

So: even with the ABI entry in place and even if the generator could see the
derived relations, `kIdentity` would eliminate **zero** waits on either
accepted model. The hypothesis that the TaskBody ABI was what stood between
L2 and a win is now falsified, which is more useful than the ABI work would
have been on its own.

## What still blocks a tighter L2

Two things, both named rather than implied:

1. **The generator cannot see `C`.** `lib/Frontend/Frontend.cpp` never calls
   `CouplingDerivation`; it emits `fixedRelation()` placeholders
   (`{ [0] -> [0] }`, `wait/fanout/volume/count = 1`, `tier = 0`). The
   verified derivation drives the analysis experiments, not code generation.
   Recorded in skeleton §1.5.1.
2. **The stage loop is sequential.** See above; this is the one that decides
   whether a per-edge event graph can win at all.

## Resources

Generated build links three device kernels (`tilemega_l1_kernel`,
`tilemega_l2_kernel`, `tilemega_stage_kernel`) against the same `TaskSmem`
union and the same `ModelSpec` tables; ptxas reports zero spills and 49,536 B
dynamic shared memory for the 2-layer GQA build — unchanged from the L1-only
baseline in `docs/experiments/E2E_GEN/result.md`. Adding the ownership entry
and the offsets table did not change L1's or L0.5's resource footprint.

## Conclusion

L2 event synthesis is correct (bitwise-identical output, 50/50 and 25/25
fresh processes on two structurally different models), §8.2's monotonic
counters are now exercised in the cross-iteration case they exist for, and
the TaskBody ABI carries the CTA→task ownership entry it was missing.

L2 is **not** faster than L1: 1.014× / 1.016×, down from 1.182× and
1.355×. Every step of that improvement is attributed to a specific measured
cause, and the residual gap has a structural explanation rather than a
to-do. The one remaining mechanism that was expected to close it —
`kIdentity` — is measured here to be worth zero waits on these models. This
is the first meaningful performance number this project has produced, and it
says the per-edge event graph does not pay for itself under a sequential
stage loop.
