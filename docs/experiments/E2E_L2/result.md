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

Re-measured on 2026-09-04 after Part 1 and Part 2 wired the derived `C` into
the generator (`raw/part3/`, `paired_default_kappa.txt`). 25 fresh processes
per model; each process times L1 and L2 itself, so the pairing is
within-round and the ratio is taken per round before it is aggregated
(`raw/part3/paired.py`: median of ratios, 20000-replicate bootstrap CI on
that median, two-sided Wilcoxon signed-rank on the paired differences).

| Model | l1 median | l2 median | l2/l1 (median of ratios) | bootstrap 95% CI | Wilcoxon | rounds L2 faster |
|---|---:|---:|---:|---|---|---:|
| 2-layer GQA | 1.083232 ms | 1.122304 ms | **1.036186×** | [1.035951, 1.036719] | W+=325 W−=0, p=1.30e−05 | 0/25 |
| 4-layer MHA | 2.158592 ms | 2.239488 ms | **1.037540×** | [1.036984, 1.037936] | W+=325 W−=0, p=1.30e−05 | 0/25 |

**L2 is still slower than L1, and by more than the 1.014×/1.016× this file
used to report.** ✅ That comparison is not admissible, and the reason is the
first thing this round found.

### The 1.014×/1.016× baseline does not reproduce, and nothing about the code changed it

Re-running the *previous* binary — `docs/experiments/E2E_GEN/generated_e2e`,
the 35-edge `kAll` build from before Part 2, byte-identical, 25 fresh
processes today (`raw/part3/paired_prev_binary.txt`):

| | l1 median | l2 median | l2/l1 | bootstrap 95% CI |
|---|---:|---:|---:|---|
| recorded 2026-09-03 | 1.095808 ms | 1.110992 ms | 1.0136× | not computed |
| same binary, 2026-09-04 | 1.082560 ms | 1.122272 ms | **1.036713×** | [1.035648, 1.036862] |

The same executable moved by 2.3 points. The absolute L1 median moved too
(0.998400 ms in the 07:41 sweep of the same day, 1.082560 ms here), and
`nvidia-smi` reports the SM clock idling at 210 MHz against a 3105 MHz
maximum, so the machine's clock/thermal state is not held fixed between
sessions. **Only within-session paired ratios are comparable here**, and the
old table is superseded rather than reproduced. Every number in this section
was collected in one session against one control.

### Attribution: it is not the dependency table

Four builds of the same model, differing only in the contents of
`kDependencies`/`kDependencyOffsets`, measured in the same session. The
control tables are produced by `raw/part3/retable.py`, which rewrites only
those two arrays; it is validated by the fact that its `reduced` mode
reproduces the generator's own previous 35-edge table exactly.

| gqa2 variant | edges | l2/l1 | bootstrap 95% CI |
|---|---:|---:|---|
| derived windows (20 `kAll` / 3 `kIdentity` / 15 `kWindow`) | 38 | 1.036186× | [1.035951, 1.036719] |
| every edge forced to `kAll` | 38 | 1.035734× | [1.035019, 1.036644] |
| `kAll` + full transitive reduction | 35 | 1.036036× | [1.034876, 1.036862] |
| previous binary (old harness) | 35 | 1.036713× | [1.035648, 1.036862] |

| mha4 variant | edges | l2/l1 | bootstrap 95% CI |
|---|---:|---:|---|
| derived windows (40 / 7 / 31) | 78 | 1.037540× | [1.036984, 1.037936] |
| every edge forced to `kAll` | 78 | 1.037583× | [1.036984, 1.037910] |
| `kAll` + full transitive reduction | 71 | 1.036451× | [1.036007, 1.036791] |

All four CIs overlap on each model. The exact table is worth nothing at the
default build, and the three extra edges that Part 2's narrowing costs the
transitive reduction (35 → 38, because a narrowed edge may be dropped when
implied but may not serve as an intermediate) cost nothing measurable either.

The reason is mechanical and was not visible before the profile existed:
`TILEMEGA_EVENT_KAPPA` defaults to **0**, one event per producer *stage*, and
at κ = 0 `WaitDependencies` takes a branch that polls exactly one event per
edge **regardless of `map`**. A window cannot narrow a wait set that is
already one event wide. The narrowing factor of the default build is
**1.000×** — which is why Part 3.2 had to re-measure κ rather than assume it.

### What the wait set is worth once κ > 0

Full numbers in `waitset.md` (device-side poll counts) and in
`docs/experiments/COARSEN/result.md` (the paired hardware sweep). The short
form, from `raw/part3/kappa_polls_summary.txt`:

| model | κ=1 | κ=2 | κ=4 | κ=8 | κ=16 | κ=32 |
|---|---:|---:|---:|---:|---:|---:|
| gqa2 polls vs. `kAll`, seq=4 | 1.0244× | 1.0161× | 1.0000× | 1.0000× | 1.0000× | 1.0000× |
| mha4 polls vs. `kAll`, seq=4 | 1.0222× | 1.0147× | 1.0000× | 1.0000× | 1.0000× | 1.0000× |

✅ measured. The wait set narrows by at most 2.4%, and by nothing at all for
κ ≥ 4. 89.9% of the poll mass sits on edges with a `kElementChunk` end, where
no per-CTA window is admissible at all (`waitset.md`); of the 5.9% that the
windows do reach, every fitted tile-row window has `count = scale = Tm = 128`,
which equals `gridDim.x` on this GPU, so one grid stride already covers every
CTA. That is a §2.3 **Place** property, not a weakness in `C`.

And the cost is larger than the benefit. At each κ, the windowed table is
slower than the `kAll` control with **disjoint** bootstrap CIs
(`raw/part3/paired_kappa.txt`); at κ = 32 the two tables poll *identical*
counts (276/276 at seq=4, 6028/6028 at seq=512), so the whole 0.48-point gap
there is window-*evaluation* arithmetic — the per-edge grid-stride union with
its div/mod — and nothing else.

### How the 18–36% was attributed

Each step was measured, not reasoned about. This history is unchanged; only
the endpoint it lands on is restated above.

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

### Why the residual gap is structural

The megakernel's stage loop is **sequential per CTA**: every CTA walks stages
`0 … stage_count-1` in order. L2 can therefore only ever *remove waits*; it
can never let a CTA run a later stage first. On a stage DAG that is
essentially a chain — which is what a transformer decoder layer lowers to —
there are almost no waits left to remove after transitive reduction, and what
remains is one epoch poll per incoming edge versus L1's one arrival counter
per stage. L2 pays a slightly larger constant for the same ordering.

This is now measured from both ends rather than argued. The exact dependency
table changes the ratio by less than its own confidence interval (above), and
the wait set it narrows carries 5.9% of the polls at best. A per-edge event
graph beats a barrier when consumers can start out of order; the ordering
information is now exact and it still does not, because nothing consumes it
out of order.

Against ETC's reported 6–9% upper bound for orchestration on comparable
models, TileMega's orchestration layer measures **−3.6% / −3.8%** here (it is
a loss, not a gain). The previously recorded −1.4% is from a session whose
absolute timings do not reproduce and is withdrawn as a comparison point, not
as a measurement.

## What `kIdentity` is worth

Re-measured against the production path — the derived isl `C`, and
`LiftedOp::ownership` cross-checked against `OwnershipOf`, the table the
TaskBodies themselves return from `Ownership` (`TaskBase.h`). The previous
probe answered the ownership half from a hand-written table keyed on operator
*name prefixes*; a name is not a declaration, and nothing kept the two in
step.

| | gqa2 | mha4 |
|---|---:|---:|
| derived edges | 42 | 86 |
| ranks differ (no identity possible) | 13 | 27 |
| `C ⊆ identity` | 9 | 19 |
| …and both ends `kTilePerBlock` (`kidentity_admissible`) | **7** | **15** |
| …both ends `kElementChunk` (not claimed) | 2 | 4 |
| lifted-vs-TaskBody ownership disagreements | **0** | **0** |

✅ `raw/identity.txt`, `tools/tilemega-identity-probe`. These supersede the
table this file used to carry (21 and 72 edges, 1 and 4 admissible): that
probe ran on the analytic `ReferenceModels` graphs and keyed ownership on
names, this one runs on the frontend's own graph and on the ABI. They are not
comparable and the old numbers are withdrawn rather than reproduced.

Three `kIdentity` edges reach the generated gqa2 table and seven reach mha4.
They do narrow — 4× each on the poll count — and they carry 0.03% of the poll
mass (`waitset.md`), so the earlier conclusion survives its own correction:
`kIdentity` is real now, and it is still not what stood between L2 and a win.

## What still blocks a tighter L2

Two things, both named rather than implied. The first of the two this file
used to list — "the generator cannot see `C`" — is closed by Part 1 and
Part 2 and is what this round measured.

1. **Element-chunk placement.** 89.9% of the poll mass is on edges with a
   `kElementChunk` end, and on tile-row edges the fitted window is exactly one
   grid stride wide. Both are §2.3 **Place** decisions. No amount of
   exactness in `C` moves either. A blocked placement
   (`task → task / ceil(count/grid)`) would preserve the tile-row windows; the
   same counting model gives 1.047× at κ=1 and 1.082× at κ=32
   (`raw/placement_whatif.txt`, ❌ inferred — a counting model, not a run),
   and even that is bounded by the 5.9% ceiling.
2. **The stage loop is sequential.** See above; this is the one that decides
   whether a per-edge event graph can win at all.

## Resources

Generated build links three device kernels (`tilemega_l1_kernel`,
`tilemega_l2_kernel`, `tilemega_stage_kernel`) against the same `TaskSmem`
union and the same `ModelSpec` tables; ptxas reports zero spills and 49,536 B
dynamic shared memory for the 2-layer GQA build — unchanged from the L1-only
baseline in `docs/experiments/E2E_GEN/result.md`, and unchanged again by the
window encoding (`raw/part3/gqa2_ptxas.txt` vs `gqa2_kall_ptxas.txt`).

## Conclusion

L2 event synthesis is correct (bitwise-identical output; 150/150 fresh
processes across κ ∈ {0,1,8} for the GQA model, 25/25 for MHA, and 600/600
across the 24 κ-sweep arms), §8.2's monotonic counters are exercised in the
cross-iteration case they exist for, and the dependency table now carries the
derived relation rather than a placeholder.

L2 is **not** faster than L1: 1.0362× / 1.0375×, measured against a
same-session control. Wiring the analysis layer in did not change that, and
the measurement says why in a way that is falsifiable rather than
speculative: forcing every derived edge back to `kAll` costs nothing outside
the confidence interval, the exact wait set is at most 2.4% narrower and only
for κ ≤ 2, and the arithmetic that exploits it costs more than it saves. The
remaining levers are both placement and scheduling decisions, not derivation
ones.
