# Part 7 — cluster-scoped synchronization (DSMEM)

**需要 sm_90+ 硬件，本机未运行。** This box is an RTX 4090 (sm_89); its
`TargetSpec::Probe()` reports `caps.cluster == false`, so no number in this
document is a measurement of cluster behaviour on hardware. What *is* verified
here is everything that does not need a Hopper: the capability gating, the
generated PTX, and the fact that the measurement script refuses to run on the
wrong machine.

Evidence markers: ✅ measured/executed here · ⚠️ stated by a source ·
❌ inferred.

## 7.1 What was implemented

`include/tilemega/Codegen/tasks/ClusterSync.cuh` — one `ClusterSync<Arch>`
template, five entry points:

| entry point | cluster path (`Caps<Arch>::kCluster`) | fallback |
| --- | --- | --- |
| `Rank()` / `Size()` | `this_cluster().block_rank()` / `.num_blocks()` | `0` / `1` |
| `Sync()` | `cluster.sync()` | `__syncthreads()` |
| `Peer(self, rank)` | `cluster.map_shared_rank(self, rank)` — DSMEM | `rank == 0 ? self : nullptr` |
| `Publish(self, epoch)` / `WaitPeer(self, peer, epoch)` | pairwise release/acquire on a peer's shared-memory epoch, `fence.acq_rel.cluster` on both sides | degenerates to a block fence |
| `StageBarrier(arrivals, epoch, iteration, clusters)` | `cluster.sync()`, then rank-0 CTAs meet on a global epoch | plain grid barrier |

`ClusterEvent` is one `unsigned long long` epoch living in the *producer's*
shared memory; a consumer in the same cluster reads it through
`map_shared_rank`, so a release never leaves the GPC. Counters are monotone
(skeleton §8.2) — they are never reset between iterations, which is what makes
a late waiter safe.

`StageBarrier` deliberately does **not** use `Publish`/`WaitPeer`: after
`cluster.sync()` the whole cluster is already ordered, and a DSMEM handshake on
top of it would be dead weight. The pairwise pair exists for the
producer→consumer edges that do not want a cluster-wide barrier.

## 7.2 The finding: `if constexpr` is not enough ✅

The task asked for `Caps<Arch>::kCluster` rather than `#ifdef`. The first
version was exactly that — one `if constexpr (kEnabled)` around every
cooperative-groups call — and it does not compile for sm_89:

```
error: namespace "cooperative_groups" has no member "this_cluster"
```

`if constexpr` discards a *statement*; it does not suppress name lookup for
non-dependent names. `cooperative_groups::this_cluster` does not depend on
`Arch`, and CUDA declares `cluster_group` only when the **device pass** targets
sm_90+ (`_CG_HAS_CLUSTER_GROUP`). So on sm_89 the discarded branch still has to
name something that was never declared.

The honest construction separates the two questions:

* **availability** — does this device pass have the API? `#if defined(_CG_HAS_CLUSTER_GROUP)`, and nothing else may test it;
* **policy** — should this target use clusters? `Caps<Arch>::kCluster`, at every use site;
* and a `static_assert(!kEnabled || kClusterGroupAvailable, …)` ties them
  together, so a target whose capability table claims clusters can never
  silently fall through to the single-CTA stubs. That assertion is the reason
  the single `#if` is not a second policy switch.

## 7.3 Cross-compilation and PTX evidence ✅

A probe TU (`ClusterSync<arch::CurrentArch>`, all five entry points used)
compiled with `nvcc 12.8`, `-Iinclude -Ithird_party/cutlass/include`:

| target | compiles | `mapa` | `barrier.cluster` | `fence.acq_rel.cluster` | `bar.sync` | registers |
| --- | --- | --- | --- | --- | --- | --- |
| sm_89 | ✅ | 0 | 0 | 0 | 6 | 10 |
| sm_90 | ✅ | 1 | 4 | 2 | 4 | 14 |
| sm_120 | ✅ | 1 | 4 | 2 | 4 | 14 |

`mapa` is the DSMEM address mapping emitted by `map_shared_rank`; it appears
exactly on the targets whose capability table sets `kCluster`, and the
sm_89 build contains no cluster instruction at all. This is the whole gating
claim, checked at the instruction level rather than asserted.

The release/acquire pair is spelled `fence.acq_rel.cluster` on both sides.
✅ `.release`, `.acquire`, `.acq_rel` and `.sc` all assemble at `.cluster`
scope on sm_90 — the pair is a choice, not a syntax constraint. What is a
constraint is the target: ptxas rejects *any* `.cluster`-scoped fence below
sm_90 (`Feature '.cluster scope' requires .target sm_90 or higher`), which is
why the fence has to sit behind the same capability gate as the API calls, and
why the sm_89 row above must read 0.

## 7.4 The measurement script

`docs/experiments/CLUSTER/run_on_h100.sh` — single file, self-contained: it
carries its own CUDA sources and needs only the repository headers, nvcc and a
GPU. Run it on an H100 with

```
bash docs/experiments/CLUSTER/run_on_h100.sh          # 200 fresh processes per size
RUNS=500 bash docs/experiments/CLUSTER/run_on_h100.sh
```

**Self-check.** It compiles `TargetSpec::Probe()` (the real one, from
`lib/Target/TargetSpec.cpp`) and hard-fails unless `caps.cluster == true`:

```
$ bash docs/experiments/CLUSTER/run_on_h100.sh          # on this sm_89 box
FAIL: TargetSpec::Probe() reports caps.cluster == false on sm_89.
      This experiment needs sm_90+; refusing to measure the
      single-CTA fallback and call it a cluster result.
$ echo $?
3
```
✅ executed here — this is the only thing about the script this machine can
prove, and it is the important one: no fallback run can ever be reported as a
cluster result. (An earlier version used `read … < <(probe)`, which silently
discards the probe's exit status; the check now tests the parsed flag itself.)

**What it measures.** Three kernels, launched with `cudaLaunchKernelEx` +
`cudaLaunchAttributeClusterDimension`:

1. `dsmem_visibility` — every CTA writes its own rank-tagged payload, then
   reads every peer's shared memory through `Peer()` and checks the tag.
2. `pairwise_ordering` — 64-word payload published behind `Publish`, read by
   the peer after `WaitPeer`; catches a release that is not actually ordered
   before the epoch store.
3. `two_level_barrier` — 16 stages of `StageBarrier`; a CTA that observes
   `counters[stage] != gridDim.x` after the barrier records a mismatch.

Each is run in **fresh processes** — `RUNS` (default 200) per cluster size, so
≥200 per size, ≥600 total — because a race that is invisible within one process
is often visible across launches. Cluster size sweep {2, 4, 8}, sizes above
`res.max_cluster_size` are skipped and marked as such rather than dropped.
Outputs: `raw/target.txt`, `raw/build.txt`, `raw/runs.tsv` (one row per run),
`raw/summary.tsv` (pass/total per size). Exit status is non-zero if any run
mismatched.

Both `cluster_test.cu` and the header cross-compile for sm_90 and sm_120 ✅ —
so the script's failure mode on a real H100 will be a *result*, not a build
error.

## 7.5 The megakernel now launches as a cluster ✅ built, ❌ never executed

The gap §7.5 used to document is closed in code. Three changes, each gated so a
non-cluster target emits exactly the bytes it emitted before:

* **Generator.** `lib/Codegen/Codegen.cpp` accepts `sync_kind = "cluster"` and
  a `PlacementOp` whose `cluster` exceeds 1, and emits
  `#define TILEMEGA_GENERATED_CLUSTER_DIM <n>`. A cluster is a property of the
  *launch*, so the contract is all-or-nothing: every placement must name the
  same width, and the sync kind must agree with it in both directions. A CG
  with cluster couplings and `cluster = 1` placements, or the reverse, is
  rejected rather than resolved to whichever side the generator happened to
  read first. ✅ `frontend_import_test` flips the imported 179-task CG one side
  at a time and asserts the rejection, then both sides and asserts the macro.
* **Barrier.** With `TILEMEGA_GENERATED_CLUSTER_DIM > 1`, `GridBarrier` becomes
  `ClusterSync<CurrentArch>::StageBarrier`, so the cluster closes over itself in
  hardware and only rank 0 of each cluster pays the global round trip: the
  arrival count drops from `gridDim.x` to `gridDim.x / dim`.
* **Launch.** `LaunchPersistent` replaces both `<<<>>>` sites with
  `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension` when the macro is
  set. `RunModel` trims the resident grid down to a whole multiple of the
  cluster width (a partial cluster is a driver error, not a remainder) and
  refuses to run at all when the device's own `res.max_cluster_size` is smaller
  than the compiled width.

**The gate is checked, not asserted.** ✅ On sm_89, `-DTILEMEGA_GENERATED_CLUSTER_DIM=2`
does not build:

```
ModelHarness.cuh(54): error: static assertion failed with "a cluster stage
barrier needs a cluster-capable target; a cluster-shaped kernel must never fall
back to the flat grid barrier and keep reporting itself as a cluster result"
```

`Caps<CurrentArch>` is only meaningful in the device pass, and this repository
allows exactly one `__CUDA_ARCH__` site, so `ArchDispatch.h` now exports
`arch::kDevicePass` and defines `CurrentArch = void` for the host pass — the
primary `Caps` template, every capability off — which keeps every `__device__`
body parseable in both passes without a second architecture switch.

✅ Cross-compiled here, `barrier.cluster` / `UCGABAR` counted in the emitted
code of the *whole megakernel*, not of a probe:

| target | dim 1 | dim 2 | dim 8 |
| --- | --- | --- | --- |
| sm_90 | 0 UCGABAR, 0 `barrier.cluster` | 8 / 4 | 8 / 4 |
| sm_120 | 0 / 0 | 8 / 4 | 8 / 4 |
| sm_89 | 0 / 0 | refused at compile time | refused at compile time |

✅ The default (`dim 1`) build is SASS byte-identical to the pre-cluster
binary — `cuobjdump -sass` diffed on `generated_e2e.cu`, no instruction and no
constant-bank offset moved. Every measured number elsewhere in this repository
therefore still describes the same kernel.

**What is still not known.** No cluster barrier in this repository has ever
retired an instruction. `run_on_h100.sh` gained a megakernel arm that builds
`generated_e2e.cu` at dim ∈ {1, 2, 4, 8}, checks each binary's SASS for
`UCGABAR` (0 required at dim 1, non-zero required above it — a "cluster" arm
that quietly contains no cluster barrier exits 4), and runs `MEGA_RUNS`
(default 50) fresh processes per dim reporting pass rate and median l1/l2.
Until that runs, the honest claim is *built and gated*, not *works*.

⚠️ One regression was found and fixed while re-running the self-check here: the
script's `TargetSpec::Probe()` probe stopped linking when P4.1 gave `TargetSpec`
a JSON calibration file to read. The probe now compiles `lib/Support/Json.cpp`
alongside it. A self-check that no longer builds is a self-check that no longer
checks, and this one is the single thing standing between a fallback run and a
"cluster result".

## 7.6 What a reviewer should distrust

* No cluster instruction in this document has ever *executed*. The PTX table
  says what the compiler emitted, not what the hardware did.
* `Caps<Sm90>::kMaxClusterSize = 8` is ⚠️ stated (from the CUDA programming
  guide's non-portable cluster size); the script re-reads the runtime's own
  `res.max_cluster_size` and skips sizes above it rather than trusting the
  table.
* The fallback paths (`Peer` returning `nullptr` for a non-zero rank, `Size()`
  returning 1) are exercised only by compilation here; on sm_89 there is no
  cluster to make them wrong.
* §7.5's megakernel arm is **built and gated, never executed**. The SASS counts
  prove the barrier is in the binary; nothing here proves it is correct at
  runtime, and `StageBarrier`'s two-level arrival protocol is exactly the kind
  of code that passes a compiler and races on hardware.
* §7.7's capture ratios are an upper bound on *traffic*, computed from the
  analytic model at S=512 — not from the fixture the harness measures, whose
  single M tile collapses several of these task spaces to one task.

## 7.7 Labelling: how much traffic could a cluster actually hold? ✅ analytic

`ClusterLabeling` (`lib/Solver/ClusterLabeling.cpp`) answers §4.3's Label
question — which task spaces share a cluster — under the three hard constraints
(size, temporal reach, shared memory) with §4.3's own objective
`w(A, B) = Volume × Frequency`. The weights are not estimated: they are
`metrics.volume × metrics.count` off the derived `CouplingEdge`s, so the graph
being partitioned is the one `CouplingDerivation` produces. `cluster_probe.cpp`
links barvinok and runs it; `tools/tilemega-solve` does not, because it works
from the generated stage tables and those carry no volume or count.

The heuristic is heavy-edge agglomeration and is labelled as one — maximum
weight k-way partitioning is NP-hard — so the report's headline is
`internal_weight / total_weight`, the fraction of coupling traffic the plan
keeps inside a GPC, which makes the gap to the unattainable optimum visible
instead of assumed. Raw output: `raw/labeling.txt`.

| model | nodes | edges | total weight (elements) | size 8, reach 4 | size 8, reach ∞ |
| --- | --- | --- | --- | --- | --- |
| gqa2 | 36 | 44 | 5.83e9 | **capture 0.985**, 11 clusters, largest 5 | 0.994, 6 clusters, largest 8 |
| mha4 | 64 | 72 | 1.27e10 | **capture 0.971**, 20 clusters, largest 4 | 0.994, 12 clusters, largest 8 |

Three things the table says that are worth more than the headline:

* **Reach, not size, is what binds.** At reach 1 (only consecutive stages may
  share a cluster) capture is 0.14 / 0.19 whatever the size cap; at reach 4 it
  jumps to 0.985 / 0.971 and the size cap takes over. The heavy edges of these
  models span three to four operators (RMSNorm → QKV → RoPE → attention), so a
  cluster that can only hold neighbours captures almost nothing. Every plan
  reports which constraints refused a merge, and more than one usually did.
* **The greedy is visibly a greedy.** mha4 at size 4 produces *fewer* clusters
  at reach 4 (20) than at reach ∞ (24) for identical capture: an unbounded
  reach lets it spend its size budget on heavy distant pairs early, which then
  blocks merges it would otherwise have made. This non-monotonicity is a
  property of agglomeration, and it is reported rather than smoothed.
* ⚠️ The shared-memory budget is `max_smem_per_sm × max_cluster_size` — the
  distributed window at one CTA per SM — against the harness's measured 10496 B
  `TaskSmem`. It only binds at size 16, which is above any real target's
  `kMaxClusterSize`. On this sm_89 box `max_cluster_size` is 1, so the sweep
  reads the *capability table's* sizes rather than the probe's; the budget row
  is therefore ⚠️ stated, not a measurement of a Hopper.

**The conclusion, and it agrees with §4.5.** Reach is not a free parameter. A
cluster keeps traffic on-GPC only if the producer's tile is still in shared
memory when the consumer runs, and in the stage-serial megakernel shared memory
is reused by the next stage: a value produced in stage *i* has been written to
global before stage *i+1* starts. **Reach 1 is therefore the only reach the
current TaskBody ABI can implement**, and at reach 1 the capture is **0.136
(gqa2) / 0.187 (mha4)** — the 0.97–0.99 rows require holding four to five
operators' outputs resident across stage boundaries, which no code in this
repository does. This is the skeleton's own §4.5 claim ("簇的粒度匹配「算子内
跨 CTA 归约」，不匹配「算子间数据流」") arriving as a number rather than as an
assertion: inter-operator clustering is worth a sixth of the traffic, and the
rest of it is behind an smem-residency ABI that does not exist.

Capture is in any case an upper bound on *traffic*, not a speedup. The P4.6
speed-of-light arm bounds the other side of the trade: see
`docs/experiments/COARSEN/result.md` for what deleting the grid barrier outright
is worth on this hardware, which is the ceiling on what any synchronization
change — clusters included — can ever return.
