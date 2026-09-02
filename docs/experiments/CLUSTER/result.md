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

## 7.5 Boundary: the megakernel is not launched as a cluster ❌ not done

`lib/Codegen/Codegen.cpp:324` still rejects every `sync_kind` other than
`"global"`, and `ModelHarness.cuh` still launches with `<<<grid, threads>>>`.
Wiring `sync_kind = "cluster"` through the generator would require the whole
harness to be *instantiated* on a cluster-capable `Arch` at build time; on this
box that instantiation trips the `static_assert` in §7.2 by construction, and
the resulting kernel could never be executed or measured here. Emitting a code
path that has never run once, into the generator that produces every measured
number in this repository, is worse than a documented gap — so the gap is
documented. `ClusterSync` is the primitive layer that path will call; what is
missing is the emitter change plus a `cudaLaunchKernelEx` variant of
`LaunchL1`/`LaunchL2`, and neither can be honestly validated before an sm_90+
machine is available.

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
