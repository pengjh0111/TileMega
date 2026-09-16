# R4 implementation design and corrected premises

## Prerequisite corrections

✅ Verified by source inspection: `TILEMEGA_EVENT_RED_PUBLISH` currently lowers
unused-result `atomicAdd` to a relaxed arrival, with ordering supplied by
`NotifyTask`'s fence. The R3 PTX audit is retained in `premise_audit/`. A SASS
`STRONG.GPU` suffix does not establish PTX release semantics.

✅ Verified by 1,800 new processes: `litmus_v3/scan` separates two sensitivity
tests. Cache visibility uses address reuse, no consumer fence and no writer
delay. Barrier sensitivity keeps the consumer fence and barrier and delays
nonpublisher warps on odd CTAs by a fixed 2,000,000 cycles in **all** arms.
Only the producer barrier is removed in the no-barrier arm. Both suites cover
grid 64/128/256 and tile 1024/4096. Each positive cell passes 50/50; each
suite-specific negative mismatches 50/50. No expected value was changed.
The original insensitive R4 rerun remains in `litmus_recheck/`.

✅ Verified by source inspection: with `TILEMEGA_BARRIER_V2`, the trailing
NotifyTask CTA barrier is already absent. Nonpublisher warps can already enter
the next wait. C2 therefore must specialize the publishing warp and distribute
independent fine/aggregate event publications within it; simply deleting the
already-absent barrier would not implement another mechanism.

✅ Verified by source inspection: at kappa 1, W>1 same-worker dependencies are
already encoded in `slot_local_deps` and checked against the executor's
per-thread `done_mask`. They do not poll global events. C3(a) implements the
requested shared-memory completion flags and measures their actual effect;
it must not claim to eliminate global polls that the host already removed.

The user's resume instruction authorizes correcting these premises locally
while preserving evidence and continuing independent items. No stop finding
above is interpreted as a global task cancellation.

## C2 ordering argument to test

⚠️ Inferred, pending implementation and its raw correctness matrix: all task
writers first converge at the release barrier. Thread 0 fences their writes,
then the publishing warp synchronizes before its event-writing lanes proceed.
Other warps may enter only the next slot's wait. The publishing warp completes
its publications before entering that wait, including when the next slot
retains an event dependency on this one at kappa >1. It cannot block on its
own unpublished event. The wait's CTA convergence prevents any warp entering
the next RunTask before publication and all waits finish. Event publication
touches global event rows, not TaskSmem. No prefetch is introduced.

## C3(a) completion flags and TaskSmem

⚠️ Inferred, pending implementation and tests: a separate shared completion
word and head, outside TaskSmem, can be written by thread 0 before the existing
release barrier. All warps read the next window state after that barrier.
Tasks without outgoing events need equivalent convergence for the shared
state. The next wait's barrier still precedes RunTask. This preserves the
union lifetime and does not require a shared-state barrier after publication
on the ordinary publishing path.

## C3(b) scope boundary

⚠️ Inferred, pending sm_120 execution: only the cluster-local fan-in arrival
may use cluster scope. The last local arrival must retain the GPU-scoped
forwarding publication for consumers in other clusters. One globally polled
event cannot be blindly changed to cluster scope. The existing DSMEM shard
path provides the local arrival site; `caps.cluster=false` must discard the
new site at compile time.

PTX syntax and scope semantics are taken from NVIDIA's
[PTX ISA 8.7 atomic instructions](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom).
No sm_120 performance or correctness is asserted from sm_89 compilation.

## B pricing and implementation expectations

✅ Verified: `../FENCE/raw/paired` supplies 25 rounds in every five-arm cell,
with all 200 full processes passing. Default-placement fence medians are
15.360 / 22.528 / 31.744 / 56.160 us in gqa2 s4/s128, mha4 s4/s128 order;
rotate gives 13.376 / 16.480 / 24.576 / 45.056 us. Fence represents only
21–33% of full-minus-neither at default placement and 6–7% at rotate.

⚠️ Inferred design consequence: C1's expected gain is a fraction of that
measured marginal fence delta. It cannot be assumed to meet the research gate
alone. C2 must expose actual publication overlap and independent event stores;
C3(a) must be priced as replacing existing register completion tracking, not
as removing nonexistent local global polls. Fence/notify ratios above one in
rotate are interactions between the waiting-on and waiting-off probe contexts;
they are preserved in FENCE's table rather than clipped or reclassified.
