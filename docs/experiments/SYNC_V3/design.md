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

⚠️ Inferred ordering argument, exercised by the separate raw correctness and retained-dependency matrices: all task
writers first converge at the release barrier. Thread 0 fences their writes,
then the publishing warp synchronizes before its event-writing lanes proceed.
Other warps may enter only the next slot's wait. The publishing warp completes
its publications before entering that wait, including when the next slot
retains an event dependency on this one at kappa >1. It cannot block on its
own unpublished event. The wait's CTA convergence prevents any warp entering
the next RunTask before publication and all waits finish. Event publication
touches global event rows, not TaskSmem. No prefetch is introduced.

## C3(a) completion flags and TaskSmem

⚠️ Inferred ordering argument for the implemented shared flags: a separate shared completion
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

## Composition and measurement decisions

The existing RED-plus-shards compile-time exclusion prevented the completed
protocol from reaching the cluster experiment. The exclusion is removed and
consumer RED targets use the number of nonempty shards when local fan-in is
active. A closed shard contributes exactly one global reduction per iteration;
the monotonically increasing target therefore counts closed shards, not all
original writers. The one-member/S=1 paths are unchanged. `sharded_red` tests
this composition on sm_89 before relying on the sm_120 cluster script.

`matrix_manifest.json` fixes seven configurations before measurement. All
70 configuration/placement/probe combinations rotate within each round and
share one measurement session. The required five rows are retained, with W=4
control/local rows added. Real-width seq=4 legacy has the required five rows.
Every configuration is evaluated on all four default-placement reference
cells against the unchanged registered ratio threshold. One configuration
must achieve at least three cells. The representative configuration maximizes
achieved cells, then minimizes the geometric mean of the four protocol/barrier
medians; fastest end-to-end time is reported separately. A winning window
configuration remains opt-in, not a new default. All configurations remain
visible. `research_rule.json` records this evaluation before the corrected
measurement session. Candidate-specific target positions additionally
measure all three W=1 configurations on each unchanged frozen Plan; the new
Chain2 Plan is reported separately and does not inherit the old chain floor.

The prompt's static MEMBAR-count prediction is not the mechanism's instruction
count. C1 changes the participants at the release site from 128 threads to
one; it need not remove the static site. The BARRIERS reports and neighboring
SASS predicates show static L2 MEMBAR.SC.GPU 2 -> 2, while the release moves
after CTA convergence and under the thread-0 condition. This correction is
reported explicitly; static instruction elimination is not claimed.

## Window no-wait coverage correction

The first partial C ablation exposed a source-level coverage error: at W>1,
NO_EVENT_WAIT removed the blocking wait but ProbeTaskDependencies retained
its event reads and CTA reduction. The partial session is retained separately
and is not used as final data. The repair removes those operations only in
unsafe no-wait builds. Complete safe W=2 SASS is byte-identical for both
models when compiled under the same source identifiers; the unsafe nowait
kernel loses one BAR.RED and one ATOMG event-poll site. All seven configurations
are remeasured in a new paired session. B and all safe correctness data are
unaffected. Independent Chain2 and frozen-candidate measurements continue
while unsafe window binaries are rebuilt; commit dependencies A/E/B/C remain
unchanged. See `window_probe_fix/` for commands and raw instruction evidence.
