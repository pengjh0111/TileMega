# BE-5 — role-granularity release: the litmus, and why §8.5 is unchanged

## Result first

| arm | cells | rounds | outcome |
|---|---|---|---|
| `roles` (the rule under test) | 9 | 450 | **450/450 pass** |
| `nofence` (negative control) | 9 | 450 | **fails every round of every cell** |
| `nobarrier` (negative control) | 9 | 450 | **passes every round of every cell** |

§5.3 requires *both* controls to fail before §8.5 may be rewritten. One does.
So **B-a is not met**, and under §8.3 the consequences are taken rather than
argued around: §8.5 is left exactly as it was, no TaskBody is specialized, no
harness barrier is converted to a named barrier, and the role path is recorded
as undelivered. `TaskRoles` (BE-1) stays declared and unused.

## The matrix

`run_litmus.py`, one fresh process per round, 50 rounds per cell:
grid ∈ {64, 128, 256} × elements ∈ {256, 1024, 4096} × three arms.
Tiles stay at and below the 4096 floats F-10 names, addresses are reused (two
slots rewritten every iteration), and the write is cooperative. Raw rounds are
under `raw/<arm>_g<grid>_e<elements>/r*.log`, tallied in `raw/litmus.tsv`.

The shape follows F-1: producing threads write a tile, each writer fences, the
producers converge, and one thread then publishes the flag a consumer polls.
What differs is the granularity of that convergence — a named barrier over the
producing role (`bar.sync 1, 64`) instead of `__syncthreads()` over the CTA,
because a CTA barrier is unreachable once the other role runs different code.

## Why the barrier control does not fail, and what that does not mean

Five constructions were tried before concluding, each fixing a real defect in
the previous one:

1. **Both roles in one CTA.** Every arm passed, including `nofence`. Two roles
   of one CTA share an L1, so no fence is needed for them to see each other —
   the test could not observe any ordering rule. Discarded.
2. **Cross-CTA publication** (CTA *b* produces, CTA *b*−1 consumes). Now the
   compliant arm failed too: the producer kept advancing and overwrote its
   tile while the consumer was reading it. That was the test racing itself.
3. **A two-slot ring with an acknowledgement**, so a payload is stable while
   its flag names it, and the same two addresses are still reused. Compliant
   passed, `nofence` failed — and `nobarrier` passed.
4. **The signalling thread owns one element** and the other writers share the
   rest. This is F-1's actual hazard — the thread that publishes finishes
   first — and it is realistic. Still no failure, at tiles from 256 up to
   262144 elements.
5. **A load-dependent store** in each writer, because a loop of independent
   stores retires faster than the flag propagates. Still no failure.

⚠️ **Inferred: the window is closed by latency, not by the absence of the
hazard.** Each writer's own device-scope `__threadfence()` pushes its stores
toward the L2, and the consumer cannot observe the flag until a global atomic
round trip has completed — which on this device is longer than the tail of the
producing role's writes in every configuration reached. F-3 is explicit about
how to read this: a control that passes is *not* evidence that the ordering it
removes is unnecessary. It is evidence that this construction on this device
cannot see it.

One consequence worth stating for the next round: an sm_90-class part, where
the producer and consumer roles are a warpgroup pair inside one CTA and the
publication is an `mbarrier` rather than a global flag, is a different
experiment — and the one that would actually exercise the rule.
`run_sm120.sh` carries the same matrix for a part with clusters, which is the
nearest thing this repository can run unattended.

## Files

| file | what it is |
|---|---|
| `litmus.cu` | the three arms, one kernel, selected by template parameter |
| `run_litmus.py` | the matrix and the per-round process launcher |
| `raw/litmus.tsv` | cell → rounds, passing, mismatching |
| `raw/<cell>/r*.log`, `r*.json` | every round's own output and manifest |
| `barriers.md` | the fifteen harness barriers, what each protects, and what role granularity would change |
