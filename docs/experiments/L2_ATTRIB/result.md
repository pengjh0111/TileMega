# Part 3 — where L2's time actually goes

Reproduce: build the four arms of the generated model and run
`raw/`'s protocol (25 interleaved rounds, arm order rotated each round so a
warm-up bias cannot land on one arm; medians of within-round values, 20 000-
replicate bootstrap CI). Raw rows are `raw/arms.tsv`, the reduction is
`raw/summary.txt`.

Evidence status: ✅ measured on an RTX 4090 (sm_89), BF16, structured
ownership, on 2026-09-06 in one session.

## The method: four arms that add one mechanism each

| arm | what it runs | build |
|---|---|---|
| `neither` | L2's stage loop, no events at all | `-DTILEMEGA_UNSAFE_NO_EVENT_WAIT -DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY` |
| `nowait` | + the notify side | `-DTILEMEGA_UNSAFE_NO_EVENT_WAIT` |
| `full` | + the wait side (the shipped L2) | — |
| `l1nosync` | L1's stage loop without its grid barrier | `-DTILEMEGA_UNSAFE_NO_GRID_SYNC` |

The two new probe macros are cost probes in the same sense as the existing
`TILEMEGA_UNSAFE_NO_GRID_SYNC`: their output is wrong by construction and they
exist only to price one mechanism. Neither stub leaves a fence behind, so the
fence is not charged to the arm that follows it.

`L2 = neither + notify + wait` and `L1 = l1nosync + barrier` hold to the last
digit in every cell, which is the check that the decomposition is complete.

## Two of the brief's five candidates are excluded by construction

`RunModel` sizes **one** grid as the *minimum* of both kernels' residency
limits (`ModelHarness.cuh`, "the resident bound is the minimum over them"),
precisely because L2 can need a register more than L1. So L1 and L2 always
launch the same number of CTAs at the same occupancy: "L1 uses a wider grid"
and "L2 has worse occupancy" cannot be the explanation of a gap between them
on this harness. They are ruled out before the first measurement.

## The measured decomposition

`kappa` is `TILEMEGA_EVENT_KAPPA`: **0** is one event per producer *stage* —
the default and what every shipped build uses — and **1** is one event per
producer *task*.

| model | seq | κ | L1 ms | L2 ms | ratio | gap ms | wait | notify | barrier | loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gqa2 | 4 | **0** | 0.400576 | 0.438272 | **1.094** | 0.036864 | +144.4% | +91.8% | −105.6% | −33.3% |
| gqa2 | 128 | **0** | 0.563200 | 0.593920 | **1.055** | 0.030752 | +173.8% | +95.5% | −126.5% | −40.0% |
| mha4 | 4 | **0** | 0.792768 | 0.866304 | **1.093** | 0.072704 | +145.2% | +84.3% | −94.4% | −32.3% |
| mha4 | 128 | **0** | 1.126400 | 1.195264 | **1.061** | 0.068384 | +161.7% | +85.4% | −109.3% | −37.4% |
| gqa2 | 4 | 1 | 0.400512 | 0.496544 | 1.240 | 0.095296 | +112.9% | +40.6% | −39.8% | −12.8% |
| gqa2 | 128 | 1 | 0.563360 | 1.012736 | 1.798 | 0.449536 | **+104.1%** | +7.5% | −8.7% | −3.0% |
| mha4 | 4 | 1 | 0.792640 | 1.000288 | 1.262 | 0.207712 | +110.5% | +33.5% | −33.1% | −11.0% |
| mha4 | 128 | 1 | 1.127616 | 2.042880 | 1.812 | 0.914944 | **+103.9%** | +7.4% | −8.4% | −3.0% |

Shares are of the L2−L1 gap and therefore exceed 100% where the barrier that
L2 does *not* pay is subtracted. The absolute components with their CIs are in
`raw/summary.txt`.

## Three answers

**1. It is the poll, and nothing else.** At κ = 1, seq = 128 the wait side is
**104.1% / 103.9%** of the whole gap. Notify is 7.4–7.5%, and L1's own barrier
gives back 8.4–8.7%. The dispatch and table-reading the brief lists as
candidates 1 and 3 are not a cost at all: **L2's stage loop is 12–28 µs
*faster* than L1's** in every one of the eight cells (−1.9% to −40% of the
gap), because a CTA that owns nothing in a stage skips its work.

**2. The 2.13× in the brief is a κ = 1 number, not the shipped build.** At the
default κ = 0 the ratio is **1.055 / 1.061** at seq = 128 — and it *falls* as
the sequence grows (1.094 → 1.055 and 1.093 → 1.061), rather than rising to
2.1×. The reason is visible in the components: at κ = 0 the wait is **flat in
seq** (gqa2 0.053248 ms at seq = 4, 0.053440 ms at seq = 128) because there is
one event per producer stage regardless of length, while at κ = 1 it grows
4.1–4.3× (0.107584 → 0.468032, 0.229504 → 0.950560). The growth the brief
attributes to "an unidentified L2-specific overhead that scales with seq" is
the per-task event count, which is exactly what κ selects.

**3. Even at κ = 0, events cost about twice a barrier.** wait + notify is
0.083 ms against the grid barrier's 0.039 ms on gqa2 (0.169 vs 0.075 on mha4).
That is the honest residual: a stage-granular event scheme is not free, it is
~2× the barrier it replaces, and the megakernel's remaining 5.5–6.1% is that
difference minus the 1.9–3.0% its stage loop wins back.

## What was fixed once the poll was named

The whole wait loop ran on `threadIdx.x == 0` — a serial walk over
edges × owned tasks × event groups with a spin on each, while 255 threads
idled. Every epoch is monotone (§8.2), so no poll depends on any other poll:
the loop nest is now walked by every thread for its arithmetic and the spin is
taken only on the iterations belonging to that thread, with the
`__syncthreads` that already ended the function making the union complete.
`TILEMEGA_SERIAL_POLL=1` keeps the previous shape compilable so the two can be
paired in one session rather than compared across two.

✅ Paired in one session, 25 rounds, order alternating, at κ = 1 and
seq = 128 — the configuration where the poll is the whole gap
(`raw/poll_parallel.tsv`):

| model | serial poll | parallel poll | delta | 95% CI | Wilcoxon p |
|---|---:|---:|---:|---|---|
| gqa2 | 1.191936 ms | 1.012640 ms | **−15.05%** | [−15.10, −15.03] | 1.29e−05 |
| mha4 | 2.397184 ms | 2.042816 ms | **−14.78%** | [−14.83, −14.70] | 1.31e−05 |

15% of the L2 path, from distributing a loop that had no reason to be serial.
It is not the 256× a thread count might suggest, and the reason is worth
recording: the polls are volatile global loads on the same few epoch words, so
spreading them across threads parallelizes the *walk* but not the memory system
they contend for. Everything below the walk — deduplicating the repeated
`(producer, group)` polls that the per-owned-task loop generates, or hoisting a
poll that has already been satisfied — is untouched and is the obvious next
step at κ > 0.

## What this means for the claim the megakernel makes

The narrative cannot be "fine-grained waiting is cheaper than a barrier". On
this hardware a grid barrier costs 0.039 ms (gqa2) / 0.075 ms (mha4) for the
whole model, and the event scheme that replaces it costs about twice that even
at its coarsest granularity. Making the events *finer* — one per task instead
of one per stage — makes it 8–13× more expensive again, which is the same
conclusion the κ ablation reached from the other side (F-55, F-59).

What L2 does buy is visible in the `loop` column and nowhere else: a CTA that
owns no task in a stage does not wait, so it reaches the stage where it does
own work sooner. That is worth 12–28 µs per model, consistently, and it is
smaller than the event machinery costs. ⚠️ On this pair of models, at these
sizes, the honest statement is that **the megakernel's advantage is launch
elimination and cross-operator overlap, not the dependency granularity** —
which is what the decision table in the brief calls the third row, reached with
an attribution rather than by exhaustion.
