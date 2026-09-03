# P4.5 — chain DP over the implementation space

`tools/tilemega-solve.cpp` drives `lib/Solver/ChainDP.cpp`; reproduce with
`bash docs/experiments/SOLVER/run.sh` (no GPU) and
`bash docs/experiments/SOLVER/run_per_operator.sh` (GPU; `RUNS=400` was used
here, which is 3200 fresh processes).

The recurrence is §4.1's:

```
DP[i][s] = min_{s'} { DP[i-1][s'] + Cost_i(s) + Interface(s', s) }
```

with the state `s` = (tile m,n,k; stages; split factor) and a residency level
carried outside the chain, because §4.3's `ctas_per_sm` is a property of the
whole kernel: the shared-memory union and the register maximum are taken over
every operator's choice, so residency cannot be decided one layer at a time.
The solver therefore enumerates residency levels, pins the level, admits only
the candidates that reach it, and keeps the best chain over all levels. Three
levels are admissible on sm_89 for these candidate sets.

## Acceptance (a) — the uniform-`g` DP solution against the oracle

The DP picks `16x64x16s2k16` on both models, at `ctas_per_sm = 2`.

| model | candidates | measured rank (screen, 1077) | top-3% band | finals rank | finals l05 | finals best |
|---|---|---|---|---|---|---|
| gqa2 | 1077 | 14 | 32 | **2 / 15** | 0.183296 ms | 0.181248 ms (`16x64x32s3k16`, +1.13%) |
| mha4 | 1077 | 8 | 32 | **1 / 13** | 0.324608 ms | itself |

Both ranks are inside the measured top 3%, so **(a) PASSes on both models**.
The screen ranks come from `ORACLE/raw/screen_*.tsv` (single process per
configuration) and the finals ranks from `ORACLE/raw/final_*.tsv`, which is the
25-fresh-process protocol §0.2 calls for; the finals ranking is the stronger
statement and is the one to quote.

Predicted vs measured is *not* the criterion (§0.3), and it is not good: the
model predicts 0.0939503 ms for a configuration that measures 0.165856 ms on
gqa2. What the model has to get right is the ordering near the top, and it
does.

## Acceptance (b) — what per-operator `g` is actually worth

Four arms are compiled into real megakernels and measured end to end:

| arm | what it is |
|---|---|
| `uniform` | the DP's `kUniform` answer: one configuration for every GEMM |
| `split_only` | `kPerOperatorSplit`: one tile shape, per-operator split factor |
| `per_op` | `kPerOperator`: per-operator tile shape *and* split factor |
| `best_uniform` | the oracle's own 25-fresh-process best uniform configuration |

Running per-operator tile shapes needed a real codegen change, not a scripting
trick: `GemmStageTaskBody` now compiles up to four `GemmVariant`s and the host
tags each `GemmInvocation` with the variant that runs it. Two facts fell out of
the implementation and are worth recording.

* `Mainloop::Params` is a member type of the tile-parameterised `CollectiveMma`,
  so it is a **distinct nominal type per variant** even though its four fields
  are identical. `GemmInvocation` therefore carries a variant-independent
  `GemmMainloopOperands` POD built directly on the host. The epilogue collective,
  by contrast, does not depend on the tile at all and is shared. ✅ verified by
  the `static_assert`s in `GemmStageTaskBody.h`.
* The shared storage is a **union over variants**, which is §4.3's globality made
  concrete: the gqa2 per-operator plan mixes `16x64x16s2` with `16x32x16s2`, and
  the kernel's smem stays at 10496 B — variant 0's, the larger. A narrower tile
  on one operator buys that operator nothing in occupancy. ✅ verified
  (`E2E_RESOURCE smem=10496 ctas_per_sm=2 grid=256` on the `per_op` arm).

A single-variant build takes an `#if`-guarded direct call rather than the
variant `switch`, so the three single-variant arms are byte for byte what they
were before variants existed and the multi-variant arm is the only one paying
for the feature.

### The answer

400 interleaved rounds per model, one fresh process per arm per round, paired
within round. Percentages are the median within-round ratio; the bracket is a
95% bootstrap over rounds and `p` a two-sided Wilcoxon signed-rank.

| model | contrast | median | 95% CI | p |
|---|---|---|---|---|
| gqa2 | `split_only` vs DP `uniform` | **−0.512%** | [−0.610, −0.019] | 6.8e−3 |
| gqa2 | `per_op` vs DP `uniform` | **−1.214%** | [−1.371, −1.100] | 1.4e−14 |
| gqa2 | `per_op` vs `split_only` | **−1.128%** | [−1.212, −0.617] | 7.2e−10 |
| gqa2 | `per_op` vs oracle best uniform | **−2.820%** | [−2.976, −2.410] | 4.5e−35 |
| mha4 | `split_only` vs DP `uniform` | **−0.725%** | [−0.928, −0.641] | 1.4e−24 |
| mha4 | `per_op` vs DP `uniform` | **−0.769%** | [−0.962, −0.645] | 1.4e−22 |
| mha4 | `per_op` vs `split_only` | +0.000% | [−0.091, +0.045] | 0.75 (null control) |
| mha4 | `uniform` vs oracle best uniform | +0.000% | [−0.060, +0.222] | 0.63 (null control) |

The two null controls are the calibration: on mha4 the DP's split-only and
per-operator plans are the same plan, and its uniform plan is the oracle's own
best uniform, so those two rows measure nothing but the protocol. Both come out
at a median of exactly 0.000% with a CI half-width of ~0.15%. Anything outside
that band is a real effect of the plan; anything inside it is not a claim.

**Per-operator `g` is worth −1.21% on gqa2 and −0.77% on mha4** against the
DP's own best uniform, and −2.82% on gqa2 against the best uniform anybody
measured. ✅ verified, 400 fresh processes per arm.

The decomposition is the interesting part, and it differs between the models:

* On **mha4** the whole gain is the **split factor**. The per-operator tile
  shape adds nothing, because the DP did not ask for one — its per-operator
  answer *is* its split-only answer, which is why that row is a null control
  rather than a measurement of zero.
* On **gqa2** the split factor is worth −0.51% and the per-operator **tile
  shape** is worth a further −1.13% on top of it. That is the part of the plan
  the megakernel could not express before this round: two compiled variants,
  `16x64x16s2` and `16x32x16s2`, in one kernel.

So per-operator granularity does have headroom, but it is ~1% on these two
models, not the 4.9×–12.1× per-GEMM spread of §0.3. That spread is the range
over *operators* of how much each one gains from its own optimum; almost all of
it is already collected by the uniform choice, because the operators that gain
most are also the ones that dominate the total and therefore drive the uniform
pick. What is left for per-operator `g` is the small operators' residue. This
is a negative-ish result against the §0.3 expectation and it is recorded as
measured, not adjusted.

An honest caveat about absolute numbers: gqa2's `uniform` arm medians 0.172032
ms in the 25-round session and 0.169856 ms in the 400-round session — a 1.3%
session-to-session shift, larger than most of the effects in the table above.
That is exactly §0.2's drift, and it is why every number here is a within-round
paired contrast and none of them is a difference of two medians. The 25-round
run is kept alongside as `raw/per_operator_n25.*`; at n = 25 the gqa2
per-operator CI still straddled zero ([−1.786, +0.409]) and no conclusion was
drawn from it.

### The protocol had to be fixed before the numbers meant anything

The first measurement ran each arm as a block of 25 fresh processes. It
reported `per_op` at −1.211% against `uniform` on gqa2 with rank-sum
p = 3.37e-3, and `split_only` at −0.639% against `uniform` on mha4 with
p = 1.76e-4. The second of those is impossible: on mha4 the DP's split-only and
per-operator plans are the **same plan**, and
`diff plan_mha4_split_only.h plan_mha4_per_op.h` differs only in a generated
comment. Two identical binaries had "significantly" separated by 0.64%.

Within-session drift, not sampling noise, was the dominant term, and a
block-per-arm layout aliases drift onto arm identity. The experiment is
therefore run **interleaved** — one fresh process per arm per round, 400
rounds, the starting arm rotated each round — and reported **paired**: the statistic is
the within-round ratio, so drift shared by a round cancels. Arms that compile
to an identical plan are detected automatically and flagged as null controls;
they are the protocol's own floor.

## §4.2 — what the DP does with a fork

Neither reference model is a chain in the dataflow sense. gqa2's Q/K/V is a
three-way fork off one RMSNorm (stages 1–3 of `generated_e2e.cu`) and its FFN
gate/up is a two-way fork (stages 10–11); mha4 has the same shape four times.

The DP linearises a fork **in stage order** and charges the interface between
the branches like any other interface. That is not an approximation of a
branching recurrence — it is the right recurrence for this target. In an L1
megakernel the stages are separated by grid barriers and execute one after the
other, so the schedule is already a total order; the chain the DP walks *is*
that order, not the DAG. A branching DP would model a concurrency the runtime
does not have.

What the fork does change is the interface term, and that is where it is
handled: `Interface(s', s)` between two branches of a fork contains no
non-GEMM stage work (the branches are adjacent stages, so `BetweenNs` is
exactly zero) and only the reuse term `CarryNs`, which is driven by the shared
live footprint rather than by either branch's tile shape. On both reference
models that term evaluates to zero, for the reason in the next section.

## §4.3 — globality, and why the interface term vanishes here

`resident_tiles_per_SM` is taken from the global maximum over the shared-memory
union and the register maximum, exactly as §4.3 requires, and it is *not* a
per-layer quantity: the solver enumerates the achievable residency levels
(three are admissible on sm_89 for this candidate set), pins one, admits only
the candidates that reach it, solves the chain, and keeps the best over levels.
Solving without the pin — letting each layer pick its own residency — would
score plans at an occupancy the hardware would never grant.

`ChainDpStats` reports two invariants instead of asserting them:

| invariant | gqa2 | mha4 | meaning |
|---|---|---|---|
| `decomposition_error_ns` | 1.46e-11 ns | 5.82e-11 ns | the sum of per-layer costs plus interfaces reproduces the whole-model evaluation to floating-point noise: no barrier double-charged, no stage charged to nobody ✅ |
| `interface_spread_ns` | 0 ns | 0 ns | `Interface(s', s)` does not vary with the pair `(s', s)` ✅ |

The second one is the interesting one, and it is a *measured property of these
two models*, not a modelling assumption. `CarryNs` charges the producer's whole
output at the DRAM-minus-L2 differential, scaled by the miss probability; both
models' live footprints sit below the calibrated L2 knee, so the miss
probability is zero and the term vanishes. The chain therefore **separates**:
every layer's optimum is independent, and the general transition loop and the
separable shortcut reach the same objective.

That is reported rather than exploited. The general loop stays the default and
the shortcut is checked against it, because a model with a larger working set
would make the term non-zero and the shortcut wrong; `interface_spread_ns` is
the switch that would catch it.

## Complexity and solve time

The transition count is `(layers − 1) × Σ_r |C_r|²`, summed over admitted
residency levels `r` — the `O(层数 × |C|²)` of §4.4 with the residency
multiplicity as its constant.

| model | layers | \|C\| | transitions | Σ\|C_r\|²/\|C\|² | general solve | separable solve |
|---|---|---|---|---|---|---|
| gqa2 | 14 | 1077 | 19,980,727 | 1.325 | 901.5 ms | 14.7 ms |
| mha4 | 28 | 1077 | 41,498,433 | 1.325 | 1820.4 ms | 50.1 ms |

Both `Σ|C_r|²` values are 1,536,979 — the same candidate set, as they must be.
✅ verified (`docs/experiments/SOLVER/raw/summary.txt`).

Solving is under two seconds for a 28-GEMM model against a 1077-point
implementation space; the oracle sweep that produced the validation set took
hours of GPU time for the same space at one model.

## What the unit test pins

`test/unit/chain_dp_test.cpp` builds the smallest model with a prefix, an
interior interface and a suffix, and pins one mutation per assertion:

| assertion | mutation it catches |
|---|---|
| `decomposition_error_ns < 1e-6` | a barrier charged twice, or an interior stage charged to nobody |
| `interface_spread_ns == 0` and separable ≡ general | the shortcut silently disagreeing with the transition loop |
| `general ≤ split ≤ uniform` | a wider state scoring worse than a narrower one |
| `uniform_keys.size() == 1`, `split_shapes.size() == 1` | a mode leaking a per-operator choice the megakernel cannot compile |
| `residency.ctas_per_sm == min over chosen configs` | scoring a plan at an occupancy the hardware would not grant |
| duplicated-GEMM model throws | answering a different recurrence silently |

The residency assertion is a guard on the formulation, not a caught bug:
relaxing the pin to a bound does not change the answer on this model or on
either reference model. It is kept because §4.3's residency is a minimum, and
that is recorded here rather than presented as a regression test that ever
failed.
