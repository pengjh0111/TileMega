# P4.6 — the event-granularity ablation, and why κ is not a DP state variable

Reproduce: `bash docs/experiments/COARSEN/run.sh` (GPU; `RUNS=60`, 11 arms,
two models = 1320 fresh processes). `MODELS=mha4` re-runs one half. The
analytic probe needs no GPU.

The question §4.4 asks is whether the event granularity κ belongs in the DP's
state. §P4.6 phrases the test as "κ 消融曲线；曲线平坦则该维度无收益". The
curve is not flat. It is steeply monotone — and it points at the endpoint the
generator already emits, which answers the question just as decisively and for
a better reason.

## 1. What κ is

`C_κ = ⌊·/κ⌋ ∘ C` (§2.3). In the harness (`ModelHarness.cuh`) it groups κ
consecutive CTAs of a stage into one event, so a stage publishes
`⌈active/κ⌉` events instead of one. κ = 0 in the macro means the per-stage
scheme — one event per stage — which is the degenerate coarsest case and the
default build. L1 has no κ knob at all: its grid barrier is one event per
stage by construction, which is what makes every κ arm's `l1_ms` a null
control.

**The measurement is a cost-only measurement.** A consumer waits on *every*
group of each producer it depends on, so the sweep measures κ's overhead
exactly and its benefit not at all. The benefit is bounded from the other side
by the `nosync` arm, and §4 below shows the two bounds meeting.

When this was first written the reason was that the generated table carried
only "these two stages are coupled" (E2E_L2's blocker 1, and the §1.5.1 debt
that `Frontend.cpp` never called `CouplingDerivation`). That debt is now
closed and the table carries the derived relation — but the reason survives in
a different form: every arm here compiles with `-include plan_<model>_uniform.h`,
which reparameterizes the GEMM task space and makes the fitted windows
inadmissible, so every arm reports `wait_table=degraded` and waits exactly as
before. §8 records why that is a correctness requirement rather than a
shortcoming, and §7 carries the benefit-side measurement, taken on the exact
table instead.

## 2. The analytic half — the two columns move in opposite directions ✅

`kappa_probe.cpp` applies `C_κ` to every derived edge of both accepted models
and reports, per κ, summed over edges:

- `waits(κ) = |C_κ(x)|` — events one consumer task polls. Falls with κ.
- `overwait(κ) = κ·waits(κ) − waits(1)` — producer tasks a consumer is forced
  to wait for beyond the ones it actually reads. Rises with κ.

| κ | gqa2 waits | gqa2 overwait | mha4 waits | mha4 overwait |
|---|---|---|---|---|
| 1 | 9071 | 0 | 18064 | 0 |
| 2 | 4807 | 543 | 9568 | 1072 |
| 4 | 2675 | 1629 | 5320 | 3216 |
| 8 | 1609 | 3801 | 3196 | 7504 |
| 16 | 1077 | 8161 | 2136 | 16112 |
| **32** | **812** | **16913** | **1608** | **33392** |
| 64 | 808 | 42641 | 1600 | 84336 |
| 128 | 806 | 94097 | 1596 | 186224 |
| 256 | 806 | 197265 | 1596 | 390512 |
| 1024 | 806 | 816273 | 1596 | 1616240 |

✅ verified. The knee is at **κ ≈ 32** on both models: past it `waits` is flat
(812 → 806, 1608 → 1596) while `overwait` keeps doubling. If κ were free on
the hardware, 32 is where the analytic trade stops paying.

Both models saturate at a nonzero `waits` because a handful of edges are
already single-event at κ = 1 (`waits = 1`), and grouping cannot take those
below one.

## 3. The hardware half — every κ is worse than no κ ✅

60 interleaved rounds per model, one fresh process per arm per round, starting
arm rotated, reported paired within round (F-46). Percentages are the median
within-round ratio; the bracket is a 95% bootstrap over rounds and `p` a
two-sided Wilcoxon signed-rank.

The tables below are the **2026-09-04 re-run**, after Parts 1 and 2 of the
wiring round landed. The 2026-09-03 run they replace is preserved verbatim in
`raw/kappa_arms_2026-09-03.tsv` and `raw/log_2026-09-03/`; §3.5 compares the
two, because the difference between them is itself a measurement.

### gqa2 — `l2_ms`, the κ curve

| arm | median (ms) | vs `stage` | 95% CI | p |
|---|---|---|---|---|
| `stage` (κ = ∞) | 0.183296 | — | — | — |
| `k1` | 0.627712 | **+242.478%** | [+241.904, +243.241] | 1.7e−11 |
| `k2` | 0.411520 | +124.511% | [+124.022, +124.859] | 1.7e−11 |
| `k4` | 0.301056 | +64.255% | [+64.062, +64.976] | 1.7e−11 |
| `k8` | 0.247808 | +35.264% | [+35.000, +35.469] | 1.7e−11 |
| `k16` | 0.222208 | +20.949% | [+20.670, +21.229] | 1.7e−11 |
| `k32` | 0.209776 | +14.525% | [+14.065, +14.705] | 1.7e−11 |
| `k64` | 0.203776 | +11.064% | [+10.789, +11.173] | 1.7e−11 |
| `k128` | 0.201616 | +9.747% | [+9.325, +10.038] | 1.7e−11 |
| `k256` | 0.200608 | +9.445% | [+9.114, +9.807] | 1.7e−11 |
| `nosync` | 0.183456 | +0.535% | [+0.122, +0.672] | 1.1e−03 — **null control** |

Every κ arm is PASS in 60 / 60 fresh processes; `nosync` is MISMATCH in
60 / 60, as it must be — it is a timing probe with the grid half of L1's
barrier deleted, and the harness is built so it cannot report otherwise.

⚠️ `nosync`'s L2 column is +0.535 % with p = 1.1e−03 here, where the Sep-3 run
had +0.105 % at p = 0.62. `nosync` deletes L1's grid barrier and must not touch
L2 at all, so this is a half-percent bias in the *control*, reported rather
than rounded away. It is the same order as the L1 null controls' spread below,
it is 450× smaller than the effect it is a control for, and the mha4 half of
the same run puts the same quantity at +0.063 % (p = 0.19). ❌ inferred: an
ordering effect within a round, since `nosync` is the last arm of eleven.

### gqa2 — `l1_ms`, nine null controls and one ceiling

| arm | median (ms) | vs `stage` | 95% CI | p | role |
|---|---|---|---|---|---|
| `stage` | 0.167936 | — | — | — | |
| `k1` … `k256` | 0.167936–0.168928 | −0.010% … +0.411% | widest [−0.191, +0.714] | 0.010–0.84 | nine null controls |
| `nosync` | 0.106896 | **−36.410%** | [−36.585, −35.966] | 1.7e−11 | **ceiling** |

κ does not touch L1, and nine independent arms say so at a CI half-width of
about ±0.5 %. That is this experiment's own noise floor, measured inside the
experiment rather than assumed — and every entry in the κ table above is two
to three orders of magnitude outside it.

### mha4 — `l2_ms`, the same shape at a different scale

| arm | median (ms) | vs `stage` | 95% CI | p |
|---|---|---|---|---|
| `stage` (κ = ∞) | 0.357712 | — | — | — |
| `k1` | 1.360896 | **+280.789%** | [+279.896, +281.328] | 1.7e−11 |
| `k2` | 0.874224 | +144.448% | [+144.000, +144.718] | 1.7e−11 |
| `k4` | 0.626624 | +75.181% | [+74.901, +75.518] | 1.7e−11 |
| `k8` | 0.502928 | +40.857% | [+40.490, +41.118] | 1.7e−11 |
| `k16` | 0.445584 | +24.642% | [+24.361, +24.896] | 1.7e−11 |
| `k32` | 0.418608 | +16.966% | [+16.715, +17.176] | 1.7e−11 |
| `k64` | 0.404480 | +12.901% | [+12.821, +13.408] | 1.7e−11 |
| `k128` | 0.397040 | +11.032% | [+10.765, +11.207] | 1.7e−11 |
| `k256` | 0.395264 | +10.632% | [+10.291, +10.958] | 1.7e−11 |
| `nosync` | 0.357376 | +0.063% | [−0.027, +0.287] | 0.19 — **null control** |

Every κ arm is PASS in 60 / 60 fresh processes; `nosync` is MISMATCH in 60 / 60.

Two details worth stating rather than smoothing:

- **The tail is flat, not still falling.** `k128` and `k256` differ by 0.4
  points with overlapping CIs on mha4 and by 0.3 points on gqa2. The curve has
  reached its asymptote by κ ≈ 128, and the asymptote is **+9.4 % / +10.6 %,
  not 0 %** — the same size on a model with twice the stages (60 vs 30).
  ❌ inferred, not measured apart: a residual that does not scale with stage
  count is what a fixed per-stage difference between the two builds would look
  like — every κ > 0 build indexes a grid-deep event array where `stage`
  indexes a stage-deep one. It is a floor either way, and no κ crosses it.
- **The knee is at the same place as gqa2's**, κ ≈ 32, and at the same place as
  the analytic knee. The two models agree on the shape and disagree only on the
  scale of the κ = 1 penalty, 281 % vs 242 %.

### mha4 — `l1_ms`, nine null controls and one ceiling

| arm | median (ms) | vs `stage` | 95% CI | p | role |
|---|---|---|---|---|---|
| `stage` | 0.324736 | — | — | — | |
| `k1` … `k256` | 0.324608–0.324864 | −0.079% … +0.005% | widest [−0.292, +0.236] | 0.085–1.00 | nine null controls |
| `nosync` | 0.207632 | **−36.102%** | [−36.325, −36.006] | 1.7e−11 | **ceiling** |

**Self-correction against the Sep-3 write-up.** That run reported all nine of
mha4's L1 null controls positive (+0.069 % … +0.305 %, five with p < 0.05) and
attributed the offset to the grid-deep event array. It did not reproduce: in
this run the same nine straddle zero (−0.079 % … +0.005 %) and none has
p < 0.08. The offset was a property of that session, not of the array, and the
explanation offered for it is withdrawn. What survives is the bound — the
effect is ≤ 0.31 % in the run that saw it and ≤ 0.08 % in the run that did not,
against a κ curve spanning 281 %.

### 3.5 The two runs, side by side — the shape reproduces, the absolute does not

Same script, same binaries' sources, same machine, 24 hours apart, 1320 fresh
processes each:

| quantity | 2026-09-03 | 2026-09-04 | Δ |
|---|---|---|---|
| gqa2 `stage` l1_ms | 0.166912 | 0.167936 | +0.61 % |
| gqa2 `stage` l2_ms | 0.183328 | 0.183296 | −0.02 % |
| gqa2 `k1` l2_ms | 0.618560 | 0.627712 | **+1.48 %** |
| gqa2 `k1` vs `stage` | +237.693 % | +242.478 % | +4.8 points |
| mha4 `stage` l2_ms | 0.355328 | 0.357712 | +0.67 % |
| mha4 `k1` l2_ms | 1.337488 | 1.360896 | **+1.75 %** |
| mha4 `k1` vs `stage` | +276.413 % | +280.789 % | +4.4 points |
| gqa2 `nosync` l1_ms | 0.107520 | 0.106896 | −0.58 % |

Every median moves by up to 1.8 % between sessions while every within-run
paired comparison is unchanged in sign, in ordering and in knee position. This
is F-61 again, on a second experiment: **only within-session pairing is
admissible**, and an absolute millisecond figure from one session must not be
compared against another's. The ordering conclusion — κ monotone, knee at 32,
asymptote about +10 % above `stage` — is identical across both runs.

## 4. What the two bounds say together

Three numbers, all from the tables above, on one scale (2026-09-04 run):

| quantity | gqa2 | mha4 |
|---|---|---|
| L1, grid barrier (`stage`) | 0.167936 ms | 0.324736 ms |
| L1, all grid synchronization deleted (`nosync`, wrong answer) | 0.106896 ms | 0.207632 ms |
| L2, event scheme at its cheapest granularity (`stage`) | 0.183296 ms | 0.357712 ms |
| L2, finest events (`k1`) | 0.627712 ms | 1.360896 ms |

- Synchronization is **not** a rounding error here: it is **36.4 % / 36.1 %** of
  L1's runtime and **41.7 % / 42.0 %** of L2's. There is real headroom, and the
  two models agree on how much.
- But the event scheme, at the coarsest granularity it can have, already costs
  **+9.1 % / +10.2 %** against the grid barrier it replaces. A hypothetical κ
  that eliminated *all* overwait would still have to overcome that before it
  reached parity with the code the repository already ships.
- And the direction κ moves is the wrong one, monotonically, by up to
  **242 % / 281 %**.

So the headroom is real and κ is not the instrument that reaches it. The
analytic knee at κ ≈ 32 is where `overwait` starts to dominate `waits`; the
hardware says the fixed cost of the grouped-event machinery dominates *both*
long before that, because each extra event is an extra `atomicAdd` on a
distinct cache line plus an extra polling loop per consumer. Both models
bottom out around +10 % and neither reaches parity at any κ measured.

⚠️ `nosync` is a lower bound on time, not an achievable target: it produces the
wrong answer. It bounds what any correct scheme could win; it does not promise
that any correct scheme wins it.

## 5. Decision: κ is not in the DP state, and why that is not "the curve is flat"

`ChainDpOptions` has no κ field and none was added. The reason recorded here is
stronger than the one §P4.6 anticipated:

- **The argmin over κ is the same κ for every configuration**, on both
  accepted models. The curve is monotone decreasing in κ, and its limit is the
  per-stage scheme, which is what the generator already emits unconditionally. A state variable whose
  optimal value never depends on any other decision is not a state variable.
- Adding it would cost `|C|` × |κ| states per layer and could only ever return
  the value it already has.

What *would* change this, stated so it is falsifiable: if the generated
dependency table carried the coupling relation, a consumer would wait on only
the groups it actually reads and `overwait` would stop being paid. The cost
curve above would then have to be re-measured against a *benefit* curve, and
only then would κ become a decision.

That condition has since been met — §1.5.1's "耦合推导尚未接入真实前端路径"
debt is closed and the table now carries the derived relation — so §7 answers
it. The answer is still "κ is not a DP state variable", and the reason is a new
one: the benefit exists, is bounded at 2.4 % of polls, and reaches exactly zero
by κ = 4.

## 6. Detours

1. **The first sweep died silently at the gqa2/mha4 boundary.** `grep -c` exits
   1 when nothing matches, and under `set -euo pipefail` that killed the script
   on exactly the arm whose PASS count is supposed to be zero. Fixed as
   `{ grep -c … || true; } | awk …`. Recorded because the failure mode is a
   sweep that stops halfway and leaves a plausible-looking partial table.
2. **`gpu_stat_run.sh` exits non-zero when any run mismatches**, which the
   `nosync` arm does by design. The loop therefore reads
   `|| [[ "${arm}" == nosync ]]` — narrow enough that a real mismatch on any
   other arm still stops the sweep, and the final `awk` still requires
   `nosync` to have mismatched every single time.
3. **`pgrep -f <script>` in a waiter loop self-matches**, so a background sweep
   looked alive for half an hour after it had died. The re-run is launched
   under `setsid nohup` and watched by tailing its log, not by polling `pgrep`.
4. **`MODELS` was added to `run.sh`** so an interrupted half can be re-run
   without re-measuring the 660 processes that had already completed. Default
   is both models, which is how the 2026-09-04 table in `raw/kappa_arms.tsv`
   was produced; the superseded 2026-09-03 table was assembled from two
   invocations.
5. **"The re-run sweep died mysteriously" — a misdiagnosis, corrected.** After
   the wiring round the sweep stopped on the `gqa2 k1` arm, and the first
   explanation written down was that `timeout --foreground` had killed it. It
   had not: the arm genuinely reported `RESULT status=MISMATCH`, and detour 2's
   guard stopped the sweep because it is supposed to. The guard was working and
   the product was broken — see §8. Recorded because the wrong explanation was
   the comfortable one, and it would have hidden a silent under-wait.

## 7. Re-measured with the derived table wired in — κ is still monotone, for a new reason ✅

§5 above made the decision falsifiable: it said κ could only become a decision
once the generated table carried the coupling relation, so that a consumer
stopped paying `overwait`. Part 1 and Part 2 of the wiring round did exactly
that, and the re-measurement is `docs/experiments/E2E_L2/` part 3.2 —
κ ∈ {1, 2, 4, 8, 16, 32}, both models, the **exact** window table against a
`kAll` control of the same binary, 25 fresh processes per arm (24 arms, 600
runs, all PASS), paired within round.

The old reason ("the sweep measures κ's cost exactly and its benefit not at
all") may no longer be reused. It is now measurable on both sides.

### The benefit side — nonzero, small, and gone by κ = 4

Device-side poll counts from `TILEMEGA_WAIT_PROFILE`
(`E2E_L2/raw/part3/kappa_polls_summary.txt`), total over all edges, window
table vs the same model as `kAll`:

| κ | gqa2 seq=4 | gqa2 seq=512 | mha4 seq=4 | mha4 seq=512 |
|---|---|---|---|---|
| 1 | **1.0244×** | 1.0002× | **1.0222×** | 1.0003× |
| 2 | 1.0161× | 1.0002× | 1.0147× | 1.0002× |
| 4 | 1.0000× | 1.0000× | 1.0000× | 1.0000× |
| 8 | 1.0000× | 1.0000× | 1.0000× | 1.0000× |
| 16 | 1.0000× | 1.0000× | 1.0000× | 1.0000× |
| 32 | 1.0000× | 1.0000× | 1.0000× | 1.0000× |

The benefit peaks at **2.4 %** of polls, at the finest granularity, and is
**identically zero from κ = 4 on**: a group of four or more CTAs already
straddles every window boundary the table can draw, so the narrowed set and the
relaxed set name the same groups.

### The cost side — the exact table is slower than its own `kAll` control at every κ

`l2_over_l1`, median of within-round ratios, 25 rounds, 95 % bootstrap:

| κ | gqa2 window | gqa2 `kAll` | mha4 window | mha4 `kAll` |
|---|---|---|---|---|
| 1 | 1.069314 [1.069095, 1.069665] | 1.066121 [1.065951, 1.066225] | 1.075861 [1.075766, 1.076267] | 1.072457 [1.072004, 1.072941] |
| 2 | 1.064067 [1.063421, 1.064262] | 1.059277 [1.058660, 1.059446] | 1.067740 [1.067669, 1.068215] | 1.062969 [1.062530, 1.063033] |
| 4 | 1.056107 [1.055766, 1.056652] | 1.050902 [1.050774, 1.051583] | 1.057930 [1.057820, 1.058283] | 1.052544 [1.052030, 1.052632] |
| 8 | 1.053937 [1.053639, 1.054425] | 1.048324 [1.047546, 1.049025] | 1.054976 [1.054903, 1.055465] | 1.049289 [1.048939, 1.049664] |
| 16 | 1.053591 [1.052775, 1.053926] | 1.048064 [1.047166, 1.048419] | 1.054473 [1.053960, 1.054976] | 1.048738 [1.048341, 1.049151] |
| 32 | 1.053023 [1.052880, 1.053770] | 1.048220 [1.048095, 1.048959] | 1.054153 [1.053980, 1.054466] | 1.048407 [1.048301, 1.048803] |

Every pair has **disjoint** CIs, in the same direction, on both models: the
narrowed table is 0.48–0.57 percentage points *slower* than waiting on
everything. κ = 32 isolates the mechanism cleanly, because there the two tables
poll an identical number of groups (276 / 276 at seq = 4, 6028 / 6028 at
seq = 512) — so the whole ~0.48-point gap is the arithmetic of evaluating the
window: a division, a multiply, two clamps and a modulo per owned task, inside
thread 0's serial wait loop.

### The new reason

κ is still monotone toward κ = 0, and the ordering is unchanged
(κ = 0 at 1.0362 < κ = 32 at 1.0530 < κ = 1 at 1.0693 on gqa2). The reason on
record is now:

> The benefit of a finer event granularity is real but bounded at 2.4 % of
> polls and identically zero from κ = 4, because every fitted window on a
> tile-row edge has `count = scale = Tm = 128`, which on this GPU equals
> `gridDim.x`; under the grid-stride placement one contiguous task interval a
> stride wide covers every CTA. That is a §2.3 **Place** property, not a
> weakness of `C` — the windows are exact. And the arithmetic that exploits the
> window costs 0.48–0.57 points, more than the ≤ 2.4 % it can win.

`ChainDpOptions` still has no κ field. What would change *this* answer is
changing the placement, not the derivation: `E2E_L2/waitset.md` carries a
counting what-if for a blocked placement (1.047× at κ = 1, 1.082× at κ = 32,
❌ inferred), and even that is bounded by the 5.9 % of poll mass that sits on
tile→tile edges at all.

## 8. The precondition this sweep discovered: a window is only valid at the granularity it was fitted at ✅

This is a **correctness** finding, and it was found by the sweep failing, not
by a test.

The first re-run of `run.sh` after the wiring round died on the `gqa2 k1` arm,
and it died correctly: `set -e` plus the `|| [[ "${arm}" == nosync ]]` guard
(detour 2 below) stops the sweep the moment an arm that must pass does not.
✅ verified **0 / 50** fresh processes pass on that build,
`l2_vs_l1_mismatch=4096` in every one, and `l2_iter1_vs_iter0_mismatch=2319` —
nondeterministic, i.e. a genuine under-wait rather than a wrong constant.

The cause is a compile-time coupling nobody had written down. The wait window's
`div`, `scale`, `offset` and `count` are integers over a GEMM task
decomposition, and that decomposition is a *compile-time* knob the generator
does not control:

```
generator:  tilemega.g = {Tm 128, Tn 128}, reduction unsplit
this sweep: -include plan_gqa2_uniform.h  →  TILEMEGA_GEMM_TILE_M 16,
                                             TILEMEGA_GEMM_SPLIT_K 16
```

Every fitted constant then names the wrong producer tasks — and it names *too
few*, so the failure is silent rather than a deadlock. The sweep is the only
thing in the repository that rebuilds a generated model at a different `g`,
which is why 150 fresh processes of Part 2's own validation missed it: at the
default granularity the table is exact and correct.

Fixed at both ends, single-variable:

* the generator emits the granularity it fitted against —
  `TILEMEGA_GENERATED_WINDOW_TILE_M / _TILE_N / _SPLIT_K`, from `tilemega.g`;
* `ModelHarness.cuh` admits the narrow path only when the compiled granularity
  agrees (and `TILEMEGA_GEMM_VARIANT_COUNT == 1`), degrading the whole table to
  `kAll` otherwise. `kAll` is always a superset of any window, so the
  degradation is safe by construction rather than by measurement;
* every run reports which it got: `E2E_KAPPA event_kappa=… wait_table=exact`
  or `wait_table=degraded`, so a measurement can never be read as exercising
  windows when it is not.

✅ verified after the fix, `raw/waittable/fresh_processes.txt`:

| build | wait_table | fresh processes | distinct L2 hash |
|---|---|---:|---|
| `-include plan_gqa2_uniform.h`, κ = 1 (pre-fix) | — | **0 / 50 PASS** | — |
| `-include plan_gqa2_uniform.h`, κ = 1 (post-fix) | `degraded` | **50 / 50 PASS** | `8a737188b958a2ae` |
| default granularity, κ = 1 (post-fix) | `exact` | **50 / 50 PASS** | `5245714bc5d3ab4d` |

**What this means for the tables in §3.** Every arm of this sweep carries
`-include plan_<model>_uniform.h`, so every arm now reports
`wait_table=degraded` and waits exactly as the pre-wiring build did. The κ
curve in §3 is therefore measuring the same wait code it measured before the
dependency table existed, which is what makes §3 and §7 answer two different
questions rather than the same one twice: §3 is κ against a relaxed table at
the solver's granularity, §7 is κ against the exact table at the generator's.
