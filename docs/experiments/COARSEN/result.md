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

**The measurement is a cost-only measurement, and that is deliberate.** A
consumer waits on *every* group of each producer it depends on, because the
generated dependency table carries only "these two stages are coupled" and no
coupling relation to narrow the range with (E2E_L2's blocker 1, and the §1.5.1
debt that `Frontend.cpp` never calls `CouplingDerivation`). So the sweep
measures κ's overhead exactly and its benefit not at all. The benefit is
bounded from the other side by the `nosync` arm, and §4 below shows the two
bounds meeting.

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

### gqa2 — `l2_ms`, the κ curve

| arm | median (ms) | vs `stage` | 95% CI | p |
|---|---|---|---|---|
| `stage` (κ = ∞) | 0.183328 | — | — | — |
| `k1` | 0.618560 | **+237.693%** | [+236.981, +238.728] | 1.7e−11 |
| `k2` | 0.406464 | +121.768% | [+120.805, +122.121] | 1.7e−11 |
| `k4` | 0.297984 | +62.392% | [+62.011, +62.758] | 1.7e−11 |
| `k8` | 0.244928 | +33.704% | [+33.278, +34.078] | 1.7e−11 |
| `k16` | 0.219136 | +19.658% | [+19.337, +20.049] | 1.7e−11 |
| `k32` | 0.206848 | +12.875% | [+12.425, +13.301] | 1.7e−11 |
| `k64` | 0.201728 | +9.969% | [+9.506, +10.548] | 1.7e−11 |
| `k128` | 0.198800 | +8.403% | [+8.205, +8.799] | 1.7e−11 |
| `k256` | 0.197632 | +7.865% | [+7.727, +8.300] | 1.7e−11 |
| `nosync` | 0.183520 | +0.105% | [−0.123, +0.541] | 0.62 — **null control** |

Every κ arm is PASS in 60 / 60 fresh processes; `nosync` is MISMATCH in
60 / 60, as it must be — it is a timing probe with the grid half of L1's
barrier deleted, and the harness is built so it cannot report otherwise.

### gqa2 — `l1_ms`, nine null controls and one ceiling

| arm | median (ms) | vs `stage` | 95% CI | p | role |
|---|---|---|---|---|---|
| `stage` | 0.166912 | — | — | — | |
| `k1` … `k256` | 0.166912–0.167136 | −0.115% … +0.192% | widest [−0.460, +0.572] | 0.36–0.91 | nine null controls |
| `nosync` | 0.107520 | **−35.631%** | [−35.802, −35.533] | 1.7e−11 | **ceiling** |

κ does not touch L1, and nine independent arms say so at a CI half-width of
about ±0.5 %. That is this experiment's own noise floor, measured inside the
experiment rather than assumed — and every entry in the κ table above is two
to three orders of magnitude outside it.

### mha4 — `l2_ms`, the same shape at a different scale

| arm | median (ms) | vs `stage` | 95% CI | p |
|---|---|---|---|---|
| `stage` (κ = ∞) | 0.355328 | — | — | — |
| `k1` | 1.337488 | **+276.413%** | [+275.847, +276.669] | 1.7e−11 |
| `k2` | 0.857952 | +141.652% | [+141.201, +141.798] | 1.7e−11 |
| `k4` | 0.613376 | +72.613% | [+72.346, +72.857] | 1.7e−11 |
| `k8` | 0.492544 | +38.704% | [+38.458, +38.905] | 1.7e−11 |
| `k16` | 0.435200 | +22.494% | [+22.193, +22.711] | 1.7e−11 |
| `k32` | 0.406448 | +14.271% | [+13.829, +14.551] | 1.7e−11 |
| `k64` | 0.395200 | +10.958% | [+10.693, +11.299] | 1.7e−11 |
| `k128` | 0.385248 | +8.321% | [+8.034, +8.675] | 1.7e−11 |
| `k256` | 0.385024 | +8.480% | [+7.988, +8.696] | 3.6e−11 |
| `nosync` | 0.356112 | +0.000% | [−0.306, +0.311] | 0.85 — **null control** |

Every κ arm is PASS in 60 / 60 fresh processes; `nosync` is MISMATCH in 60 / 60.

Two details worth stating rather than smoothing:

- **The tail is flat, not still falling.** `k128` and `k256` have overlapping
  CIs and their medians differ by 0.06 %; the paired ratio even reverses sign
  between them. The curve has reached its asymptote by κ ≈ 128, and the
  asymptote is **+8.5 %, not 0 %** — the same size as gqa2's +7.9 %, on a model
  with twice the stages (60 vs 30). ❌ inferred, not measured apart: a residual
  that does not scale with stage count is what a fixed per-stage difference
  between the two builds would look like — every κ > 0 build indexes a
  grid-deep event array where `stage` indexes a stage-deep one. It is a floor
  either way, and no κ crosses it.
- **The knee is at the same place as gqa2's**, κ ≈ 32, and at the same place as
  the analytic knee. The two models agree on the shape and disagree only on the
  scale of the κ = 1 penalty, 276 % vs 238 %.

### mha4 — `l1_ms`, nine null controls and one ceiling

| arm | median (ms) | vs `stage` | 95% CI | p | role |
|---|---|---|---|---|---|
| `stage` | 0.324608 | — | — | — | |
| `k1` … `k256` | 0.324832–0.325632 | +0.069% … +0.305% | widest [−0.020, +0.555] | 4.1e−04 … 0.14 | nine null controls |
| `nosync` | 0.205552 | **−36.665%** | [−36.754, −36.607] | 1.7e−11 | **ceiling** |

⚠️ Unlike gqa2's, mha4's nine null controls do not straddle zero: all nine are
positive, at +0.069 % to +0.305 %, and five of the nine have p < 0.05. This is
reported as it came out rather than rounded to "no effect". The most likely
cause is that every κ > 0 build allocates a **grid-deep** event array
(`stage_count × gridDim.x`) where `stage` allocates a stage-deep one, so all
nine share one allocation-size offset — consistent with the offset being
independent of κ (the spread across the nine, 0.24 %, is the same size as the
offset itself) and with gqa2, whose smaller array puts the offset under its
noise. Either way the effect is ≤ 0.31 % and the κ curve it would have to
confound spans 276 %.

## 4. What the two bounds say together

Three numbers, all from the tables above, on one scale:

| quantity | gqa2 | mha4 |
|---|---|---|
| L1, grid barrier (`stage`) | 0.166912 ms | 0.324608 ms |
| L1, all grid synchronization deleted (`nosync`, wrong answer) | 0.107520 ms | 0.205552 ms |
| L2, event scheme at its cheapest granularity (`stage`) | 0.183328 ms | 0.355328 ms |
| L2, finest events (`k1`) | 0.618560 ms | 1.337488 ms |

- Synchronization is **not** a rounding error here: it is **35.6 % / 36.7 %** of
  L1's runtime and **41.3 % / 42.1 %** of L2's. There is real headroom, and the
  two models agree on how much.
- But the event scheme, at the coarsest granularity it can have, already costs
  **+9.8 % / +9.5 %** against the grid barrier it replaces. A hypothetical κ
  that eliminated *all* overwait would still have to overcome that before it
  reached parity with the code the repository already ships.
- And the direction κ moves is the wrong one, monotonically, by up to
  **238 % / 276 %**.

So the headroom is real and κ is not the instrument that reaches it. The
analytic knee at κ ≈ 32 is where `overwait` starts to dominate `waits`; the
hardware says the fixed cost of the grouped-event machinery dominates *both*
long before that, because each extra event is an extra `atomicAdd` on a
distinct cache line plus an extra polling loop per consumer. Both models
bottom out around +8 % and neither reaches parity at any κ measured.

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
only then would κ become a decision. That work is §1.5.1's "耦合推导尚未接入
真实前端路径" debt, not this one.

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
4. **`MODELS` was added to `run.sh`** so the interrupted half could be re-run
   without re-measuring the 660 processes that had already completed. Default
   is both models; the gqa2 rows in `raw/kappa_arms.tsv` come from the first
   invocation and the mha4 rows from the second.
