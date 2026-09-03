# P4.8 — Place: the worth-it judgment first, and it did not come out as predicted

Reproduce: `bash docs/experiments/PLACE/run.sh` (GPU; `RUNS=60`, 5 arms, two
models = 600 fresh processes). The analytic half needs no GPU.

§P4.8's last line is the order the work was done in: *"oracle 判断投入价值：
round-robin vs 强启发式 vs oracle，差距 <2% 则简化"*. Two things came out of
following it, and they point in opposite directions:

- The **objective §4.3 specifies** — temporal locality, `|R(c₁) ∩ R(c₂)|` — is
  worth **at most ~1 %** on these models. Under the brief's own rule, simplify.
- **Placement itself** is worth **7 %**, by a mechanism that objective does not
  model at all. And the one permutation the objective argues for is the
  **worst** arm measured, by 17 %.

Both are reported. The second is not a reason to keep the first.

## 1. The objective, and when it is flat by construction

A placement is a bijection from CTAs to the tasks of a stage. Every temporal
locality objective on that bijection has the same shape — a sum over
co-located task pairs of

```
w(c₁, c₂) = |R(c₁) ∩ R(c₂)|
```

— so it distinguishes two placements only if `w` distinguishes two pairs.
`place_probe.cpp` computes `w` exactly: `R(c)` is `AccessRelation`'s element
box, the pair overlap is an isl map from `(c₁, c₂)` to the elements both read,
and the count is `isl_map_card`. Parameters bound at `S = 512, L_s = 1024,
past = 512`.

## 2. The measured affinity structure ✅

126 operands over both models, 0 uncounted:

| model | operands | `w` constant in every axis (degenerate) | `w` varies along some axis (structured) |
|---|---|---|---|
| gqa2 | 46 | 26 | 20 |
| mha4 | 80 | 44 | 36 |

The structured cases have three shapes, and all three are one fact:

| shape | count (gqa2) | reading |
|---|---|---|
| `m=0 n=524288` | 32 | a GEMM's A operand: two tasks differing only in `n` share the **whole** A panel; two differing in `m` share nothing |
| `m=0 n=1835008` | 6 | the same, on the wider MLP down-projection |
| `s=16384 h=0 j=0`, `s=0 h=0 j=128`, `s=16384 h=0..16384 j=0` | 18 | attention's K/V panels: full sharing along the chunk axis, none across heads |

So `w` is two-valued: full sharing inside a group, zero outside. There is a
real objective at `S = 512`, and it says "co-locate the tasks of one `m` row".

## 3. On the measured kernel the objective is *constant*, and that was predicted ✅

The E2E fixture is `seq = 4` (`manifest.json`) and the DP's uniform answer is
`tile_m = 16`, so every GEMM in the measured megakernel has exactly **one M
tile**. All its tasks share the whole A panel; `w` is one number; the objective
takes the **same value on all five arms**, including `scatter`.

This was written down before the sweep ran. What the sweep then found is that
the arms are *not* equal on hardware — so the prediction was right about the
objective and wrong about the conclusion drawn from it. The objective being
constant does not make placement free; it makes the objective blind.

## 4. The hardware half — five arms, two of them bounds ✅

| arm | what it is |
|---|---|
| `ident` | `TILEMEGA_PLACEMENT=0`, the default build |
| `ident2` | a second binary compiled exactly like `ident` — its measured "effect" is the noise floor |
| `pair` | the only permutation the affinity argues for: CTAs co-resident on one SM take consecutive task indices |
| `reverse` | `g−1−b`: keeps adjacency exactly, only relabels |
| `scatter` | `31·b mod g` — destroys index adjacency outright |

60 interleaved rounds, one fresh process per arm per round, starting arm
rotated, reported paired within round (F-46). `ident` and `ident2` are required
to have byte-identical SASS, and the script checks it, so the knob cannot have
moved the baseline it is measured against. Every arm PASSed **60/60** on both
models: a placement is a correctness change, since `active` and the event group
are both derived from `blockIdx.x` and must be relabelled together.

### gqa2

| metric | `ident` (ms) | `ident2` (floor) | `pair` | `reverse` | `scatter` |
|---|---|---|---|---|---|
| `l05_ms` | 0.173888 | −0.037% p=0.38 | +15.958% [+15.467,+16.667] | −1.095% [−1.360,−0.567] | +0.708% [+0.444,+1.698] |
| `l1_ms` | 0.167936 | +0.124% p=0.70 | **+17.017%** [+16.594,+17.178] | **−7.053%** [−7.279,−6.748] | +1.204% [+1.047,+1.555] |
| `l2_ms` | 0.184144 | −0.148% p=0.18 | +14.190% [+13.491,+14.795] | −1.589% [−1.964,−1.114] | +0.649% [−0.113,+1.140] |

### mha4

| metric | `ident` (ms) | `ident2` (floor) | `pair` | `reverse` | `scatter` |
|---|---|---|---|---|---|
| `l05_ms` | 0.322656 | −0.457% p=0.16 | +16.663% [+16.049,+17.217] | −1.265% [−1.615,−0.658] | +0.656% [+0.089,+1.208] |
| `l1_ms` | 0.325712 | −0.005% p=0.14 | **+18.172%** [+17.868,+18.530] | **−6.939%** [−7.187,−6.769] | −0.751% [−1.237,−0.511] |
| `l2_ms` | 0.356560 | −0.144% p=0.99 | +13.619% [+13.429,+13.973] | −1.986% [−2.163,−1.724] | +0.545% [+0.272,+0.857] |

Every `pair` and `reverse` entry has p ≤ 1.0e−4; the `pair` and `reverse` L1
entries are p = 1.7e−11. The two models agree to within a percentage point on
every arm and every metric, which is the strongest thing in the table: two
different models, two different fixtures, one number.

⚠️ The `ORACLE` lines in `raw/place_summary.txt` quote the `l2_ms`
ident-vs-ident2 contrast as the floor for all three metrics. On `l05_ms` the
floor is looser (mha4's is −0.457 % with a CI reaching ±1.1 %), so the ~1.2 %
`l05` effects are the least secure numbers here. The L1 effects are two orders
of magnitude clear of any of them.

## 5. What the arms mean, separated by how well each is established

**`pair` loses 17 %, and this is derivable from the map — ✅.** With
`grid = 256, ctas_per_sm = 2, num_sms = 128` (from `E2E_RESOURCE`), the pair
map puts logical index `L` on the CTA whose SM is `⌊L/2⌋`. A stage with `A`
active tasks therefore runs on `⌈A/2⌉` SMs, against `min(A, 128)` under the
identity — **half the machine, on every stage that does not fill the grid.**
That is not a side effect of the heuristic; it *is* the heuristic. Making
co-resident CTAs take consecutive task indices is the same thing as packing the
low task indices onto few SMs. §4.3's objective cannot see this, because
occupancy is not a term in it.

**The locality component is worth ≤ ~1 % — ✅, and it is `scatter` that says
so.** `scatter` destroys index adjacency completely and costs +1.204 % / −0.751 %
on L1: one model slightly worse, the other slightly *better*, both about one
percent. If task-index adjacency carried real reuse, scatter would be the
expensive arm. It is not. Against §P4.8's own "差距 <2 % 则简化" threshold, the
temporal-locality objective is below the line and list scheduling on it is not
worth building.

**`reverse` gains 7 % on L1 and the mechanism is not established — ❌.** This is
the honest state of it. What is ruled out:

- *Not locality.* `reverse` preserves index adjacency exactly, so it has the
  same locality structure as `ident`; and `scatter`, which has none, costs ~1 %.
- *Not occupancy.* Under `g − 1 − b`, the active set `{0…A−1}` lands on hardware
  CTAs `{g−A…g−1}`, which is the same count of CTAs per SM as `ident`'s
  `{0…A−1}`, only on the other resident slot.
- *Not the barrier's structure.* `GridBarrier` has no master CTA — the last
  arriver notifies — so "CTA 0 is busy" cannot be it.

What is left is direction-specific: which SM runs which task, and in which
resident slot. The effect is 5–6× larger on L1 (persistent grid + barrier) than
on L05 (one kernel per stage), which points at something about repeated
residency rather than a single launch's scheduling. Recorded as an open
question, not as a mechanism. The experiment that would settle it is per-stage
timing broken out by active count, which this harness does not emit.

## 6. Decision

- **Do not build list scheduling on §4.3's temporal-locality objective.** Its
  measured worth is under the 2 % the brief itself set as the simplify
  threshold, and on the accepted fixture its value is provably constant across
  every placement.
- **Do not ship `pair`.** The one permutation the objective recommends is the
  worst arm by 17 %, for a reason the objective structurally cannot represent.
- **Do not ship `reverse` either**, despite the 7 %. It is a real, reproducible,
  two-model effect, but shipping a permutation whose mechanism is unknown means
  shipping a number that can silently invert on the next fixture or the next
  architecture. It is recorded as measured headroom, and P4.8's remaining boxes
  stay open rather than being closed with an unexplained win.
- What §4.4's cost model should take from this is **occupancy under the active
  count**, which it already has as the wave decomposition — not a locality term.

## 7. Detours

1. **The `pair` arm needed a `static __device__` variable** and a
   `cudaMemcpyToSymbol` to know the blocks-per-SM it pairs over. Both live
   inside `#if TILEMEGA_PLACEMENT == 1`: merely declaring the symbol shifted
   constant-bank offsets and made the default build's SASS differ from the
   pre-knob binary, which would have invalidated the `ident`/`ident2` equality
   check for a reason unrelated to placement.
2. `grep -c` exits 1 on no match and under `set -euo pipefail` that kills the
   sweep; wrapped as `{ grep -c … || true; } | awk …`, the same fix P4.6 needed.
3. **The analytic half and the hardware half measure different sizes.**
   `place_probe.cpp` binds `S = 512`; the E2E fixture is `seq = 4`. §2's
   affinity structure is a statement about `S = 512`, and §3's "one M tile" is
   a statement about the fixture. Neither is generalised into the other.
