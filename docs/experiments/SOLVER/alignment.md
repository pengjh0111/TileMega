# P4.3 — tier-2 alignment propagation

`lib/Solver/AlignmentPropagation.cpp`, driven from `tools/tilemega-solve.cpp`;
reproduce with `bash docs/experiments/SOLVER/run.sh` (no GPU). Per-model traces
land in `raw/alignment_gqa2.txt` and `raw/alignment_mha4.txt`.

## What the constraint is

A producer tile that straddles the boundary of the task that reads it forces
that reader to wait on one more producer tile than its own width needs, and
behind an L1 grid barrier that extra wait is pure serialisation. The `m`-th
consumer task of width `Tr` waits on

```
WaitTiles(m) = ceil((m+1)·Tr / Tm) − floor(m·Tr / Tm)
```

producer tiles of width `Tm`. The inflation is `max_m WaitTiles(m) −
ceil(Tr/Tm)`: the excess over the fewest tiles the consumer's own width could
possibly require. Tier 2 keeps a candidate when the inflation is zero on the N
axis (against the consumer's granularity) and on the K axis (against the
producer's).

## §3.2 — the `d` constraint is derived, not written down

The rule "a QKV GEMM's column tile must align to the head dimension" is never
stated in the solver. `DeriveAlignmentConstraints` walks the generated stage
list forward from each GEMM to the first non-GEMM stage whose operand list
contains that GEMM's output buffer, and reads the granularity off that stage:
`width` for RoPE and KVAppend, `extent` for the elementwise tail. The head
dimension arrives because RoPE's `width` in the generated table *is* `d`.

From `raw/alignment_gqa2.txt` (`Tr` is the derived granularity):

| gemm | n | consumer | `Tr` | producer | `Tk` |
|---|---|---|---|---|---|
| 0 (Q) | 512 | RoPE @4 | **128** | RMSNorm @0 | 512 |
| 1 (K) | 256 | RoPE @5 | **128** | RMSNorm @0 | 512 |
| 2 (V) | 256 | KVAppend @7 | **128** | RMSNorm @0 | 512 |
| 3 (O) | 512 | RMSNorm @10 | 512 | Attention @8 | 128 |
| 4,5 (gate,up) | 1024 | Elementwise @13 | 1024 | RMSNorm @10 | 512 |
| 6 (down) | 512 | RMSNorm @15 | 512 | Elementwise @13 | 1024 |
| 13 (last) | 512 | — | — | Elementwise @28 | 1024 |

`128` on the three QKV GEMMs is `head_dim`, and it is there because the model
says so. ✅ verified. Changing the model changes it: the unit test builds the
same three-stage skeleton with `head = 96` and the admissible set moves with it
(48 and 80 become legal, 112 and 128 illegal — neither exclusion set is a
subset of the other).

## §3.3 — how much it collapses the joint space: nothing, here

| model | before | after (min..max per operator) | joint space | unconstrained operators |
|---|---|---|---|---|
| gqa2 | 1077 | 1077..1077 | 10^42.451 → 10^42.451 | 0 |
| mha4 | 1077 | 1077..1077 | 10^84.902 → 10^84.902 | 0 |

**Tier 2 prunes nothing on this candidate set.** That is the measured result and
it is not a bug: every operator *is* constrained (0 unconstrained), and every
one of the 1077 candidates satisfies its constraint. The reason is arithmetic.
Tier 1 emits `tile_n ∈ {16, 32, 64, 128, 256}` and `tile_k ∈ {8, 16, 32}` —
powers of two — and every granularity the models expose is a power of two
(128, 512, 1024). A power-of-two tile never straddles a power-of-two reader
boundary, so the inflation is identically zero.

To show the derivation computes something rather than nothing, the same
propagation is run over a counterfactual `tile_n` axis stepped by 16:

| model | axis | before | after (per operator) | joint space |
|---|---|---|---|---|
| gqa2 | `tile_n` 16..256 step 16 | 48 | 21..48 | 10^23.537 → 10^21.383 (**≈143× collapse**) |
| mha4 | `tile_n` 16..256 step 16 | 48 | 21..48 | 10^47.075 → 10^42.767 (**≈2.0×10⁴ collapse**) |

On the QKV operators 21 of 48 survive; on the operators whose consumer reads at
1024 all 48 do. ✅ verified.

The honest summary: the mechanism is correct, derived from the model, and
proven to bind on a dense axis — and on the axis the tier-1 generator actually
produces it is a no-op. §3.3 asked for the post-tier-2 count and for an
explanation if it stayed far above 20; it stayed at 1077, and the explanation
is the two sentences above plus one more: 1077 = 216 tile shapes × 5 split
factors, and the split factor is not an alignment-constrained axis at all, so
even a maximally aggressive tier 2 could only ever touch the 216.

## How it reaches the DP

`ChainDpOptions::per_operator_candidates` carries one index list per GEMM. An
empty list — the default — admits everything, so the tier-1 solver is unchanged
by this feature. A uniform answer must satisfy *every* operator's constraint,
not just its own, which is a separate mask (`allow_all`) inside the solver; a
residency level is skipped when the pruning leaves some operator with no state.

Because the pruning is a no-op here, the DP's answer must not move, and the
solver checks that rather than assuming it: `tilemega-solve` solves the uniform
problem twice, once with the mask and once without, and prints
`tier-1-only control picks 16x64x16s2k16 (same answer)` on both models.
✅ verified.
