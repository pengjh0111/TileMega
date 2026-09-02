# Part 4 — three-tier implementation pruning and the backend cost query

Reproduce the host-side numbers with `build-portable/backend_query_test` (no
GPU needed). The tier-1 and tier-3 measurements come from the Part 6 oracle
sweep (`docs/experiments/ORACLE/`), which compiles the same candidate space
for real.

Target of every number below: RTX 4090 (sm_89, 128 SMs, 101376 B opt-in
dynamic smem per CTA, 102400 B/SM, 1536 threads/SM), read from
`TargetSpec::Probe()`, not hard-coded.

## 4.1 Why a middle tier — the measured gap

| tier | what it answers | per candidate | how measured |
|---|---|---|---|
| 1 — host closed form | legality, threads, smem, cluster, alignment, arch | **2.8–3.1 µs** | 636–700 µs for 224 candidates, `backend_query_test` |
| 1' — CUTLASS `constexpr` traits | the same, from the collective itself | **65.4 ms** | 19.61 s for 300 candidates, one nvcc TU (`ORACLE/raw/tier1_summary.txt`) |
| 2 — analytical rank | an ordering, never a verdict | **2.9 µs** | 650 µs for 224 candidates × 14 GEMM shapes |
| 3 — real nvcc + ptxas | registers, and *actual* compilability | **0.845 s wall at 32-way**, **21.3 s serial** | 1893.85 s for 2240 compiles at 32-way (`ORACLE/raw/tier3_summary.txt`); ✅ one compile of the same TU run alone: 21.28 / 21.50 / 20.05 s wall, 21.2 s user+sys (median) |

✅ verified. The tier-1/tier-3 ratio is six to seven orders of magnitude,
which is the concrete form of last round's 0.176 s vs 4.658 s observation:
enumerating and ranking the whole space costs less than **1.4 ms**, while
compiling it costs **13.2 CPU-hours** (2240 × 21.2 s). A middle tier is what
makes the tier-3 budget spendable on a top-k rather than on the space.

The serial figure is the honest per-candidate cost; the 0.845 s wall is what
32-way parallelism buys, and 21.2 / 0.845 = 25.1 of a possible 32 is the
scaling actually achieved. It was measured while the oracle's screening pass
held the GPU, which costs CPU time but not GPU time — the compile never
touches the device.

The 1' row is the reason the closed form exists at all. Asking CUTLASS itself
is 21000× dearer than the host formula, and it is only needed once — to prove
the formula right.

## 4.2 The query

`include/tilemega/Solver/BackendCostQuery.h`:

```cpp
bool isLegal(TargetSpec const&) const;
int threads() const;
int smemBytes() const;
ClusterShape clusterShape() const;
AlignmentRequirement alignmentRequirement() const;
int architectureRequirement() const;
std::optional<int> estimatedRegisters() const;   // tier 3 only
```

`estimatedRegisters()` returns `std::optional` and is empty until
`RecordPtxas()` is handed a real `-Xptxas=-v` log. This is not a placeholder:
CUTLASS's `constexpr` traits carry no register count and no closed form
predicts one, so an eagerly-filled field could only be a guess. The test
pins both halves of that contract:

```
ptxas: l1 kernel uses 167 registers
```

parsed from `test/fixtures/ptxas_gqa2_128x128x16s3.txt`, a real compile log of
the control configuration (l2 kernel 168, l1 167, stage kernel 145).

The closed form is locked to CUTLASS by three `static_assert`s in
`include/tilemega/Backend/CutlassGemmCandidate.h`:

```cpp
static_assert(kShapeLegal == SimtF32ShapeLegal(TileM, TileN, TileK, Stages));
static_assert(kThreads    == kSimtF32Threads);
static_assert(kSmemBytes  == SimtF32SmemBytes(TileM, TileN, TileK, Stages));
```

so the host answer cannot drift from the collective's own without failing to
compile. ✅ verified end to end: over the full 300-shape envelope the closed
form and the CUTLASS traits agree on **264/300 shape-legal** and **224/300
fitting**, exactly (`tier1_summary.txt` vs `backend_query_test`).

## 4.3 Tier 2 ranks, it does not judge

`CandidateGenerator::Analyze` is a padded-issue-cycle model and nothing more:

```
ctas       = ceil(M/tile_m) * ceil(N/tile_n)
co_resident= CoResidentPerSM(target)              // smem- and thread-bound
waves      = ceil(ctas / (num_sms * co_resident))
k_iters    = ceil(K / tile_k)
cta_cycles = tile_m * tile_n * tile_k / threads   // one FFMA/thread/cycle
cycles     = waves * k_iters * cta_cycles
```

It captures the one effect that dominates a decode-shaped model — padding
waste on a `seq = 4` token axis — and it does place the control far down the
list:

```
tier-2 cycles: best=8192 control(128x128x16s3)=524288
```

a 64× predicted penalty for the currently fixed `g`, which the oracle
confirms in direction (the control ranks 826/937 measured).

### Where it misses — reported as measured, not tuned

❌ **The tier-2 ordering is not good enough to be a filter.** Against the
oracle's measured optimum:

```
tier-2 rank of the measured best (32x32x32s3): 42 of 224
tier-2 top-8: 16x16x16s2 16x16x16s3 16x16x16s4 16x16x32s2
              16x16x16s5 16x16x32s3 16x16x32s4 16x16x32s5
```

and ✅ **all eight of that top-8 fail to compile.** The 80 tier-3 failures in
the joint sweep are exactly these 8 shapes × 5 split factors × 2 models, and
all 80 die at one site:

```
cute/int_tuple.hpp(890): error: no operator "<" matches these operands
    operand types are: const cute::C<0> < const cute::ArithmeticTuple<int,int>
```

reached from `CollectiveMma<MainloopSm80CpAsync<Stages,…>, Shape<_16,_16,_K>>`
instantiated by `GemmStageTaskBody::operator()`. ❌ inferred cause: when the
tile MN equals the 16×16 thread layout the per-thread predicate tensor
degenerates to rank 1 and CuTe's residue predication compares a scalar against
a rank-2 coordinate. ✅ verified consequence, which is the part that matters
here: **collective-level trait legality is not megakernel compilability.**
Tier 1 says these 8 shapes are legal, tier 2 puts them first, and only tier 3
finds out. That is a direct, measured justification for keeping tier 3 rather
than an argument from principle.

[!] **The model has no memory-traffic term.** A roofline term needs
`TargetSpec::Calib` (`bytes_per_cycle`, achieved bandwidth), whose fields are
all zero with `calibrated:false`, and `tools/tilemega-calibrate.cpp` is still
the `TODO(P4.1)` stub. Inventing constants would make the ranker look better
and mean nothing, so the gap is recorded instead. Closing P4.1 is the
prerequisite for tier 2 becoming a coarse *pre-filter*; today its honest role
is exactly what §4.3 states — ordering, so tier 3 can be spent on a top-k, and
the top-k must be verified by tier 3, never trusted.

## 4.4 How large is the space, really

Enumeration expands from the native SIMT tile `(16,16,8,2)` by doubling one
axis at a time in canonical order, so every shape is reached once with no
visited set. Shared storage is monotone in all four axes, which makes the
smem budget a subtree cut:

```
enumeration: touched=279 wall_pruned=19 rejected=36 legal=224 cartesian=300
```

For **one GEMM operator** on this target:

| stage | candidates | removed by |
|---|---|---|
| Cartesian product of the envelope | 300 | — |
| shape-legal | 264 | CuTe tile/thread divisibility (36) |
| fits shared memory | **224** | 101376 B budget (40) |
| compiles inside the megakernel | **216** | tier 3 (8) |

✅ verified: the walk visits 279 of the 300 shapes and never constructs the
other 21 — 19 nodes hit the wall and their subtrees (21 shapes) are cut
unexplored. 264 − 224 = 40 = 19 + 21 accounts for every smem rejection
exactly, and no candidate is rejected by a non-smem budget.

So the answer to "how many legal candidates does one GEMM operator actually
have" is **224 by traits, 216 in practice** — not 300, and not the Cartesian
product of an unbounded tile table. The generator never enumerates the
product: `cartesian=300` is computed as a counter for this table, while the
walk itself touches 279.

Split-K multiplies this per-operator space by the split factor at the *plan*
level, not the backend level (it is a host-side decomposition in
`ModelHarness.cuh::Create`), which is why the oracle sweep's own space is
1120 = 224 × 5 per model rather than 224.
