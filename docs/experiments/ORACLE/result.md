# Part 6 — the partition oracle

Fix Place and the event structure; sweep the GEMM granularity `g` exhaustively;
measure end-to-end latency per configuration on both reference models. `g` is
`(tile_m, tile_n, tile_k, stages, split_k)` applied uniformly to every GEMM
stage of the model. Everything else — the semantic stage list, the task
bodies, the L0.5/L1/L2 event structure, the buffer plan — is held fixed.

Reproduce with `run.sh` (needs the GPU; ~81 min wall). It resumes per tier from
`raw/`, so a re-run of one stage does not repeat the others.

Target: RTX 4090, sm_89, 128 SMs, 65536 registers/SM, 1536 threads/SM, 101376 B
opt-in dynamic smem per CTA, read from `TargetSpec::Probe()`.

Models: **gqa2** (2 layers, hidden 512, intermediate 1024, 4 heads / 2 kv heads,
head_dim 128, seq 4, past 3 — 14 GEMMs) and **mha4** (the same shape at 4
layers, 8 heads — 28 GEMMs). Control `g` = the value hard-coded before this
sweep, `128x128x16 stages=3 split_k=1`.

Timing protocol: screening = 3 fresh processes per configuration, minimum;
finals = **25 fresh processes**, median reported. `l05_ms` (per-stage kernel
launches) is the headline metric because it is the level all 1077 runnable
configurations share; L1 and L2 numbers are in the same tables.

---

## 6.3 The five questions

### Q1 — how much faster is the optimal `g` than the current fixed value?

| model | control `128x128x16s3k1` | best `g` | speedup |
|---|---|---|---|
| gqa2 | **1.106944 ms** | `16x64x32s3k16` — **0.181248 ms** | **6.11×** |
| mha4 | **2.192384 ms** | `16x64x16s2k16` — **0.324608 ms** | **6.75×** |

✅ verified — `raw/final_gqa2.tsv`, `raw/final_mha4.tsv`, median of 25 fresh
processes each, 25/25 PASS with the reference-matching output hash.

**The honest caveat: the optimum is a plateau, not a point.** Two independent
25-process replicates of the same finals set disagree on the winner
(`raw/replicate/final_gqa2_run1.tsv`): run A picks `16x64x32s3k16` at 0.181248
ms, run B picks `16x64x16s5k16` at 0.187392 ms, and run A's winner moves +9.04%
between the two. Per-configuration replicate spread across the 15 finalists:

| spread between replicates | configurations |
|---|---|
| < 1% | 4 |
| 1–3% | 6 |
| 3–8% | 4 |
| 9.04% | 1 (`16x64x32s3k16`, the run-A winner) |

The **control** is stable across replicates (1.106944 vs 1.112064, +0.46%), so
the 6.11×/6.75× headline is not at risk — but "which member of the top group"
is not resolvable at 25 processes. The defensible claim is *the top group is
6.1–6.8× faster than the control*, not *this exact tuple is optimal*.

A second, separate drift: the control's **screening** time (3 processes,
minimum) is 1.009664 ms on gqa2 against 1.106944 ms in the finals — 9.6%
apart, and in the direction the protocols predict (minimum-of-3 vs median-of-25).
On mha4 the same comparison is 2.190336 vs 2.192384, 0.1% apart. Screening
numbers are used only to rank; every number quoted as a result comes from the
finals.

### Q2 — what is the optimal configuration, and which operators carry the difference?

The whole difference is in the GEMM stages. Per-stage attribution
(`raw/stages_*.txt`, minimum over runs, one stage timed at a time):

| gqa2 | control | best | |
|---|---|---|---|
| Gemm (14) | 1.0639 ms (93.1%) | 0.0899 ms | |
| SplitKReduce (14) | — | 0.0508 ms | new work the split introduces |
| **GEMM total** | **1.0639 ms (93.1%)** | **0.1407 ms (67.6%)** | **7.56×** |
| RMSNorm (4) | 0.0213 | 0.0173 | |
| RoPE (4) | 0.0167 | 0.0153 | |
| KVAppend (4) | 0.0165 | 0.0147 | |
| Attention (2) | 0.0151 | 0.0126 | |
| Elementwise (2) | 0.0092 | 0.0075 | |
| **non-GEMM total** | **0.0789 ms (6.9%)** | **0.0674 ms** | **1.17×** |

| mha4 | control | best |
|---|---|---|
| Gemm (28) | 1.9452 ms (93.0%) | 0.1945 ms |
| SplitKReduce (28) | — | 0.1229 ms |
| **GEMM total** | **1.9452 ms (93.0%)** | **0.3174 ms (67.0%)** — **6.13×** |
| **non-GEMM total** | **0.1453 ms (7.0%)** | **0.1562 ms** — **0.93×** |

✅ verified. Two things to read off this:

1. **Every GEMM improves, none regresses.** gqa2: 14/14, from 5.96× (`gemm12`,
   the second layer's up projection) to 12.14× (`gemm13`, the second layer's
   down projection — the largest contraction, K = intermediate = 1024).
   mha4: 28/28, 4.90×–10.00×. There is no operator for which the control tile
   is the right choice, which is why a *single* uniform `g` still wins by 6×
   here and why a per-operator `g` is the obvious next question.
2. **Non-GEMM cost is flat, and on mha4 slightly worse** (0.1453 → 0.1562 ms,
   −7%). The best configuration runs at 2 CTAs/SM against the control's 1, and
   the memory-bound stages do not benefit; on mha4 they measurably lose. This
   is recorded, not smoothed: an oracle that optimizes GEMM granularity alone
   pays a small tax on everything else.

After optimization the GEMMs are 67% of the time rather than 93%, so the
remaining headroom is no longer concentrated in one operator class.

**Shape of the winners.** Among the 34 (gqa2) / 37 (mha4) configurations within
10% of best, `split_k ∈ {8, 16}` in *all* of them (gqa2 24×k16 + 10×k8; mha4
24×k16 + 13×k8), `tile_m ∈ {16, 32}` in all but one, and `tile_n = 64`
dominates (23/34, 24/37). `stages` is nearly free: all four values appear in
the top group. The control is wrong on three of the five axes at once — its
tile is 8× too large in M, 2× too large in N, and it does not split K.

### Q3 — how large is the legal candidate space, and how much survives pruning?

| tier | test | gqa2 | mha4 |
|---|---|---|---|
| 0 | enumerated tile shapes (5 `tile_m` × 5 `tile_n` × 3 `tile_k` × 4 `stages`) | 300 | 300 |
| 1a | CUTLASS `constexpr` shape legality | 264 | 264 |
| 1b | smem ≤ 101376 B | **224** | **224** |
| — | × 5 split factors `{1,2,4,8,16}` | **1120** | **1120** |
| 3 | real nvcc + ptxas compiles | 1080 (**40 fail**) | 1080 (**40 fail**) |
| 4 | runs to completion | 1077 (**3 hang**) | 1077 (**3 hang**) |
| 5 | bitwise-correct output | **1077** | **1077** |

✅ verified — `raw/tier1_summary.txt`, `raw/tier3_summary.txt`,
`raw/screen_*.tsv`. 2240 compiles total across both models: 2160 OK, 80 FAIL.

**Tier 1a rejects exactly 36 shapes, and the rule is one predicate:**
`min(tile_m, tile_n) == 16 && tile_k == 8`. Tier 1b rejects 40 more, all
large-tile × high-stages combinations that exceed the opt-in smem budget.

**Split-K is not prunable at tier 1.** `split_k` is a granularity choice on the
host, not a backend trait — its legality is `split_k ≤ k_tiles`, clamped at run
time. So the 224 surviving shapes multiply out by the full 5 rather than being
filtered. This is a real limit of the traits tier as specified in Part 4: it
prunes the axes the collective knows about, and the split factor is not one.

**Pruning removes 25.3% of the tier-3 budget.** 1120 of 1500 configurations
reach a compiler; without the traits tier, 1500 would. That is a real saving
and a modest one, precisely because the unprunable split axis multiplies the
survivors back up by 5. But see Q4 — the compile is not where the wall time
went anyway.

### Q4 — total compile + measurement cost

`raw/wall_time.txt`, both models together, 32-way compile parallelism, 112 CPUs:

| stage | wall |
|---|---|
| tier 1 — build the traits probe TU | 19.61 s |
| tier 3 — 2240 nvcc + ptxas compiles | 1893.85 s |
| screening — 2160 binaries × 3 fresh processes | 2739.73 s |
| finals — 28 binaries × 25 fresh processes | 193.37 s |
| **total** | **4846.56 s ≈ 1 h 20 min** |

✅ verified. Two observations that matter for Phase 4:

- **Measurement, not compilation, is the bottleneck** — 2739.73 s of screening
  against 1893.85 s of compiling, and the screening is *already* the cheapest
  useful protocol (3 processes, minimum). The GPU is serial by construction:
  every configuration needs exclusive use of the device. Compilation scales to
  32 ways on this machine; measurement scales to 1.
- Serially the compile alone would be **13.2 CPU-hours** (2240 × 21.2 s
  measured single-compile wall). 21.2 / 0.845 = 25.1 of a possible 32 is the
  parallel efficiency actually achieved.

### Q5 — what shape is the performance distribution?

**少数配置显著好.** Sharply peaked, with a broad mediocre bulk and a very long
bad tail.

| within … of best | gqa2 (of 1077) | mha4 (of 1077) |
|---|---|---|
| 1.05× | **10** (0.9%) | **10** (0.9%) |
| 1.10× | 34 (3.2%) | 37 (3.4%) |
| 1.25× | 152 (14%) | 146 (14%) |
| 1.50× | 342 (32%) | 331 (31%) |
| 2.00× | 561 (52%) | 528 (49%) |

| | gqa2 | mha4 |
|---|---|---|
| best | 0.171904 ms | 0.324832 ms |
| median | 0.338208 ms (1.97× best) | 0.657440 ms (2.02× best) |
| worst | 255.93 ms (**1489× best**) | 510.74 ms (**1572× best**) |
| control rank | **951 / 1077** | **960 / 1077** |

✅ verified — screening tables, minimum of 3 processes; used here for
distribution shape only, not for the headline speedups.

Reading: a random legal `g` costs about **2× the optimum**, so autotuning is
not the difference between working and not working. But the top 1% is a
genuinely distinct group, the tail reaches three orders of magnitude, and the
hand-picked control landed in the **88th percentile** — worse than 88% of the
space it was chosen from. The distribution is exactly the shape that makes a
cost model worthwhile and a cost model's *precision* nearly irrelevant: landing
anywhere in the top 3% is within 10% of optimal.

**Tile shape vs split-K, separated.** Holding `split_k = 1` and optimizing the
tile alone recovers most but not all of the win:

| model | control | best tile at `k=1` | tile-only | + split-K | total |
|---|---|---|---|---|---|
| gqa2 | 1.009664 | 0.343584 | **2.94×** | **2.00×** | 5.87× |
| mha4 | 2.190336 | 0.693568 | **3.16×** | **2.14×** | 6.74× |

(screening numbers, so all three columns are on one protocol.) Both axes are
first-order. A cost model that models tiles but not the split factor captures
roughly half the available speedup in log terms.

---

## 6.4 Verdict

The skeleton's decision rule: >10% → a full cost model + DP is worth building;
3–10% → worth building but simplifiable; <3% → Phase 4 needs redesign.

**611% (gqa2) and 675% (mha4). The full cost model + DP is warranted**, with
two qualifications that the data forces and that change what should be built:

1. **Aim for the plateau, not the point.** The top ~34 configurations are
   inside a ±10% band that a 25-process median cannot rank reliably (two
   replicates disagree on the winner; the run-A winner moves 9.04%). A cost
   model that lands *in* the top 3% has captured essentially all of the
   available win. Investing in a model precise enough to pick the exact
   argmin would be spending on a distinction the hardware does not make.
2. **The split factor must be in the model.** It is 2.00×/2.14× on its own,
   it is present in every top-10% configuration, and it is invisible to the
   tier-1 traits query. Part 4's pruning tier prunes the wrong axis if the
   split factor is left to a fixed default.

A third observation, weaker but recorded: 93% of the control's time was GEMM
and 67% of the optimum's is, so a *second* round of the same exercise would
have far less to work with. The 6× is a one-time correction of a badly chosen
default, not evidence that continued granularity search keeps paying.

---

## 6.5 What the sweep found in the implementation

Three bugs, all found by the sweep and none visible at the control point. They
are the reason 143 configurations initially failed, and the reason the
pre-fix evidence is preserved in `raw_gridbound/` rather than deleted.

### F-36 — `tasks ≤ grid` was an unstated precondition

Before this sweep, `GemmStageTaskBody` read `int task = blockIdx.x;` — one task
per CTA, no grid stride. At the control `g` no stage has more tasks than the
resident grid, so the assumption held silently. Small tiles and split-K break
it: `16x32x16s2k16` on gqa2 has a gate/up GEMM with **512 tasks** against a
resident grid of **384**, and 128 tasks were simply never executed.

✅ verified, before and after, same config, same binary flags:

| | grid | ctas/SM | smem | result |
|---|---|---|---|---|
| pre-fix `gqa2_16x32x16s2k16` | 384 | 3 | 6400 | `MISMATCH l05_vs_l0_mismatch=4090` |
| post-fix, same config | 384 | 3 | 6400 | `PASS` |

Resource use is identical, so the grid stride costs nothing here. **141 of the
143 pre-fix RUNFAILs recovered**; 2 remained (16×16-tile compile failures, F-37)
and 1 new failure appeared (F-38). `raw_gridbound/screen_gqa2.tsv` and
`raw_gridbound/fail_grid.txt` hold the pre-fix run.

### F-37 — trait legality is not compilability

80 configurations (8 tile shapes × 5 splits × 2 models) pass tier-1 constexpr
legality *and* the smem budget, then fail nvcc. All eight are 16×16 tiles:
`16x16x{16,32}s{2,3,4,5}`. The error is in the **mainloop**, not the epilogue:

```
cute/int_tuple.hpp(890): error: no operator "<" matches these operands
    operand types are: const cute::C<0> < const cute::ArithmeticTuple<int, int>
```

instantiated through `cute::copy_if` → `CollectiveMma<MainloopSm80CpAsync<…>>`
→ `GemmStageTaskBody::operator()`. ✅ verified from `raw/log/*.ptxas`.

This is the empirical justification for Part 4's tier 3 existing at all: 8 of
the 224 trait-legal, smem-legal shapes (**3.6%**) do not compile, and no
host-side query predicts which 8.

### F-38 — a persistent spin-wait kernel must run at its *own* resident grid

3 configurations per model hung rather than failing: `128x16x32s2k{4,8,16}`.
Traced with gdb to `cudaEventSynchronize` inside `LaunchL2`, then to the
register counts:

| kernel | registers | CTAs/SM | resident grid |
|---|---|---|---|
| `tilemega_l1_kernel` | 128 | 2 | 256 |
| `tilemega_l2_kernel` | **144** | **1** | **128** |

The harness sized one grid from L1 and launched both kernels at it. L2 at grid
256 has only 128 resident CTAs; the resident ones spin forever on arrivals that
only the non-resident ones can make. ✅ The predicate
`occ(l2) < occ(l1) ∧ max_stage_tasks > resident_l2` is **exact over all 1080
configurations on both models**: `occ(l2) < occ(l1)` holds for exactly the five
`128x16x32s2k*` configurations (`raw/occupancy_l1_l2_*.tsv`), and exactly the
three whose largest stage exceeds 128 tasks hang — k=4 → 256, k=8 → 512,
k=16 → 1024 tasks, against k=1 → 64 and k=2 → 128, which pass.

Fixed in `ModelHarness.cuh` by taking the grid as the **minimum** over both
persistent kernels' resident grids. ✅ verified, all ten affected
configurations (`raw/f38_verify.tsv`):

| model | config | pre-fix | post-fix |
|---|---|---|---|
| gqa2 | `128x16x32s2k1` | PASS, grid 256, 0.655200 | PASS, grid **128**, 0.667648 |
| gqa2 | `128x16x32s2k2` | PASS, grid 256, 0.417792 | PASS, grid **128**, 0.428032 |
| gqa2 | `128x16x32s2k4` | **hang** | **PASS**, grid 128, 0.330880 |
| gqa2 | `128x16x32s2k8` | **hang** | **PASS**, grid 128, 0.330592 |
| gqa2 | `128x16x32s2k16` | **hang** | **PASS**, grid 128, 0.376032 |
| mha4 | `128x16x32s2k1` | PASS, grid 256, 1.282848 | PASS, grid **128**, 1.177600 |
| mha4 | `128x16x32s2k2` | PASS, grid 256, 0.808960 | PASS, grid **128**, 0.744448 |
| mha4 | `128x16x32s2k4` | **hang** | **PASS**, grid 128, 0.562176 |
| mha4 | `128x16x32s2k8` | **hang** | **PASS**, grid 128, 0.618592 |
| mha4 | `128x16x32s2k16` | **hang** | **PASS**, grid 128, 0.710976 |

All ten reproduce the recorded per-split output hashes (gqa2 k=1
`5245714bc5d3ab4d`, k=8 `b37a2e7f0ec44cb5`, k=16 `8a737188b958a2ae`; mha4 k=1
`fd15fa2e89cdb915`, k=8 `118f8e253b29f694`, k=16 `472fe4e8596656af`), so the
smaller grid changes scheduling and nothing else.

**These five configurations are the only ones the fix changes**, because
`occ(l2) < occ(l1)` holds nowhere else in either model's 1080. The screening
and finals tables in `raw/` were produced *before* the fix, and are **not**
re-measured: the best of the five now lands at rank 512/1077 (gqa2) and
415/1077 (mha4), nowhere near the top-8 finals set, so no headline number
moves. Stated explicitly rather than left implicit — the tables carry the
pre-fix grid for these five rows, and `raw/f38_verify.tsv` carries the
post-fix ones.

---

## 6.6 The occupancy closed form, validated on the sweep

Part 4's tier-1 query predicts CTAs/SM without compiling. The sweep is 1077
real measurements per model to check it against:

```
ctas_per_sm = min( ⌊65536 / (8 · ⌈regs · 32 / 256⌉ · 256)⌋,   // register term
                   ⌊101376 / smem⌋,                            // smem term
                   ⌊1536 / 256⌋ )                              // thread term
```

(sm_89 numbers from `TargetSpec::Probe()`: 65536 registers/SM, per-warp
allocation granularity 256, 256 threads/CTA, 101376 B opt-in smem, 1536
threads/SM.)

| | gqa2 | mha4 |
|---|---|---|
| candidates | 1077 | 1077 |
| **matches the harness** | **1077** | **1077** |
| misses | 0 | 0 |
| register-bound | 605 | 605 |
| smem-bound | 150 | 150 |
| both terms tie | 322 | 322 |
| thread-capped | 0 | 0 |
| wrong if the smem term is dropped | 605 | 605 |
| wrong if the register term is dropped | 150 | 150 |

✅ verified — `occupancy.sh`, pure re-analysis of `raw/`, no GPU and no compile.
Both terms bind on a real candidate space; neither is redundant. The thread
term never binds at 256 threads/CTA but is kept because it would at 512.

The register and smem footprints are **identical between gqa2 and mha4** for
every one of the 1077 shared configurations. That is a property of the task
body and the smem union — the code is per-`g`, not per-model — and it is why
Part 4's query takes a `g` and a target, not a model.
