# P4.1 — target calibration

`tools/tilemega-calibrate` measures the machine it runs on and writes
`configs/targets/sm_XX.json`. Nothing in that file is a datasheet number, a
vendor claim or an estimate: every field is either measured on the device or
left at zero with a flag saying it was not measured.

```
bash docs/experiments/CALIB/run.sh                  # calibrate this machine
EXPECT=sm_90 bash docs/experiments/CALIB/run.sh     # refuses on anything else
```

- Hardware: NVIDIA GeForce RTX 4090 (sm_89), 128 SMs, driver-reported pin
  bandwidth 1008.1 GB/s.
- Wall time: **6.95 s** for a full run (`--repeats 41`), of which 2.0 s is
  clock warm-up. The brief asked for minutes; this is seconds.
- 55 measurement records, each carrying its sample count, relative standard
  deviation and a one-line description of the method.
- Tracked evidence: the 55 records live in
  [`configs/targets/sm_89.json`](../../../configs/targets/sm_89.json) under
  `calibration.measurements`, each with its method string. `run.sh` also writes
  `raw/sm_89.log`, which `.gitignore` excludes along with every other
  experiment log.

## What is calibrated, and what it is for

Only quantities the cost model (Part 2) and the DP (Part 4) actually consume
are measured. Nothing was calibrated "for completeness".

### (a) Per-pipeline rates — the resource vector `u(o)`

| quantity | value | method | n | rsd | run-to-run |
|---|---|---|---|---|---|
| `tc_fp16_gflops` | 180082 GFLOP/s | `mma.sync.m16n8k16` chains at full occupancy | 41 | 0.04% | 0.50% |
| `cuda_fp32_gflops` | 72316 GFLOP/s | independent FFMA chains, 8 per thread | 41 | 0.32% | 0.41% |
| `cuda_int32_gops` | 22231 GIMAD/s | IMAD chains, unroll-resistant operands | 41 | 0.18% | 0.35% |
| `sfu_exp2_gops` | 5500 Gop/s | `ex2.approx.f32` chains | 41 | 0.11% | 0.55% |
| `sfu_rsqrt_gops` | 5500 Gop/s | `rsqrt.approx.f32` chains | 41 | 0.12% | 0.38% |
| `smem_gbps` | 64183 GB/s | conflict-free 32-bit LDS at full occupancy | 41 | 0.13% | 0.52% |
| `smem_conflict_slope` | 0.4894 ×/way | least squares through the origin of t(w)/t(1) | 164 | 0.78% | 0.50% |
| `l2_gbps` | 5127 GB/s | plateau of the working-set sweep | 738 | 0.98% | 0.50% |
| `dram_gbps` | 982 GB/s | same sweep, 512–2048 MiB (**achieved**, not peak) | 738 | 0.98% | 0.02% |
| `l2_knee_bytes` | 75497472 B | largest working set still within 95% of the plateau | 738 | 0.98% | 0.00% |

The bank-conflict measurement is a curve, not a single number: 2/4/8/16/32-way
ratios are 1.0006 / 2.003 / 3.926 / 7.849 / 15.66 against conflict-free. The
2-way point costs nothing, so the slope is fitted only over the ways whose
ratio exceeds 1.05, and the model applies `max(1, slope × ways)`.

Two independent cross-checks on the memory hierarchy:

- `l2_knee_bytes` = 75497472 B = **72 MiB exactly**, which is the AD102 L2
  capacity. The sweep found the capacity without being told it.
- `dram_gbps` / pin bandwidth = **97.4%**. A streaming read that hits 97% of
  the pin rate is at the achievable ceiling, so the number is a rate and not a
  latency artefact.

### (b) Synchronisation latency

| quantity | value | method | n | rsd | run-to-run |
|---|---|---|---|---|---|
| `atomic_uncontended_ns` | 117.5 ns | one CTA, dependent `atomicAdd` chain | 41 | 0.16% | 0.43% |
| `atomic_contention_ns` | curve, below | N CTAs on one address, per completed atomic | 410 | 0.21% | — |
| `threadfence_ns` | 358.1 ns | `__threadfence()` loop minus the same loop without it | 41 | 0.08% | 0.56% |
| `syncthreads_ns` | 145.1 ns | `__syncthreads()`, same differential, 256 threads | 41 | 1.01% | 1.03% |
| `named_barrier_ns` | 144.5 ns | `bar.sync 1`, same differential | 41 | 1.09% | 1.46% |
| `cluster_sync_ns` | **uncalibrated** | needs a cluster launch — see below | — | — | — |

Contention curve, nanoseconds per **completed** atomic:

| CTAs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| ns/op | 117.5 | 61.0 | 30.5 | 15.45 | 7.72 | 3.84 | 1.92 | 0.95 | 0.48 | 0.43 |

Contention on a single address is not additive on this part: throughput rises
almost perfectly linearly to 256 CTAs and only then saturates at 0.43 ns/op.
The L2 atomic unit pipelines same-address updates, so `T_sync` for a split-K
reduction is dominated by the fixed latency, not by the peer count — which is
what makes split-K by 16 affordable at all.

`named_barrier_ns` and `syncthreads_ns` agree to 0.4%: a named barrier over all
warps of a CTA costs the same as `__syncthreads()`, so warp specialisation buys
nothing here beyond what the partition itself buys.

### (c) Stream-K four parameters

`time_CTA = a + b·[peers>1] + c·iters + d·(peers−1)` (arXiv 2301.03598 A.1).

TileMega's reduction is a separate `GemmCombine` stage that does not depend on
the GEMM tile shape at all, so `b` and `d` are held **per output element** and
fitted once for the whole target, while `a` and `c` are fitted per shape:

```
time_CTA = a + c·iters + (b + d·(peers−1))·live_outputs(tile) + fixed/ctas
```

| shape | `a` (ns) | `c` (ns/iter) | r² | `a` run-to-run |
|---|---|---|---|---|
| 128×128×16 s3 | 2638.2 | 1733.5 | 0.999984 | 6.6% |
| 64×64×16 s3 | 1329.8 | 874.1 | 0.999968 | 11.0% |
| 32×32×32 s3 | 989.4 | 864.1 | 0.999920 | 17.7% |
| 16×64×32 s3 | 1034.5 | 1088.2 | 0.999940 | 13.6% |
| 16×64×16 s2 | 923.6 | 577.8 | 0.999876 | 12.4% |
| 256×128×16 s3 | 4531.3 | 2881.0 | 0.999993 | 2.3% |

| shape-independent | value | run-to-run |
|---|---|---|
| `b` (`combine_base_ns`) | 0.002443 ns/element | 1.5% |
| `d` (L2 regime) | 0.000817 ns/peer/element | 2.1% |
| `d` (DRAM regime) | 0.004499 ns/peer/element | 0.12% |
| `combine_fixed_ns` | 46 ns — **unresolved**, see below | 150% |

Two cross-checks that the reduction fit is measuring memory and not the timer:

- `d` in the L2 regime is 0.000817 ns per peer per element. One element is
  4 bytes, so that is **4894 GB/s** — against `l2_gbps` = 5127 GB/s measured
  independently by the working-set sweep. 4.5% apart.
- `d` past the knee is 0.004499 ns, i.e. **889 GB/s**, against `dram_gbps` =
  982 GB/s. 9% apart, in the right direction for a read-modify pattern.
- `b` = 0.002443 ≈ 3.0 × `d`(L2), where "one partial read plus one output
  store" predicts 2×. The extra is the epilogue's `beta·C` read.

**The 24–26% SplitKReduce share is *not* explained by this form — measured.**

For gqa2's winning configuration (`16×64×32 s3`, split 16), summing
`(b + d·(chunks−1)) · 4N + fixed` over the 14 GEMM stages predicts **1.13 µs**
of reduction. ORACLE's per-stage attribution measures **50.8 µs**
(`docs/experiments/ORACLE/result.md`), which against that configuration's
0.2081 ms of attributed stage time is the 24% figure. The traffic terms account
for **2%** of the measured reduction cost.

Per stage that is 3.63 µs on gqa2 (14 stages) and 4.39 µs on mha4 (28 stages,
0.1229 ms) — within 21% of each other across two models whose reduction traffic
differs by much more than that. The reduction's cost is therefore essentially
**fixed per stage**, not proportional to peers × outputs. At M = 4 there is
almost nothing to move: the winner reduces 4 × 512 outputs over 16 peers, i.e.
128 KB, which at the measured L2 rate is 25 ns.

`b` and `d` are not wrong — both cross-check against independently measured
bandwidths to within 4.5% and 9% (above). They are the wrong terms to be
*dominant* at this problem size. What is missing is a per-stage fixed cost: the
megakernel's stage barrier and the wave quantisation of a 128-CTA reduction,
neither of which the per-CTA four-parameter form carries. The calibrated
synchronisation constants are the ingredients for it — `threadfence_ns` 358 ns,
`atomic_uncontended_ns` 117.5 ns, `syncthreads_ns` 145 ns — and building that
term is Part 2's job, not this one's.

`combine_fixed_ns` was **not** raised from 46 ns to 3.6 µs to close the gap.
The microbenchmark measures a resident-grid reduction with no stage barrier and
that is what it reports; the missing 3.6 µs is structural, and inventing it
here would hide exactly the thing Part 2 has to model.

### (d) Concurrency interference — the assumption is **falsified**

| quantity | value | n | rsd | run-to-run |
|---|---|---|---|---|
| `interference_neighbour_gbps` | 633.8 GB/s | 41 | 0.22% | — |
| `interference_ratio` | **1.518** | 41 | 0.82% | 0.35% |

A DRAM-bound neighbour at one CTA per SM — small enough that a GEMM CTA can
always co-reside, because a hog that fills the machine measures serialisation
rather than interference — slows the GEMM from 18.3 µs to 27.6 µs.

The skeleton's own gate reads *"干扰偏差 > 30% 则独立时长假设失效"*. The measured
deviation is **51.8%**. The independent-duration assumption is therefore
**false on this target**, and Part 2's `T_steady` cannot be a sum of
per-operator durations measured in isolation. This is a measurement, not an
assumption: it was the one quantity the brief explicitly said to measure rather
than assume, and it came back over the threshold.

## Machine-idle guard

Calibration now refuses to certify a run taken on a busy GPU.

This was not a precaution. Five consecutive runs produced two that recorded
half the DRAM bandwidth (465 vs 982 GB/s), half the tensor-core rate (89 vs
180 TFLOP/s), a third of the L2 capacity (24 vs 72 MiB) and double the
bank-conflict slope — while `cuda_fp32_gflops` did not move at all, ruling out
a clock drop. The GPU is shared: `nvidia-smi` showed 100% utilisation and
425 W with **"no running processes found"**, because the co-tenant is in
another container.

Two checks, because they fail on different neighbours:

- **achieved vs pin rate** — `dram_gbps` against `2 × memory clock × bus width`
  from `cudaDeviceGetAttribute`. Catches a neighbour present the whole run,
  which leaves no drift behind. Clean: 97.4%. Contended: 46.1%. Threshold 70%.
- **start vs end drift** — the largest working set of the sweep re-measured
  after every other group. Catches one that arrived or left partway through.
  Clean: 0.03%. Contended: 111%. Threshold 5%.

Verified by reproducing the failure deliberately: with a 512 MiB read-modify
kernel running in another process, the guard reports 46.1% of peak, 111% drift
and writes `"calibrated": false`. Without it, `"calibrated": true` at 97.4%
and 0.03%.

## Uncalibrated, and why

| target | state | reason |
|---|---|---|
| `sm_89` | **calibrated** | measured here |
| `sm_80` | uncalibrated | no A100 on this machine |
| `sm_90` | uncalibrated | no H100 on this machine |
| `sm_120` | uncalibrated | no Blackwell on this machine |
| `cluster_sync_ns` (all targets) | uncalibrated | needs a cluster launch, which lands with Part 7 |

The three foreign profiles carry the same schema with every measured field at
zero and `"calibrated": false`. No value in them is a guess, an interpolation
from sm_89 or a datasheet figure. `run.sh` hard-fails when `EXPECT` does not
match the probed architecture, so the only way to fill one in is to run it on
that hardware:

```
$ EXPECT=sm_90 bash docs/experiments/CALIB/run.sh
this machine is sm_89, not sm_90; refusing to write configs/targets/sm_90.json
run this script on sm_90 hardware -- sm_90 stays uncalibrated
```

`cluster.sync()` cannot be measured even on a cluster-capable target yet: the
probe kernel needs a `cudaLaunchKernelEx` cluster launch, which is Part 7's
work. The field stays 0 with `cluster_sync_calibrated: false` rather than
carrying a plausible-looking number.

## Detours and rejected measurements

Every constant was checked against a physical bound before acceptance. Six
measurements were rejected during this work, all of them for producing numbers
that could not be true or that were contaminated by the harness.

1. **A 33 TB/s reduction read.** The partials buffer was `cudaMemset(0)`, and a
   zero-filled buffer is compressible: the memory system never fetched it.
   Replaced by `FillKernel`, which writes a per-element hash in [1, 2).

2. **A 38 TB/s reduction read, after fixing (1).** The subtracted `NullKernel`
   grid was sized to the sweep width, so the subtraction scaled with the
   signal. Block dispatch *overlaps* memory traffic rather than adding to it,
   so subtracting a same-sized launch removed most of the traffic term.
   Replaced by one fixed grid of `2 × num_sms` for every point — which is also
   how the megakernel runs the stage, as a resident grid draining a queue.

3. **A negative `b` at r² = 0.03.** `b` was the *intercept* of a peer sweep, an
   extrapolation below the smallest measured point. Now measured directly at
   `chunks = 1`.

4. **`d` differing 8× between widths** (0.00057 ns/elem at 8 MB, 0.0045 at
   128 MB). Not noise: those are the L2 and the DRAM rate for a 4-byte read.
   The reference models reduce at most 1 MB of partials, so `d` is fitted in
   the L2 regime and the DRAM-regime value is recorded beside it as
   `combine_d_dram_ns`. The two are never averaged.

5. **`interference_ratio` swinging 0.90–2.50 across runs.** Two causes, both
   fixed. The victim GEMM at N = 512, K = 512 ran a single MAC iteration per
   chunk, so 5 µs of each 5 µs launch was launch overhead and the ratio was
   measuring interference on the launch path — enlarged to N = 1536, K = 2048.
   And the clock: the Stream-K stage is host-bound between short kernels, so
   the SM clock falls back towards its 210 MHz idle and the two halves of a
   pair landed at different points on the ramp. A second 1 s warm-up
   immediately before the stage brought it to 1.51 ± 0.007 over six runs, and
   1.50–1.51 over three runs started from a fully idle GPU.

6. **An 8-minute hang at 100% GPU.** `MemoryHogKernel` was a resident kernel
   spinning on a host-cleared flag. `TimeMs` warms up with
   `cudaDeviceSynchronize()`, which waits for *all* streams regardless of
   stream flags, so the flag write queued behind the very kernel it existed to
   stop. The design was abandoned rather than patched: the hog now runs a
   bounded pass count, and a `ShortNeighbour` guard re-runs with more passes if
   the neighbour finished before the window it was meant to disturb — which
   proves, rather than assumes, that the disturbed window was disturbed
   throughout.

Two quantities remain imperfectly resolved, and are recorded as such rather
than polished:

- **`a` (Stream-K prologue + epilogue), 2–18% run to run.** `a` is a few
  hundred nanoseconds to a few microseconds, the same order as the launch path
  subtracted from every point it is fitted through. Two fixes were applied.
  Extending the sweep down to a single MAC iteration (from `{4…64}` to
  `{1,2,4,8,16,32,64}`) shortened the extrapolation. Pairing the launch
  subtraction with each point, instead of measuring it once before the sweep,
  removed the systematic shift — measured once, that estimate's own jitter
  lands undiluted on the intercept, because shifting every point equally *is*
  shifting the intercept. Before: −355 to +2882 ns, three of six shapes
  negative. After: all six positive, monotone in tile area, worst spread 17.7%
  on the smallest tile. `c`, which is what the model multiplies by the
  iteration count, is stable to 0.66%, and the per-shape fits sit at
  r² ≥ 0.99988. A negative `a` is unphysical and none is shipped.

- **`combine_fixed_ns`, 150% run to run and either sign.** This is the
  intercept of a width sweep and is smaller than the launch latency it sits on;
  the median over five runs is ~110 ns. The profile carries `max(0, fit)` and
  the raw fitted value stays in `measurements` as evidence. It is not tuned to
  a convenient number and it is not claimed to be resolved.

Detours from earlier in this work, kept for the record: nvcc constant-folds
affine integer chains under unrolling, which silently deleted the IMAD
benchmark until the operands were made unroll-resistant; `cudaStreamCreate`
produces a blocking stream that serialises against a resident kernel while
`cudaStreamCreateWithFlags(…, cudaStreamNonBlocking)` does not; and AD102
delivers ~1.48 scalar LDS warp-instructions per clock per SM, which is the
number behind `smem_gbps` and the reason the SIMT f32 path's scalar
`Copy_Atom<DefaultCopy, float>` is a real cost rather than a detail.

## Reproducibility

Five consecutive full runs on an idle GPU, all of which passed the machine-idle
guard. "Spread" is (max − min) / median.

| quantity | median | unit | spread |
|---|---|---|---|
| `tc_fp16_gflops` | 180085 | GFLOP/s | 0.50% |
| `cuda_fp32_gflops` | 72315.6 | GFLOP/s | 0.41% |
| `cuda_int32_gops` | 22231.3 | GIMAD/s | 0.35% |
| `sfu_exp2_gops` | 5495.77 | Gop/s | 0.55% |
| `sfu_rsqrt_gops` | 5494.72 | Gop/s | 0.38% |
| `smem_gbps` | 64081 | GB/s | 0.52% |
| `smem_conflict_slope` | 0.489677 | x/way | 0.50% |
| `l2_gbps` | 5111.81 | GB/s | 0.50% |
| `l2_knee_bytes` | 7.54975e+07 | B | 0.00% |
| `dram_gbps` | 981.856 | GB/s | 0.02% |
| `atomic_uncontended_ns` | 119 | ns | 0.43% |
| `threadfence_ns` | 360 | ns | 0.56% |
| `syncthreads_ns` | 145.5 | ns | 1.03% |
| `named_barrier_ns` | 145.859 | ns | 1.46% |
| `cluster_sync_ns` | 0 | ns | 0.00% |
| `combine_fixed_ns` | 111.921 | ns | 149.93% |
| `combine_d_dram_ns` | 0.00450346 | ns/peer/elem | 0.12% |
| `interference_ratio` | 1.50785 | ratio | 0.35% |
| `a_ns[128x128x16s3]` | 2629.89 | ns | 6.56% |
| `c_ns[128x128x16s3]` | 1734.5 | ns/iter | 0.59% |
| `a_ns[64x64x16s3]` | 1328.18 | ns | 10.99% |
| `c_ns[64x64x16s3]` | 875.463 | ns/iter | 0.60% |
| `a_ns[32x32x32s3]` | 1102.53 | ns | 17.73% |
| `c_ns[32x32x32s3]` | 863.122 | ns/iter | 0.52% |
| `a_ns[16x64x32s3]` | 1245.24 | ns | 13.56% |
| `c_ns[16x64x32s3]` | 1084.12 | ns/iter | 0.37% |
| `a_ns[16x64x16s2]` | 1069.79 | ns | 12.38% |
| `c_ns[16x64x16s2]` | 575.124 | ns/iter | 0.66% |
| `a_ns[256x128x16s3]` | 4561.84 | ns | 2.25% |
| `c_ns[256x128x16s3]` | 2876.29 | ns/iter | 0.64% |
| `b_ns (all shapes)` | 0.00240925 | ns/elem | 1.48% |
| `d_ns (all shapes)` | 0.000816498 | ns/peer/elem | 2.11% |

