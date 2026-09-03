# P4.4 — analytical cost model, validated on the 2154 measured points

Reproduce: `bash docs/experiments/COST_MODEL/run.sh` (no GPU needed; a few
seconds). Inputs are all committed: `ORACLE/raw/screen_{gqa2,mha4}.tsv` for the
measured latencies, `ORACLE/raw/log/*.ptxas` for the register counts,
`E2E_GEN/raw/generated_e2e.cu` and `P3_GENERALIZATION/raw/generated.cu` for the
model descriptions, `configs/targets/sm_89.json` for the hardware constants.

## 1. Why §2.1's roofline was replaced

✅ The tier-2 analytical ranking (`CandidateGenerator::RankKey`) reaches
Spearman **0.461 / 0.446** against the two measured sets, places **0 of its top
10** inside the measured top 3%, and ranks the measured optimum **95th / 191st
of 1077**. That is the baseline row in every table below. A roofline over the
same resource constants (the `roofline` ladder rung) does no better: ρ 0.444 /
0.430.

## 2. What the model is

`CostModel::Evaluate` walks the generated stage list. Per stage:

- **GEMM.** `chunks = min(split_k, k_tiles)`; `ctas = ceil(seq/Tm)·ceil(n/Tn)·chunks`;
  `iters = ceil(k_tiles/chunks)`. The CTAs are decomposed into waves of
  `num_sms·ctas_per_sm`, and **the tail wave is re-evaluated at its own active-SM
  count** `o = active/num_sms` (§2.2(d)) rather than charged an additive
  quantisation term.
- **Steady state** is `max` over the resource lanes
  `⟨t_TC, t_CUDA, t_SFU, t_SMEM, t_L2, t_DDR⟩` (§2.2(a),(c)), each lane's rate
  taken from `calib.pipelines` and scaled by `o`.
- **Prologue/epilogue** are `stages` mainloop-tile loads plus one L2 latency, and
  the output-tile store, both at the per-SM L2 rate.
- **Split-K** uses the Stream-K form of §2.3:
  `a + b·[peers>1] + c·iters + d·(peers−1)`, with `a`,`c` the shape-generalising
  fits below and `b`,`d` read from the calibrated combine table.
- **Combine, non-GEMM stages, grid barriers** are charged per §2.2(f) from
  `combine_fixed_ns`, the per-kind traffic/SFU/`__syncthreads` counts, and the
  measured `grid_barrier_*` curve.

Two scalars generalise the calibrated shapes to unmeasured ones. **Both are
least-squares fits over `calib.streamk` only — never over a measured end-to-end
latency.** The validation set and the model share no number.

| scalar | value | residual | points |
|---|---|---|---|
| `lds_ns` — per scalar `ld.shared` warp-instruction | **0.8577 ns** (≈2.16 SM cycles at 2.52 GHz) | rel rms **3.93 %** | 12 (shape, CTAs/SM) |
| `setup_ns` — per-CTA setup, traffic excluded | **706 ns** | abs rms **483 ns** | 12 |

## 3. The mainloop is LSU-issue-bound at every legal shape ✅

The SIMT f32 TN collective issues `(Tm+Tn)·Tk/2` scalar `ld.shared`
warp-instructions per CTA per mainloop iteration. Fitting `occ_c_ns` on
`o·LdsInstructions` alone reproduces all 12 calibrated points to 3.93 % rel rms
while the FFMA count varies 32× across those shapes and never enters. The
`full-lanes(smem only)` ladder rung — the identical model with the CUDA, SFU, L2
and DRAM lanes deleted — reproduces the full model to within 0.02 MAPE points
and 0.0006 Spearman. **On sm_89 the other five lanes never bind for this
collective.** The lane machinery is kept because it is the part that transfers
to a tensor-core backend, not because it earns anything here.

⚠️ Caveat, unresolvable from these six shapes: per CTA per iteration the
collective also moves `4·Tk·(Tm+Tn)` bytes through cp.async, i.e. exactly
`8 × LdsInstructions`. The SMEM lane and the L2 lane are **collinear** across
every shape in the calibration, so the fit cannot attribute the mainloop to one
rather than the other. It is called the SMEM lane on the microarchitectural
argument (LSU issue, 256 threads, 16×16 `UniversalFMA`), not on the fit.

## 4. Results — §2.4 ladder

Columns: MAPE %, Spearman ρ, then how many of the model's top 1 / 3 / 10 land in
the **measured top 3 %** (rank ≤ 32 of 1077), then the measured optimum's rank
under the model.

### gqa2 (1077 points, 14 GEMMs, 30 stages, 18.46 MiB live)

| layer | MAPE | ρ | top1 | top3 | top10 | opt rank |
|---|---|---|---|---|---|---|
| tier-2 baseline | — | 0.4608 | 0 | 0 | 0 | 95 |
| roofline | 181.35 | 0.4435 | 0 | 0 | 2 | 29 |
| + split-K | 35.51 | 0.9071 | 1 | 3 | 8 | 44 |
| + sync | 27.04 | 0.9095 | 1 | 3 | 9 | 33 |
| + wave tail | 28.53 | 0.9450 | 1 | 3 | 6 | 19 |
| + cache (SDCM) | 28.53 | 0.9450 | 1 | 3 | 6 | 19 |
| **+ non-GEMM = full** | **26.24** | **0.9450** | **1** | **3** | **6** | **19** |
| full + envelope depth | 24.94 | 0.9382 | 1 | 3 | 7 | 53 |
| full − lanes (smem only) | 26.25 | 0.9444 | 1 | 3 | 6 | 19 |

### mha4 (1077 points, 28 GEMMs, 60 stages, 41.02 MiB live)

| layer | MAPE | ρ | top1 | top3 | top10 | opt rank |
|---|---|---|---|---|---|---|
| tier-2 baseline | — | 0.4462 | 0 | 0 | 0 | 191 |
| roofline | 185.00 | 0.4303 | 0 | 0 | 1 | 252 |
| + split-K | 35.07 | 0.9043 | 1 | 3 | 6 | 22 |
| + sync | 26.90 | 0.9070 | 1 | 3 | 6 | 17 |
| + wave tail | 27.45 | 0.9435 | 1 | 3 | 6 | 12 |
| + cache (SDCM) | 27.45 | 0.9435 | 1 | 3 | 6 | 12 |
| **+ non-GEMM = full** | **25.15** | **0.9435** | **1** | **3** | **6** | **12** |
| full + envelope depth | 23.90 | 0.9361 | 1 | 3 | 7 | 17 |
| full − lanes (smem only) | 25.17 | 0.9429 | 1 | 3 | 6 | 12 |

The model's top-1 on **both** models is `16x64x16s3k16`. It measures
0.165088 ms on gqa2 (measured optimum 0.148352 ms, top-3 % cutoff 0.171808 ms)
and 0.321792 ms on mha4 (measured optimum 0.313568 ms, cutoff 0.334816 ms —
measured rank **5 of 1077**).

### §2.5 acceptance

| target | result |
|---|---|
| ≥1 of the model's top 3 in the measured top 3 % | ✅ **3 of 3**, both models; top-1 included |
| rank correlation clearly better than tier-2 | ✅ 0.9450 / 0.9435 vs **0.4608 / 0.4462** |
| < 1 ms per configuration evaluation | ✅ worst observed **33 µs** (60-stage model, 1077 evaluations) |

## 5. Layers that did not earn their place — recorded, not hidden

**(a) §2.2(b)'s fill depth `d = stages · resident_tiles_per_SM − 1` is not
identifiable from this calibration, and is off by default.** ❌ as prescribed.

The calibration measures a line `T = a(o) + c(o)·iters` per (shape, CTAs/SM).
An envelope with fill depth `d` is the *same* line reparameterised: its
intercept is `pro + epi − d·c`. Recovering a per-CTA `setup` therefore requires
adding `d·c` back — but `c` scales with `o` while a per-CTA setup constant
cannot, so the recovered constant inherits the occupancy scaling. Measured over
the 12 calibrated points:

| fill depth | setup | abs rms | span |
|---|---|---|---|
| `d = 0` | 706 ns | **483 ns** | 1685 ns |
| `d = stages − 1` (charged at `c(o)`) | 3562 ns | 1449 ns | 4802 ns |
| `d = stages − 1` (charged at `c(1)`) | 1596 ns | — | 4487 ns |

Adding the term triples the dispersion of the constant it is supposed to
isolate — e.g. `32x32x32s3` yields setup 2590 / 3955 / 6361 ns at o = 1 / 2 / 3.
Regressing the `d = 0` residual on `stages`, `o`, `Tm·Tn` or the LDS count
improves the 483 ns rms by at most 70 ns and does so with nonphysical signs
(setup *decreasing* in `stages`), so a scalar is the right form and the
dispersion is genuine noise, not a missing regressor.

End to end the term is a wash on error and a loss on ranking: MAPE 26.24 → 24.94
and 25.15 → 23.90, but ρ 0.9450 → 0.9382 and 0.9435 → 0.9361, and the measured
optimum falls from rank 19 → 53 (gqa2) and 12 → 17 (mha4). **The acceptance
criterion is ranking (§0.2), so the layer is off.** The envelope's prologue and
epilogue terms are always charged; only the depth-`d` steady-state subtraction
is disabled, `CostModelOptions::pipeline_envelope` keeps it available, and the
ladder measures it every run. The correct reading is *not* "the pipeline
envelope is wrong" but "on this target the calibration cannot separate fill
depth from setup, and forcing the separation injects the occupancy scaling of
`c` into a scalar."

**(b) The SDCM cache layer is inactive on this validation set.** ✅ measured,
not assumed. gqa2's live footprint is 18.46 MiB and mha4's 41.02 MiB, both below
the measured L2 knee of 75.5 MB, so `CacheHitProbability` returns exactly 1.0
and the `+cache` rung is bit-identical to `+waves`. The Gaussian SDCM form of
§2.2(e) (Zelen–Severo `Q`) is implemented and exercised by construction, but
these two models cannot test it. It stays for models that exceed the knee; ❌ it
is unvalidated.

**(c) §0.3's "SplitKReduce is 24–26 % of the best config total" is a measurement
artifact.** ⚠️ The profiled combine stages average 3.630 µs against a 3.456 µs
per-launch floor — a ≈0.17 µs body. Split-K still has to be a model variable,
and the ladder shows why: it moves MAPE 181 → 36 and ρ 0.44 → 0.91. But the
mechanism is not the reduction's arithmetic. It is that splitting **doubles the
stage count** (gqa2 30 → 44, mha4 60 → 88) and therefore the number of grid
barriers, while cutting `iters` per CTA and filling the machine. `+splitk` and
`+sync` are the two halves of that trade and neither works without the other.

## 6. The unexplained residual — stated, not absorbed

The model **underpredicts 99.4 % / 99.1 %** of the 1077 points. Normalised by
the post-split stage count the shortfall is a median **1.58 µs/stage (gqa2)** and
**1.47 µs/stage (mha4)**, p10 0.85 / 0.74, p90 5.75 / 5.55. That residual is what
the 25–26 % MAPE is made of, and it is why the model ranks far better than it
predicts: it is close to uniform across configurations.

Its origin is **not** the grid barrier. The barrier calibration was re-measured
with a faithful per-barrier 128-B-aligned `BarrierEvent` on the hypothesis that
the original single-hot-line kernel understated a cold-line cost; the hypothesis
was **falsified** — the per-barrier cold line measures 1045.9 ns at 128 CTAs
against 1.03 µs for a single reused hot line, i.e. no difference beyond
run-to-run spread. The per-stage residual is
recorded here as unexplained rather than fitted away; no free parameter is
allowed to absorb it, because absorbing it would let the model score against the
validation set it is being scored on.

## 7. Provenance

- Every hardware constant comes from `configs/targets/sm_89.json`
  (`"calibrated": true`); `CostModel`'s constructor throws on an uncalibrated
  target rather than substituting a default.
- `lds_ns` and `setup_ns` are fitted **only** on `calib.streamk`, which is
  microbenchmark data from `tools/tilemega-calibrate`, not from any end-to-end
  measurement.
- Occupancy enters as an input, `Residency{ctas_per_sm}`, derived by the
  validation tool from the committed ptxas logs through the F-40 closed form
  (exact on 1075/1077). Registers do not follow from the tile shape — a 16×256
  tile spans 122…248 — so they are a tier-3 quantity exactly as
  `BackendCandidate::estimatedRegisters()` documents. This is a hardware
  property of the compiled kernel, not the objective being predicted.
- The ladder refits at each rung, so every row is scored with coefficients
  consistent with its own option set. The `lds_ns` fit is option-independent;
  `setup_ns` differs only between envelope-on and envelope-off.
