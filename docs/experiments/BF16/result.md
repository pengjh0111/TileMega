# BF16 end-to-end and calibration

Evidence status: ✅ measured on RTX 4090 (`sm_89`) on 2026-09-05.

The dtype is read from every `ExportedProgram` FakeTensor and stored on the
L-sem operator and generated `ModelSpec`; it is not inferred from granularity.
The implementation registry enumerates the SM80+ BF16 Tensor Core family only
for BF16, including its distinct legality (`tile_m % 32`, `tile_n % 16`,
`tile_k % 16`, 128 threads and 8-element alignment).  Dispatch remains behind
`ArchDispatch::Caps::kBf16TensorCore`.

All bodies store BF16. GEMMs, RMSNorm and attention dot/softmax/value reductions
accumulate in FP32; explicit BF16 materialization boundaries preserve the
exported graph's `linear -> residual`, SiLU and RoPE rounding semantics.  The
two reference models passed PyTorch L0 → L0.5 → L1 → L2 in 50 fresh processes
each (see [`correctness.tsv`](correctness.tsv)). `cuobjdump` finds 96 static
`HMMA.16816.F32.BF16` instructions in each generated executable, so this is
not a BF16-storage/SIMT computation.

The BF16 tolerance is `1.6e-2 + 1.6e-2*abs(reference)`.  FP32 retains its
`3e-5` rule.  At `seq=2048`, `1e-2` left four values on legitimate BF16
quantization boundaries among millions; `1.5e-2` left zero, so the selected
bound includes a small explicit margin rather than reusing an FP32 threshold.

`tilemega-calibrate --dtype bf16 --base configs/targets/sm_89.json` now writes
`calibration_by_dtype.bf16` and preserves the original `calibration` object
byte-for-byte in meaning.  The accepted third run achieved 97.37% of the
reported DRAM pin rate and only 0.00073% start/end drift; two earlier runs were
correctly rejected (9.30% and 46.12% drift) and were not committed.  The final
profile measures the BF16 MMA instruction and the real BF16 CUTLASS Stream-K
collective; compact values are in [`calibration.tsv`](calibration.tsv) and
[`streamk.tsv`](streamk.tsv).

The cost model is now dtype-aware too: BF16 mainloop work populates the `tc`
lane rather than the CUDA-core lane, storage traffic uses two-byte elements,
and per-lane controlled ablations are emitted by `tilemega-costmodel`.  The
final Spearman/top-3% and SMEM/L2-identifiability conclusions are generated
from the BF16 oracle in `../ORACLE/raw_bf16/cost`; they must not be substituted
with the old FP32 validation set.

## The oracle has now run, and Part 2.4's acceptance fails

✅ measured, 1540 generated/compiled/measured points, `../ORACLE/result.md`
§6.7.  Part 2.4 asked that BF16's ρ and top-3% hit rate be **no worse** than
FP32's ρ 0.9450 / 0.9435 with top-1 and top-3 inside the measured top 3%.

| | gqa2 | mha4 |
|---|---:|---:|
| full model ρ | **0.5605** | **0.6239** |
| MAPE % | 39.41 | 37.59 |
| top1 / top3 / top10 | 0 / 0 / 0 | 0 / 0 / 0 |
| rank the model gives the true optimum | 104 | 51 |
| uncalibrated analytic `tier2-baseline` ρ | **0.8778** | **0.8738** |

The acceptance is not met and no threshold was moved to meet it.  In BF16 the
calibrated model ranks *worse than the analytic baseline it replaces*, having
beaten it 2:1 in FP32.

The attribution is in §6.7 and is not a defect of §2.2's structure: the
`+splitk` layer is the one that inverts, because `combine_fixed_ns` is **0** in
the BF16 profile (FP32: 108.1 ns).  The calibrator stores `max(0, fit)` for an
intercept it honestly reports as unresolved (`|value| < 300 ns`, 150% spread,
either sign); in FP32 the fit landed positive and the clamp never bound.  With
a reduction stage that costs the model ~0.12 µs across the whole graph, split-K
is nearly free and the model's eight best configurations are all split-K 16,
predicted 2–3× faster than they measure.  Three of the six BF16 Stream-K points
also fit a **negative** per-CTA setup (`a_ns` −119.2 / −438.6 / −400.7,
`fit_r2` 0.922 against FP32's 0.974), which cannot be a setup time.

⚠️ Part 2.1's premise is also not confirmed on this target.  The `tc` lane —
the reason nine lanes were restored — changes ρ by **0.001** when removed
(0.5605 → 0.5595).  Six of the nine lanes are exactly inert.  BF16 makes the
mainloop fast enough that the bottleneck moves *away* from the compute lanes,
rather than into `tc`.

Fixing this is a measurement problem (resolve the reduction stage's fixed cost
instead of clamping it) and is deliberately left to the next round rather than
attempted at the end of this one.

The requested SMEM/L2 retest has one important negative result already:
they are **still exactly collinear in the implemented BF16 model**.  Across all
20 calibrated `(shape, occupancy)` points, both lanes use the same
`occupancy * 2 * tile_k * (tile_m + tile_n)` feature; their ratio is a constant
3.4715200776 and Pearson is 1.0 (`identifiability.tsv`).  Consequently this
fit cannot independently attribute a BF16 timing change to SMEM versus L2.
Tensor Core work is a different feature and can break the old one-lane
ranking, but changing dtype alone does not make the two byte lanes
identifiable.  This is a limitation of the present feature construction, not
evidence that the hardware pipelines themselves are identical.
