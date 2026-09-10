# A12.2 FP32-partial combine calibration

## Resolved measurement (supersedes the local stop, preserves it below)

✅ `lib/Target/GemmCalibration.cu:70` now measures a pair of CUDA graphs,
each containing 64 launches of the **same** combine kernel and launch shape.
The control uses count=0; each of 41 timed rounds rotates control/work order.
Three warmup pairs precede timing. Batching resolves the interval without
changing the kernel, input, numerical tolerance, or fitting a positive epsilon.
`TILEMEGA_COMBINE_GRAPH_TIMING=0` rejects this explicit measurement mode;
`--combine-graph-batch 0` retains the historical timing protocol.

✅ Fifty fresh processes pass the CPU sequential-FP32-reduction bit check
for all 1048576 outputs with 32 chunks: **50/50**, 52428800 output elements,
1677721600 partial contributions. Actual initialized GPU partials are read
back; expected values are not a constant fill. Raw logs, per-process profiles,
frozen source diff and binary/base hashes are in `partial_graph_validation/`.
The independent verifier checks hashes, arm rotation, raw timer resolution,
profile/table agreement and correctness counts (`verification_profile.json`).
No GPU timing ran concurrently. This checks the calibration kernel, not a new
50-process end-to-end FP32-model regression claim.

| BF16-output / FP32-partial coefficient | 50-process median | 95% bootstrap median CI |
|---|---:|---:|
| fixed ns | 61.750004535 | [60.99999882, 62.50001024] |
| base ns/output element | .0044639512115 | [.004463739, .0044660920635] |
| L2 ns/extra partial/element | .00064544765595 | [.0006452556833, .0006464945116] |
| DRAM ns/extra partial/element | .0043108035155 | [.004310747438, .004310833004] |

Bootstrap: 10000 resamples of independent processes, NumPy generator seed 0.
Only the BF16 `fp32_partial_combine` subtree in `configs/targets/sm_89.json`
was published, not the rounded full-target serializer output. Other targets
remain explicitly `not_calibrated`; missing profiles reject measured pricing.
`CostModel.cpp:379` consumes all four rates instead of the analytical extra.
`TILEMEGA_MEASURED_PARTIAL_COMBINE` now defaults ON, with
`--analytic-partial-combine` / the OFF macro retaining the historical control.
No FP32 or unsplit pricing expression was changed.

✅ Both FP32 prediction tables (1077 configurations each) and the entire
ablation/ranking summary are byte-identical between measured ON and OFF
(`partial_rate_comparison/`). FP32 rho/top-k therefore does not regress.
⚠️ The preserved historical BF16 subset diagnostic changes full-model rho
gqa2 .9049→.9040, mha4 .8929→.8911; top1/3/10 remains 0/0/0. This slight
negative result is retained, **not a post-repair BF16 ranking acceptance**:
the archive contains 770/462 accepted points from the old partial format,
and its omitted 308 mha4 entries were numerical failures, not run failures.
The cost comparison changes rates only; it does not rerun that GPU oracle.

### Measurement detour and correction

The first graph attempt already resolved about 62 ns per launch, but the
first implementation compared that per-launch mean against a 100 ns
**whole timed interval** resolution test. It incorrectly rejected the run
(`first_resolution_failure.txt`). The corrected test compares
`per_launch_ns * batch > 100 ns`; the physical one-tick requirement remains
unchanged. `single_resolved.txt` and all 50 subsequent process logs report
both units. The earlier negative single-launch result below remains valid
for that protocol and is not overwritten.

## Historical unresolved single-launch measurement

✅ After A9's 1200 attribution processes finished, the prepared command below
ran alone on sm_89. ⚠️ The measured profile was rejected as `not_calibrated`;
no rate was inserted into a target config and the independent switch stays OFF.

`GemmCalibration.cu::MeasureFP32PartialCombine` reuses the existing width/peer
sweep and fit, with float partial storage and the selected output dtype.
The new kernel uses `kTensorBF16Threads` / `kSimtF32Threads` from the backend
traits; the historical calibration instantiations keep their old thread
count and same-element partial type. This calibrates the sequential partial
read/reduction/store loop, not the optional residual-add epilogue or a new
full-model timing. The existing L2/DRAM regimes stay separate.

The profile is `Calib::fp32_partial_combine`, with optional fixed/base/L2-peer/
DRAM-peer rates and explicit `not_calibrated` state. Unresolved fixed cost or
nonpositive fitted peer rates are rejected; they are not replaced by the old
BF16 coefficients. New measurements carry a `fp32_partial_` prefix and record
partial/output types, thread count, method, timestamp and device provenance.
Existing pipeline, GEMM and FP32 profiles are not overwritten by the new mode.

Executed command (no concurrent GPU measurement):

```sh
build-portable/tools/tilemega-calibrate --dtype bf16 \
  --base configs/targets/sm_89.json --fp32-partial-combine-only \
  --out docs/experiments/COST_MODEL/partial_combine_measured.json
```

The independent cost control is `TILEMEGA_MEASURED_PARTIAL_COMBINE`, currently
default OFF pending measurement; `CostModelOptions::measured_partial_combine`
and `tilemega-costmodel --measured-partial-combine --target FILE` expose it.
ON requires the new rates and removes the analytical two-byte increment;
OFF is the explicit historical comparison, never an automatic fallback.
FP32 and unsplit paths do not consult the new BF16 rates.

✅ Native calibration/code-model binaries build, `target_spec_test` and
`chain_dp_test` pass, and check-policy passes. Tests reject a missing new
profile, verify FP32 combine bit equality across split=1,2,4,8,16, and check
coefficient consumption with a clearly synthetic in-memory unit fixture.
Those synthetic values are not device measurements or acceptance coefficients.
The first run's raw log is `partial_combine_measurement.txt`. The fitted L2
peer slope was 0.000717795 ns/peer/element (R²=0.984407); DRAM peer slope
0.00429395 (R²=0.83896); base slope 0.00484034 ns/element. These are diagnostics,
not a published complete profile. All four small-width launch-subtracted
readings (1/64/256/1024 elements) were **−128 ns**. The existing measurement's
fixed-cost resolution condition therefore failed. `MeasureFP32PartialCombine`
in `lib/Target/GemmCalibration.cu:807` rejected the result; exit status 1 does
not signify a numeric correctness failure. The output target file was not made.

Analysis: the width/peer sweep resolves a bandwidth slope, but subtracting a
separately timed count=0 kernel does not resolve the small fixed component in
this run. Negative subtraction is not a physical negative cost. No positive
epsilon, old BF16 fixed coefficient, or clamped zero was used to pass the gate.
A12.2 alone remains locally stopped pending a measurement design that resolves
that component; A3/A6/A9 remain independent. Rate provenance, GPU result checks
and full FP32 regression are still owed; this item is not complete.
