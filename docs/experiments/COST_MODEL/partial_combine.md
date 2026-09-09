# A12.2 FP32-partial combine calibration (measurement unresolved)

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
