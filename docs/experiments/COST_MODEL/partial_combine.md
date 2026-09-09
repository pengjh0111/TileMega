# A12.2 FP32-partial combine calibration (prepared, not measured)

⚠️ GPU calibration has not run. No rate was inserted into a target config.
The A9 four-arm experiment owns the GPU until it completes.

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

Prepared command (run only after the GPU is free):

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
Actual rate provenance, residuals, GPU result checks and full FP32 regression
are still owed; this item is not complete.
