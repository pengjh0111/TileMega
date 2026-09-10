# B3.1 — measured cache service curve comparison

✅ CPU paired prediction comparison, identical configurations and archived
measurements in both arms. The CDF control's complete prediction tables are
byte-identical to the A6 unified-path tables. All four invocations report
`ISL_CONTEXT remaining=0`. `verification.json` hashes the inputs;
`pointwise_deltas.tsv` contains every paired prediction.

| dtype/model | measured configurations | CDF rho | service-curve rho | top1/top3/top10, both |
|---|---:|---:|---:|---|
| BF16/gqa2 | 770 | .9038734673 | .9036449015 | 0/0/0 |
| BF16/mha4 | 462 | .8910704759 | .8910404180 | 0/0/0 |
| FP32/gqa2 | 1077 | .9431793641 | .9431793641 | 1/3/6 |
| FP32/mha4 | 1077 | .9421013799 | .9421013799 | 1/3/6 |

These rho values are independently recomputed with tie-aware ranks from the
printed prediction precision; `summary.tsv` also retains the driver's original
precision. BF16 is explicitly conditional on the historical measured subsets:
the 308 numerical failures have no acceptable measured ranking observation.
This experiment does not manufacture replacement timings or claim a new GPU
oracle over the complete BF16 candidate set.

The BF16 regression gate is triggered, even though the decrease is small.
Both implementations are retained, and `TILEMEGA_MEASURED_CACHE_CURVE` remains
OFF by default. The curve changes 616/770 gqa2 predictions (0 to +.000310 ms)
and 308/462 mha4 predictions (0 to +.000330 ms); FP32 predictions are unchanged
byte-for-byte. There is no fitted correction to hide this negative result.

Implementation locations: `lib/Solver/CacheServiceCurve.cpp:9` validates and
converts measured knots; `:22` interpolates service time; `:33` supplies the
physical hit/miss mixture. `lib/Solver/CostModel.cpp` consumes the curve only
under `CostModelOptions::measured_cache_curve`.
`tools/tilemega-costmodel.cpp` exposes `--sdcm-cache` and
`--measured-cache-curve`; all other pricing controls are unchanged.

The design and limitations of interpolating service time rather than GB/s
are in `../round5_cache_design.md`. The affine branch remains available for
an experimental symbolic solver, but it cannot replace the production CDF
path on a claim of equal or improved BF16 ranking. B3.2 is not implemented by
this curve helper; in particular no finite enumeration is relabeled design
(a). Independent A7/B1/B2 work continues.

Unit verification: all 36 measured BF16/FP32 knots equal their original
reciprocals bitwise, dyadic interpolation/clipping examples pass, and eight
invalid-input branches reject. Reproduce ranking verification with
`python3 docs/experiments/PARAMETRIC/verify_cache_curve.py` into a fresh copy
of the evidence directory (the verifier refuses to overwrite its receipts).
