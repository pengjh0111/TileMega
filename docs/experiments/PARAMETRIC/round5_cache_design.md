# B3.1 design: measured service-time segments

A6/A9's entry gates passed before this work started; see
`COST_MODEL/stage_price_gate/result.md` and `EVENT_COST/round5_structured.md`.

The explicit experimental option is `TILEMEGA_MEASURED_CACHE_CURVE` (default
OFF) / `CostModelOptions::measured_cache_curve`. CDF remains the independent
control and is not removed or silently bypassed. Ranking validation completed:
both BF16 rows regress slightly; see `cache_curve/result.md`. Default remains OFF.

## Representation choice

Each observed `(bytes, GB/s)` knot becomes `(bytes, ns/byte=1/(GB/s))`.
Interpolate **service time** between knots, not bandwidth followed by a
reciprocal. Both choices reproduce the original knots, but only the former
is piecewise affine. This is a documented choice of interpolation quantity,
not a claim that the two interpolation schemes are mathematically identical.
No added fitted coefficient, numeric root search, or sampled polynomial fit
is used. Outside the measured range, the boundary segment is constant.

To feed the current separate L2/DRAM lanes, use the effective service-time
mixture

`miss = clamp((service - 1/l2_rate)/(1/dram_rate - 1/l2_rate), 0, 1)`.

The resulting hit fraction is piecewise affine. Its boundaries include the
measured footprint knots **and** crossings of the two physical clamp limits.
Those crossings must be included in B3.2's segment union; ignoring them would
not be a valid symbolic representation. No reciprocal of a theta-dependent
quantity remains. Rates and the per-knot reciprocal are device constants.

This effective fraction is a modeling bridge, not a measured cache-hit ratio:
the bandwidth curve also contains small-working-set utilization effects.
No claim of better accuracy precedes the rank comparison. Raw service knots
are retained exactly; hit-fraction clipping can flatten noisy knots outside
the separately calibrated L2/DRAM endpoints, and that distinction is explicit.
Missing curves or invalid/nonmonotonic knots throw `not_calibrated`/invalid
input errors, never use zero or the old CDF as a fallback.

Implementation: `CacheServiceCurve.cpp` / `.h`, consumed only by the explicit
branch of `CostModel::CacheHitProbability`. Unit checks compare all 36
BF16/FP32 service knots bitwise to the reciprocal of the archived measurement,
plus a dyadic affine-interpolation test and eight actual rejection branches.

## Validation / continuation rule

Use the same unified task model and measured partial-combine profile in both
arms. Compare BF16 historical 770/462 measured configurations and FP32
1077/1077, preserving their exact data provenance; do not invent timings for
the 308 numerical failures. Report full per-point predictions, rho/top-k and
changes. If BF16 rank declines, retain both branches and the negative result;
do not enable the experimental curve as the production default to force (a).
This does not block independent A7, fusion or placement work.

B3.2 is not implemented by this helper: it still requires symbolic lane
intersection/period handling and DP transitions, a >quadratic rejection,
and exact selected-plan comparison with the finite S=1..16 control.
