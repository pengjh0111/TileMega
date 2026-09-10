# Exact Quadratic Building Blocks

Verified by `../verify_quadratic.py`:

| Check | Result |
|---|---:|
| Integer ordering vs independent polynomial evaluation | 72501/72501 |
| Min/max envelope vs direct QP evaluation | 126/126 |
| Symbolic cache interpolation vs exact binary synthetic curve | 195/195 |
| Existing measured cache knots, bit equality | 36/36 |
| Lane/QP rejection branches | 12, reference delta 0 |
| Cache rejection branches | 11, reference delta 0 |

`QuasiPolynomial::QuadraticIntervals` (`lib/Analysis/QuasiPolynomial.cpp`)
uses ISL bounded floor-domain splitting, then extracts exact rational
coefficients. It verifies each emitted domain equals its integer interval
hull; congruence holes are not erased. Degree >2, remaining floors or
unbound extra parameters are explicit errors. Missing support denotes zero
as in ISL arithmetic, not infeasibility; DP feasibility must remain separate.

`OrderQuadraticLanes` / `QuadraticEnvelope`
(`lib/Solver/LaneIntersections.cpp`) clear rational denominators using GMP
and isolate quadratic roots using exact integer square root. The integer
floor of each root, its adjacent point and ties partition the parameter
domain. No iterative root search or sampled fit is used. A rational root
1+10^-38 is explicitly tested so floating-point rounding cannot silently
change the winning interval. Initial uncommitted floating-point root code
was replaced before publication for this reason.

`CacheServiceCurve::MissIntervals` (`lib/Solver/CacheServiceCurve.cpp`)
unions measured-knot crossings with the physical miss-fraction clamp
crossings. Measured binary64 coefficients are lifted exactly to rationals.
This is algebraic evaluation, not a claim of reproducing each IEEE arithmetic
operation's bit pattern. The existing concrete cache evaluation is unchanged.

During implementation, one build failed because ISL's min/max declarations
require `isl/ilp.h`; the explicit include fixed it. The tests above were run
after that correction.

Unfinished: constructing every real task/wave lane and DP transition from
these pieces, returning the complete symbolic model decision, and the
S=1..16 choice gate. These building blocks do not retire design (b), do not
satisfy B3.2 by themselves and do not alter the CDF default after B3.1's
negative ranking result.
