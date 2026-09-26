# Fixed-geometry PG-1 endpoint controls

Each model uses B=1 and B=16, as requested. Within each cell, the
PG-off and optimized 16 KiB PG-1 arms share GEMM geometry, grid,
residency, κ, attention coordinates and sync settings. The two arms
were timed in one session under the same predeclared contamination
policy. Three full-request timings and three TTFT timings per arm
follow one warmup each. The raw archive retains per-round guard,
clock, token and step-time records. These are fixed-geometry controls,
not final solver-selected EV-2 results.

`summary.tsv` also gives each arm's E2E divided by the R10-C
CG-derived request floor for the same model and B. This is a common
workload reference floor for this fixed-geometry diagnostic, not a
fresh per-arm floor derivation. The final EV-2 plans require their own
CG floor calculation.

Raw archive SHA256: `a6a1394322883b78c0c28bccbe76a12ed1e501781542c12f82937c3ff9e67282`.
