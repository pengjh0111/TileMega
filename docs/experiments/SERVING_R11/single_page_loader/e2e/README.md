# Fixed-geometry PG-1 endpoint controls

Each model uses B=1 and B=16, as requested. Within each cell, the
PG-off and PG-1 arms share GEMM geometry, grid,
residency, κ, attention coordinates and sync settings. The two arms
were timed in one session under the same predeclared contamination
policy. Three full-request timings and three TTFT timings per arm
follow one warmup each. The raw archive retains per-round guard,
clock, token and step-time records. These are fixed-geometry controls,
not final solver-selected EV-2 results.

Raw archive SHA256: `2270cb3bb497c46c0c12c92b7bfdf9e82e64f3a6ab66723759305a15dfbae25b`.
