# Current-source PG-1 diagnostics at batch endpoints

The four cells use B=1/16 as requested. `summary.tsv` gives the realized
chain at past=575 beside the exact CG-derived DRAM floor, the residual
bubble per chain link, per-CTA time with a full page ring while waiting
for dependencies, and adjacent decode-launch gaps, including their
total, p90/p99, and fraction of the decode span. Each cell's
`page_chain.tsv` retains all 1023 exact per-step floors and measurements.
The chain length is observed at past=575 and reused as the denominator
for other steps; it is not remeasured at every past value. Page stall
times are CTA-local and cannot be summed into request wall time.
Trace instrumentation perturbs timing; non-instrumented performance
is measured separately with the same geometry. `raw.tar.xz` preserves
the raw page/chain traces, floor evaluation, commands, and guard logs.
`protocol_raw.tar.xz` retains 200 distinct process results,
each with five matching token hashes and its guard log.
Protocol archive SHA256: `1da9500e5c9110cf9e2c70367b3d45b9546d999a315d9af9364294a37958b5e6`.
Raw archive SHA256: `16e044ba290854a12c0fd1625c9fce1d51206915b8277d712610c6450a058d39`.

`attention_bandwidth.tsv` derives each attention stage's effective historical
KV rate from these same traces. At B=16 and past=575 it is 838.746 GB/s for
Llama and 744.509 GB/s for Qwen3, or 85.4% and 75.9% of the target's
981.583 GB/s DRAM calibration. This is an effective byte count divided by
stage wall time, not a hardware DRAM counter; some reads may hit L2. The
numbers meet AT-3's 60% transport criterion on these fixed geometries, while
the final solver-selected plans remain to be measured.
