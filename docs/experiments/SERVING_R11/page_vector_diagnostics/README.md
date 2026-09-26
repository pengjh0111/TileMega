# Current-source PG-1 diagnostics at batch endpoints

The four cells use B=1/16 as requested. `summary.tsv` gives the realized
chain at past=575 beside the exact CG-derived DRAM floor, the residual
bubble per chain link, per-CTA time with a full page ring while waiting
for dependencies, and adjacent decode-launch gaps. Each cell's
`page_chain.tsv` retains all 1023 exact per-step floors and measurements.
The chain length is observed at past=575 and reused as the denominator
for other steps; it is not remeasured at every past value. Page stall
times are CTA-local and cannot be summed into request wall time.
Trace instrumentation perturbs timing; non-instrumented performance
is measured separately with the same geometry. `raw.tar.xz` preserves
the raw page/chain traces, floor evaluation, commands, and guard logs.
Raw archive SHA256: `878fe95b969400c8c0ea2858d1d23e1986104c8206a0d0d1f0c37200708f9a18`.
