# PG-1 chain and launch diagnostics (B=1/16)

`summary.tsv` is computed from the four `page_chain_summary.json` files. Each
`page_chain.tsv` retains every decode step's exact CG DRAM floor, instrumented
kernel span, residual bubble estimate, CTA page/dependency stall totals, and
device-visible gap to the preceding launch. `raw.tar.xz` contains the raw
per-CTA page trace, trace-v2 chain analysis, exact floor evaluation, commands,
and guard records (SHA256
`36b331af94300dd2a5a7136cb49abc3fb31f90119f1734420dd3acaabe329d75`).

The realized dependency chain was measured at `past=575` and compared only
with the CG floor at `past=575`. The per-step bubble estimates reuse that
single observed chain length as a denominator; they do not assert that the
chain has the same length at every past value. `page_full_and_dependency_wait`
is an intersection of two CTA-local states. The accumulated CTA nanoseconds
over all 128 workers and 1023 decode steps are **not** request wall time;
`page_full_and_dependency_wait_mean_cta_ns_per_step` is the comparable per-CTA
statistic.

Both trace modes add substantial instrumentation and perturb the measured
kernel time. The chain span and per-step bubbles here characterize trace-mode
execution. Production PG-1 performance must use the non-instrumented binary
and is reported separately in the controlled PG ablation.
