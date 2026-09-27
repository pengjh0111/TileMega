# R11 user-directed experiment scope amendment

The 2026-09-26 continuation first narrowed batch sweeps to B=1 and B=16.
The 2026-09-27 continuation further narrowed final acceptance to one model,
Llama, at those two batch sizes. The original R11 G-9 threshold was defined
for ten cells; its ten-cell geometric mean cannot be claimed from two
observations. Report the two-cell geometric mean separately and mark the
original ten-cell gate unmeasured. Earlier Qwen3 development checks and the
required new-process protocol checks remain useful evidence, but they are not
part of the final EV-2 performance matrix. Architecture checks and non-batch
coordinates retain their specified coverage.

The same request adds PG-1 diagnostics: residual wall-clock time above the
CG-derived DRAM floor per realized dependency-chain link; CTA time with the
page ring full while the consumer waits on dependencies; measured chain length
and duration beside the DRAM floor; and the device-visible gap between
adjacent decode launches. Diagnostic instrumentation is compiled separately
from performance binaries. Its timing overhead is disclosed, and a normal
binary supplies the performance comparison.
