# R11 user-directed experiment scope amendment

The 2026-09-26 continuation asks for batch sweeps to use only B=1 and B=16.
The R11 end-to-end matrix and similar batch-dependent comparisons therefore
cover Llama and Qwen3 at those two extremes (four cells). The original R11
G-9 threshold was defined for ten cells; its ten-cell geometric mean cannot be
claimed from these four observations. Report the four-cell geometric mean
separately and mark the original ten-cell gate unmeasured. Architecture checks,
the required new-process protocol checks, and non-batch coordinates retain
their specified coverage.

The same request adds PG-1 diagnostics: residual wall-clock time above the
CG-derived DRAM floor per realized dependency-chain link; CTA time with the
page ring full while the consumer waits on dependencies; measured chain length
and duration beside the DRAM floor; and the device-visible gap between
adjacent decode launches. Diagnostic instrumentation is compiled separately
from performance binaries. Its timing overhead is disclosed, and a normal
binary supplies the performance comparison.
