# R9b deviations and affected gates

## SV-9(e) / G-3: static FP64 in the existing RoPE math library path

Specification: every measured megakernel contains zero FP64 instructions.
Verified in the unchanged, MIDPOINT_REFINE=0 Llama seq1 binary: L2 contains
12 FP64 instructions, L1 contains 24. The separately launched stage kernel
contains another 24. Sequences are I2F.F64.S64, DMUL, F2F.F32.F64. PTX
line information points to RoPETaskBody.h:52-53 (cosf/sinf) and the CUDA
range-reduction constant __cudart_i2opi_f; these are not GEMM refinement.
This is static presence; dynamic execution of this slow branch is not claimed.

G-3 remains FAIL. SV-9(e)'s zero-count acceptance is blocked by the explicit
TaskBody/math-semantics exclusion. Other SV-9 measurements and independent
solver work continue under the limited-stop protocol. No fast-math flag or
TaskBody change is used to erase the evidence. A later authorized change
would need a proven argument bound and a numerically validated range-reduction
implementation before removing these static FP64 paths. Investigation is
small; full numerical coverage and the forbidden source change require a
separate scope decision.

Evidence: controls/llama_s1/{fp64.json,sass.log,original_build.command.json},
fp64_rope_ptx_excerpt.txt.

## SV-11(d): retain the calibrated nominal fixed term in production

Specification: refit the fixed term on physical output elements; §7.3 explicitly
permits retaining the old fixed term if the fit does not improve.
The new fit lowers mean relative error (0.317941 to 0.278472) but raises the
median (0.215559 to 0.234919). The independent fixed-only gqa2 replay also
lowers Spearman from the freshly replayed baseline 0.800355 to 0.634046.
Production FlowPreparation therefore disables physical_fixed, retaining the
new fit as an independently selectable audit component. Physical traffic and
stage latency remain enabled. Their combined ranking is still being checked;
this fallback is not a declaration that G-5 passes. Raw evidence: fit/ and
replay/{baseline_bf16,fixed_bf16,physical_bf16,stages_bf16}/.

## SV-10: data-dependent embedding image

Specification: all floor terms are quasi-polynomials in theta.
An indirect embedding read depends on token IDs as well as shape parameters.
The semantic gather placeholder cannot express the unique token rows using
seq/past/batch alone. DeriveModelDramFloor binds the actual fixture token image
and counts its union (including duplicates) exactly. It rejects a missing
indirect image rather than charging all vocabulary rows or guessing uniqueness.
The other tensor read/write counts remain theta-parametric. A theta-only exact
embedding term requires an additional distinct-token parameter or a semantic
indirect-image contract; no TaskBody or input generation is changed.

## SV-9(b) early-driver correction before accepted measurements

The first minimal driver invoked Lower(module), which preserved invocation
metadata but emitted the default single GEMM shape. The harness rejected it
before kernel execution: "ModelSpec/template mismatch in runtime variant 0
GEMM 0" (early/llama_s1.measurement/process_00.log, exit 2). No numerical
comparison or valid timing was produced. The corrected driver uses
LowerVariants, exactly as the R9 production compile driver does. It re-lowers
the already solved archived CG using the frozen baseline library; placement is
not re-solved with R9b. Original artifacts remain under early/, corrected ones
under early_corrected/. This is an experiment-driver error, not a regression
in an existing executable configuration and not evidence of internal mismatch.
