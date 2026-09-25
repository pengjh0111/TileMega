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

The first correctly lowered snapshot then failed residency preflight: requested
512 workers, actual limit 384 (early_corrected/llama_s1.measurement/process_00.log).
It also executed no kernel. The same two archived geometry vectors are now
re-solved with the frozen R9 library at residency 3 and measured under
`early_resident/`. The original 512-worker score is not attributed to the
384-worker executable. Four legacy trace jobs run independently of this repair.

## SV-9(d): combined task phases and scalable trace reconstruction

Specification: separate fixed and mainloop using phase stamps, or combine and
state the limitation when trace v2 cannot distinguish them. These four trace
builds enable trace v2 only, so TaskBody fixed plus mainloop is reported as one
measured interval. It is not inferred from the phase fit. The initial processes
set only TRACE_V2_OUT and allocated no trace buffers; activated.log processes
also set TILEMEGA_TRACE_V2=1 and retain separate evidence.

The existing analyzer's dense all-producer predecessor sets are unsuitable for
hundreds of thousands of anchored-model tasks. Its historical entry point and
outputs are retained. The added analyze_task_space_trace entry point evaluates
exact dependency-window range maxima and worker queues without expanding
all-producer edges. Four historical seq4 dumps match the original corrected
zero-sync bound exactly (unit/trace_range.log). Per-space unique-byte streaming
excess uses each complete space's wall span; these overlapping spans are not
summed into a path. chain_categories.tsv instead partitions the nonoverlapping
realized path. No fixed/mainloop separation is claimed from these timestamps.

## SV-13(g): seeds at theta points without a measured legacy winner

The seven-point grid imports the symbolic s64 model once per model and changes
only theta bindings in SolveSkeletonImported. At seq 2/8/32 there is no measured
legacy winner in the specified eight-cell control matrix. The first starting
point uses the uniform winner of the next available legacy seq (4/16/64), and
the second is still the full uniform-domain optimum at the actual theta. Each
completed.tsv records seed_seq explicitly. This does not introduce a production
seq-specific price or placement rule; it is an experiment's starting-point
choice at previously unmeasured theta points.

## SV-9(d) / chain-depth denominator: use the actual measured graph

The current legacy CGs have chain depths 228 (Llama) and 424 (Qwen3) in the
raw flow audit, not the prompt's 260 and 480. The differences are exactly two
residual additions per layer; the measured model plan already folds those
residuals into existing tasks. The R9b implementation does not add or remove
fusion. Reports use each measured CG's own D, rather than silently dividing by
the prompt's larger chain. Verify the stage/category tables when attributing
fusion opportunities: an already folded add is not an available chain removal.

## SV-12(a), P-9: exact CG releases versus wider runtime event windows

Specification: derive each release endpoint from the exact CG predecessor
Oracle and Coarsen(kappa), and confirm agreement with runtime waits; Level 1 and
the FIFO simulator should differ only in aggregation and queueing.
An independent reference-model audit at seq 4/128 and kappa 1/2/4 finds 472
endpoint differences in 2,064 sampled producer/consumer fibers. For example,
gqa2 seq4 kappa1 consumer (4,0), producer 1, has exact-CG endpoint 1 but runtime
endpoint 3. `PrepareSymbolicProblem` retains both relations: its dependencies
are replaced by ExactRuntimeDependencies, while requested_events retains
ProjectRuntimeQueues' conservative windows. `FlowPreparation` implements the
specified exact-CG endpoint; `SkeletonFinalize` correctly adds the real requested
events to the final simulator and legality check.

The prescribed exact-CG release formula is retained. No executor window,
TaskBody, ownership, or event semantics is changed to manufacture agreement.
This is a declared degradation of the Level-1 approximation: agreement with
runtime release endpoints is FAIL, and V2's correlation includes this additional
source of approximation, not just cohorts/FIFO. It must not be described as a
proof of identical release constraints. Top-M legality and FIFO simulation still
use the actual event windows; downstream measurement continues on that basis.

Resolution requires an explicit choice between adding the runtime event horizon
to Level 1 (`max(exact CG endpoint, requested-event endpoint)`, including its
wait/publication masks), or tightening execution windows in a later authorized
round. The former is a bounded solver change but changes the supplied release
specification and needs a new search/model-validation matrix; the latter touches
R9b-excluded execution semantics. Evidence: `unit/runtime_release.log` and
`test/unit/flow_runtime_release_test.cpp`. No numerical correctness regression
or global-stop condition occurred.

## SV-14(d): historical displacement field corrected from raw placement

The early top-M materializations TSV emitted `1 - home/placed`, although the
exclusive affinity bin can also contain a tile placed at its structural home.
The scheduler's independent `moved_from_home` counter is the correct quantity;
the search TSV writer now uses it too. Existing binary outputs are preserved.
`report.py` reconstructs older counts from same-configuration A/B task worker
tables, and records the original field beside the corrected value. This changes
reporting only, not placement, scores, selection or GPU binaries.
