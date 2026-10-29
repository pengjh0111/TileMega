# MX-5 MoE native development

Verified 2026-10-29. Authority: `TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
MoE native model acceptance remains **0/5**. No MoE GPU work has started.

## Host analysis

Verified `AnalyzeMxMoe(ModelPlan const&)` is now built into the root compiler
library, with no change to compiler routing or execution defaults. It reads
original stages/GEMMs/buffers directly; it does not import a legacy descriptor
program. Checks cover layer order, attention geometry, gather/scatter bindings,
expert strides, folded recipes, routing dtype, and epilogue rounding.
External/weight metadata is copied; legacy internal scratch is excluded.

Verified portable tests: 60 valid combinations of B, layer count, optional
attention merge, and routing dtype; 480 specific invalid-plan rejections.
Five real MoE-L24 bridge plans also pass: B1/2/4/8/16, 24 layers, 246 external
and weight buffers. Root `tilemega` and `mx_moe_model` targets build successfully;
`ctest -R '^mx_moe_model$'` passes. These are CPU results, not native execution.

The actual legacy head contract is FP32 argmax values plus I32 indices.
An earlier isolated assumption requiring BF16 head output was wrong and was
corrected before integration; its failed test remains under ignored runs.

Verified native buffer preparation preserves all 246 original external/weight
names, dtypes, sizes and ordering, wrapping the 121 GEMM weights with the chosen
xcore_mma TN/32 pages. The other 25 weights remain present with unused native
slots; they are not silently dropped. Both prompt/decode must use identical
recipes for each candidate. Invalid gate tiles and malformed recipes reject.

Verified root StageProgram now requires an explicit prompt tail predicate.
Four host emission/execution tests pass, including 64 prompt tokens with
144 layer counters incremented 64 times and the head counter once: 9,217
releases. This is a host simulation, not a GPU counter/litmus or prompt ABI test.

## Device body preparation

Verified isolated normal cucc builds, with exact-function MIR checks:

| Body | Instances | Result |
|---|---:|---|
| Router/top-k/dispatch | 20 | zero frame/static SMEM/generic/calls/private |
| Grouped gate/up and down | 12 | same; includes incoming down prefetch and combine LA |
| Combine | 16 | same; rank-ordered arithmetic mirrors legacy |
| Router GEMM plus tile/route LA | 8 | same; TileRange/StreamK, BF16 logits then dispatch |

These bodies remain under ignored runs. They are not integrated model kernels.
Grouped metadata is cached per unit; WK1 invalidates the RMS inverse when the
active expert changes. Geometry checks cover 5,376 configurations and
13,908,844 units. This does not establish numerical or synchronization correctness.

Verified first prefetch-integrated grouped build exposed generic accesses in
MACA's private-pointer branches of atomicAdd/atomicExch. An assumption alone
failed to remove them. Carrying the global AS1 ticket type in the operand ABI
removes them in the subsequent build. Fence/atomic/acquire/reset semantics
remain unchanged. Failed builds and exact MIR reports remain preserved;
affected kernels were never launched. No forced register flags were used.

Verified first full 24-layer program resource preparations compile for B1/B16,
both decode/prompt, but fail the zero-stack gate: 464–512 B frames at MT168.
All are rejected and unlaunched. This differs from the component-only fits;
D7 integration/resource diagnosis is in progress, without register flags or a
512-thread exception. These are manual resource preparations, not solver plans.

Prepared GPU tests compare router weights and combine outputs bitwise against
actual legacy CUDA bodies, check deterministic dispatch, and compare grouped
GEMMs to the CPU FP32 reference. They have not run yet.

## Scope and next gates

CPU preparation proceeded while the last dense selections were still running;
this scheduling difference is recorded explicitly. It is not phase-E acceptance.
MoE GPU work waits for the dense 25-cell measurement cohort to finish. G-B and
G-LLM failures remain open and are not waived by this preparation.

Still missing: GPU numerical/publication tests for the prepared bodies, model StageProgram/bindings
and runtime, prompt ABI and counters, native compiler routing, measured cost
rows/top-four selection, C-1/C-2, routing checks, and five-cell E2E timing.
DNN starts after the required LLM acceptance stage; no DNN native result exists.

Raw evidence: `moe_analyzer_cpu_preparation_v1`,
`moe_router_cpu_preparation_v1`, `moe_grouped_cpu_preparation_v1`,
`moe_combine_cpu_preparation_v1`, `moe_legacy_reference_cpu_preparation_v1`.
