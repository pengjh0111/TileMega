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

Verified root `EmitMxMoeProgramV2` now emits all six typed layer stages and
head from actual plan dimensions and class coordinates. Configuration checks
cover grouped TileRange, the gate/up no-prefetch edge, router WN1, head WN4/8,
wait immediates, and shared memory. Host tests add 120 emitted decode/prompt
programs, 240 configuration rejections and 120 buffer-tile rejections;
paired prompt/decode external tables are equal. Root compiler build and the
five actual bridge analyses pass. Native CLI integration and canonical CUDA
headers remain pending; these tests do not compile/execute the emitted CUDA.

## Device body preparation

Verified isolated normal cucc builds, with exact-function MIR checks:

| Body | Instances | Result |
|---|---:|---|
| Router/top-k/dispatch | 20 | zero frame/static SMEM/generic/calls/private |
| Grouped gate/up and down | 12 | same; includes incoming down prefetch and combine LA |
| Combine | 16 | same; rank-ordered arithmetic mirrors legacy |
| Router GEMM plus tile/route LA | 8 | same; TileRange/StreamK, BF16 logits then dispatch |

Verified canonical bodies now live in `include/tilemega/Backend/Mx/` and the
program/runtime ABI in `Codegen/tasks/mx/`. They are not accepted model kernels.
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

Verified the subsequent normal 512-thread resource references and separate
ABI preparations have MT235/243/237/241 for B1 decode/prompt and B16
decode/prompt, all with frame/static SMEM zero. Exact MIR checks pass all
16 MMA rings per kernel with zero generic/calls/private/4160 violations.
These are manual CPU preparations, not selected plans, occupancy observations,
model correctness or authorization of a 512-thread exception. The normal
576 failures remain. CPU-only elision diagnostics isolate attention as the
spill contributor; no body is elided in any execution-eligible program.
Twelve manual resource rows are append-only records in MX5_resources_c550.csv.

Prepared GPU tests compare router weights and combine outputs bitwise against
actual legacy CUDA bodies, check deterministic dispatch, and compare grouped
GEMMs to the CPU FP32 reference. Router/combine and router-GEMM fixtures now each pass five fresh processes
on card7. Grouped fixtures also pass five fresh processes. The registered queue waited
for all 25 dense measurements and the new Llama8BB2 D7 ledger before five fresh
processes per fixture; MoE jobs request 40,960 MiB and keep the 1,800 s limit.

## Scope and next gates

CPU preparation proceeded while the last dense selections were still running;
this scheduling difference is recorded explicitly. It is not phase-E acceptance.
MoE GPU unit work began after the dense 25-cell cohort and D7 ledger finished. G-B and
G-LLM failures remain open and are not waived by this preparation.

Verified canonical B1/B16 decode/prompt programs compile from the root emitter;
normal device code is byte-identical to the earlier ABI preparations. All four
have zero frame/static SMEM and exact MIR checks pass; MT235/243/237/241.
This CPU integration does not establish a 512-thread exception or GPU acceptance.

Still missing: validated model StageProgram, bindings and runtime, prompt ABI
and counters on the GPU, measured cost
rows/top-four selection, C-1/C-2, routing checks, and five-cell E2E timing.
DNN starts after the required LLM acceptance stage; no DNN native result exists.

Raw evidence: `moe_analyzer_cpu_preparation_v1`,
`moe_router_cpu_preparation_v1`, `moe_grouped_cpu_preparation_v1`,
`moe_combine_cpu_preparation_v1`, `moe_legacy_reference_cpu_preparation_v1`.

Verified native CLI now sets moe_batch, routes DM MoE directly to the native
analyzer/emitter, rejects interpreted flags, and accepts --mx-prompt 1 only
for 64-token prefill from a decode bridge. Without a measured solver/pin it
errors; it cannot fall through to legacy lowering. Explicit pins remain
manual/resource probes and never claim solver/default/final acceptance.
Actual CLI B1/B16 decode/prompt builds share one source digest, all 246
external rows/recipes agree within each pair, identities verify, L1 execution
identities bind mx_moe_kernel_v2, L2 is rejected. Resources remain
MT235/243/237/241, frame/static zero; device code equals the root emitter
preparations. CPU route/resource tests25 and identity/feature tests19 pass.
Native CLI 576 minimum-prefetch B1/B16 builds retain MT168/464B and
MT168/480B failures, including with ALLOW_RESOURCES=1; no identity is
issued for these failed artifacts. Neither failure authorizes 512 execution.
Raw: `moe_native_cli_cpu_v1`, `moe_native_cli_576_resource_failure_v1`.

Verified all three GPU body fixtures pass 5/5 fresh processes: router/combine
100 cases each, router GEMM8, grouped GEMM12, no skipped cases. Router
weights/combine use actual legacy CUDA bit comparisons; GEMMs use CPU
FP32 references with unchanged tolerances, plus bitwise repeat checks.
These do not count as model C1/C2 or a 104-CTA/2000-iteration litmus.
Independent raw/PID/guard/hash verification appends15 body-check rows.
Six actual native-CLI program queries launch no kernels: normal512 artifacts
have zero local/static memory and one resident CTA; failed576 artifacts
report their 464/480B local memory and remain rejected. Six append-only
resource rows retain exact source snapshots and binary/pin hashes.

The first grouped-stage bench driver stops at repeat comparison key1 after
one diagnostic key. Its output used uninitialized inactive rows; NaNs there
are an inferred harness explanation, not proven from the missing tensors.
The old driver/log/partial row remain; a new attempt initializes all output
to a finite sentinel, keeps the same entire-output bitwise check/tolerances,
and saves both tensors if comparison fails. Bodies and binaries are unchanged.
No row from this prototype is a qualified cost or model performance result.

Verified finite-initialized bench retry passes18 keys in each of3 fresh
processes, including unchanged full-output repeat and FP32 checks. TN64/D1
example medians (us): GU B1/WN1 42.02, B16/WN4 345.45; DN+combine B1/WN1 40.10, B16/WN4 216.76.
These remain diagnostic prototypes, not qualified solver costs. No GPU body
changed. Raw: `moe_grouped_stage_bench_gpu_v2`; first failed attempt remains.
Canonical test libraries now build from committed test/unit/mx sources; all
five device-code images equal the originally executed fixtures. The portable
pytest entry additionally runs all3 canonical fixtures successfully in one
guarded job (three child processes). Prior5-fresh evidence is retained.

Verified router-stage bench preparation:15 keys ×3 fresh processes pass
CPU FP32 GEMM, actual legacy top-k/weight bit comparison, dispatch bounds
and whole-output repeat checks. TN32/D1 B1/B16 medians (us) are TileRange/Pw0 50.27/132.29,
StreamK/Pw0 48.18/131.39, StreamK/Pw1 49.67/131.30.
The preparation uses runtime GEMM shapes and constexpr dispatch batch; its
dispatch unit factors are TN64/64, recorded in each key. These are diagnostic
values, not model M4 timing, solver costs or proof of a MoE TPOT result.
All15 kernel instances have zero frame/static memory, no private/calls or
steady-ring4160; MT118/151/152 by TR0/SK0/SK1.

Verified grouped R_next epoch freezes headers and bench source before builds:
208 configurations cover896 legal batch coordinates. Normal CPU builds
fit183/reject25;799 eligible batch zero-queries confirm API0/frame0/static0/
resident1, with zero kernel launches. Formula-derived down prefetch builds
are in progress (109 builds for486 batch coordinates);313 gate/up
coordinates retain no incoming prefetch. R_cap remains220/248. No qualified
cost timing or per-plan 512 exception follows from these resource checks.

Verified all109 derived grouped builds finish:102 fit/seven rejected.
459 additional zero-launch device queries pass, so the896-coordinate
domain has772 eligible outcomes and124 resource rejections. All317
normal builds are append-only in MX5_body_resources_c550.csv; rejected
instances remain unlaunched. The formal source-bound cost campaign uses
129 batches ×3 fresh processes, six keys per batch except the tail.
Its first batch passes all three processes; two guarded card7/card6 workers
continue. Old prototypes remain separate; no final table/solver/default
acceptance is claimed from these incomplete samples.

Verified B1 normal576 decode preRA diagnostic recompilation leaves device
code byte-identical to the failed native CLI artifact (MT168/frame464).
The existing virtual-liveness estimator reports231 32-bit units at its
maximum over1414 CFG blocks. This ignores alignment/STR/earlyclobber/path
feasibility and is neither a mathematical lower bound nor proof of an
allocator defect or a 512-thread exception. No register override or GPU
execution is used. Raw: `moe_actual576_preRA_D7_v1`.

Native routing-audit preparation is in scripts/mx_moe_routing_check.py and
moe/native_routing_check.py:65 generated tokens cover64 decode inputs,
read the64×24×B×8 diagnostic buffer, release native weights, then hook HF
mlp.gate with grouped_mm on the identical teacher-forced history. The ULP
convention is BF16 spacing at the HF eighth logit magnitude (subnormal
spacing2^-133); both gap/ULP and code distance are reported. No routing GPU
audit has run. Ten boundary tests and16 existing routing/helper tests pass.
An initial edit displaced the existing routing_check.py API; it was restored
byte-for-byte and the native audit isolated in a separate module before GPU
use. New source-bound cost-input/tamper tests8 pass; normal root builds pass.

Correction to the initial124-rejection domain summary:45 were mandatory
Pw0 gate/up rejections,52 were only failed Pw0 down references, and27
were failed first down payloads awaiting the prescribed halving branch.
The references do not by themselves reject an executed nonzero-Pw body.
All52 missing primary coordinates now compile normally at their formula
Pw=D and still fail; all27 coordinates in seven halving cases recover
zero-frame builds. This gives799 fitting builds and97 final benchmark
resource rejects; the recovered27 still need device queries and timing.
Twenty normal compile attempts append without changing any older row.
The already measured772-coordinate scope remains valid as a subset.
Its frozen epoch retains the initial124 count and must not be relabeled
as complete-domain evidence. Final catalog coverage must include this
closure and the recovered-coordinate samples. Raw: `moe_grouped_resource_closure_v2`.

Verified router full resource preparation finishes108 libraries/540 batch
kernels (Pw0 and Pw=D):440 fit/100 reject under normal compiler flags.
Device queries for these kernels and the27 recovered grouped coordinates
are queued after actual grouped-cost card7 batch127 completes; this
retains the max-two-card/one-job-per-card constraint. No third job starts.

Verified seven-class MoE selection mathematics is implemented separately
from cost qualification:720 brute-force exact-k-best comparisons cover
legal-combination filtering/tie ordering; two-process medians and the exact
1% boundary pass. Normal root build and both MoE host CTests pass.
The CLI is not yet a measured MoE solver; no selected/default claim follows.
