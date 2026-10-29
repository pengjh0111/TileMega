# MX-5 MoE native development

Verified 2026-10-29. Authority: `TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
MoE native model acceptance remains **0/5**. Manual B1/B16 C1/C2 pass, while
routing fails31/468 non-tie positions. Final selected5-cell E2E remains pending.

## Host analysis

The initial verified CPU checkpoint built `AnalyzeMxMoe(ModelPlan const&)`
into the root library before routing integration. The current native route
is documented below. The analyzer reads
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
five actual bridge analyses pass. At that checkpoint native CLI integration
and canonical CUDA headers were pending; their later results are recorded below.

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
weights/combine use actual legacy CUDA value comparisons; GEMMs use CPU
FP32 references with unchanged tolerances, plus value-equality repeat checks.
Strict byte comparisons are a separately queued check described below.
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
to a finite sentinel, keeps the same entire-output value check/tolerances,
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

Initial verified summary (domain exclusions corrected below): all109 derived
grouped builds finish:102 fit/seven rejected.
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

Verified comparison-method correction: the earlier Python fixtures and
frozen grouped cost driver use torch.equal on floating tensors. This checks
value equality, including equal signed zeros, and is not raw-byte equality.
The earlier descriptions of those comparisons as bitwise are withdrawn;
their original rows, binaries, tolerances and measured times are retained.
Canonical fixtures now compare contiguous uint8 views; three CPU tests cover
signed zero, NaN payloads and dtype mismatch. Five fresh guarded processes
per fixture are queued; no new byte-comparison GPU pass is claimed yet.

Verified source-bound grouped cost aggregation validates recorded argv,
fresh PIDs, guards, raw hashes, build/source identity, resources and sample
quantiles. Prefix audits remain unqualified; even a complete 772-key cohort
is only a measured subset pending the 27 recovered keys. The aggregator
explicitly records value-repeat semantics and an unpassed byte body gate.
Twelve CPU tests pass. No complete-domain, model or default claim follows.

Verified CPU inventory reuses source-matching zero-prefetch bench binaries
for the new MoE qkv/o shapes (760 legal coordinates) and D128/Q8 attention
(200 coordinates). It makes zero device calls; new-shape device queries,
formula-prefetch resource closure and three-process timing remain required.
Failed zero-prefetch references alone will not exclude executed payloads.

Verified the card7 grouped-cost batch43/round2 guard rejects the job with
code75, reason exit interference. Its raw rows and failed process record
remain excluded. An independent registered controller reuses the valid
round1, records a new round2 directory and completes round3; its remaining
odd batches continue alongside the original card6 worker. Guard policy and
the bound measurement driver are unchanged. Raw: `moe_grouped_cost_card7_guard_recovery_v2`.

Verified new-shape qkv/o CPU resource closure:760 legal coordinates use54
source-matching formula-prefetch libraries;675 initially fit,20 recover by
the prescribed halving,65 fail at minimum depth. Attention contributes100
zero-frame512-thread coordinates; the100 normal576 primary coordinates
retain frame468–480B (PK1), so remain rejected. The795 fitting coordinates
pass per-kernel MIR checks: no calls/private accesses or generic/4160 steady
rings. This is not device, model or per-plan512 authorization. Raw:
`moe_regular_cost_preparation_v1`; device queries remain queued.

Verified formal-cost preparation adds source/query-bound router and regular
drivers. Fifty CPU checks pass across input tampering, resource-branch,
raw-sample and byte-comparison tests. The router cohort covers220 fitting
coordinates at dispatch TN64/64 and retains50 payload resource rejections;
this fixed-dispatch cohort is explicitly a subset. The regular cohort has
695 GEMM and100 attention coordinates, measured by frozen copies of the
unchanged dense drivers after actual queries. Three processes per key and
the795-coordinate cost queue are prepared, not executed or qualified yet.
Recovered grouped27 costs have their own queued supplemental epoch, with
the failed primaries and all halving steps retained in aggregation evidence.

Verified manual watchdog1 control preparation builds B1/B16 decode/prompt
plans with MT237/241/245/249, frame/static0 and matching native ABI/identity.
Their actual zero queries and512-step checks are queued after the router
cost allocation on card6; this is manual integration evidence only, not a
solver choice, per-plan512 exception or model acceptance. Raw:
`moe_manual_watchdog_controls_cpu_v1`, `moe_manual_correctness_after_router_costs_v1`.

Verified isolated D7 CPU experiment anchors the attention O-zero definition
after Q/new-value prefix inputs. Normal576 B1/B16 code changes, but resources
remain MT168/frame464 and480B. It does not repair the resource failure;
the candidate stays off, root headers and all measurement epochs unchanged.
No GPU runs or allocator/512 conclusions follow. Raw:
`moe_attention_zero_lifetime_D7_cpu_v1`.

Verified the cell script now routes explicit MoE control pins to native CLI,
builds decode watchdog twins and uses the decode bridge for prompt64. Six
CPU script-routing tests pass. Requests for the incomplete measured solver
fail before the old script branch. The full pipeline audit was reconciled
with actual native implementation and the remaining model/cost gates.

Verified canonical host MoE domain enumeration now matches all2226 keys of
the independent CPU inventories exactly:760 qkv/o,896 grouped,270 router,
300 head across both thread shapes. It preserves per-expert N separately
from the active-expert partition size. The fixed five-point attention
interpolator averages L65..1087; constant, linear, nonlinear analytic-weight,
nonfinite and overflow checks pass. Normal root builds and both MoE host
CTests pass. These are solver-core facts, not a complete table/CLI choice.
Raw: `moe_canonical_domain_match_cpu_v1` (immutable frozen-v2 comparison).

Verified MoE head N151936/K2048 matches the existing Qwen1.7B head shape.
All archived head source dependencies match current headers. The150
qualified512-thread rows (five B ×30 coordinates) retain their original
source names, rounds and three-process evidence; no new GPU measurements
or relabeling occurs. This reuse scope excludes576 and grants no plan
thread exception. Raw: `moe_head_existing_cost_source_audit_cpu_v1`.

Verified router and regular source-bound cost aggregators now validate the
actual guarded commands/PIDs, raw hashes, resource queries, reference builds,
three-process quantiles and selected MIR. Regular samples additionally bind
the executed wrapper/helper invocation and unchanged dense oracle/tolerances.
The combined127 CPU contract tests pass; automatic prefix/full aggregation
is queued after the corresponding actual batch proofs. No new full table
or model acceptance is claimed before those measurements exist.

Verified native pair checks cover prompt64/decode phases, batch/capacity,
KV storage,145/9217 barriers and all246 external names/dtypes/sizes/recipes.
Seventeen tests include four actual WD0/WD1 control pairs. The routing audit
uses this check before loading GPU weights. Each future top4 candidate must
own its matching prompt/decode pair when TN changes; a prediction0 prompt
cannot be shared across incompatible candidate recipes. The intended build
order is prompt before decode within each candidate, then select the pair.

Verified initial grouped772 cost keys have three fresh-process rounds each;
source-bound aggregation passes. Thread counts are443 at512 and329 at576;
the recovered27 all belong to576, giving final fitting counts443/356 and
resource rejections5/92. The geometric448 keys per shape must not be read
as448 fitting keys. All original records remain; supplement timing is queued.
Raw: `moe_grouped_cost_aggregation_after_complete_v1/initial772.json`.

Verified the native program ranking layer now scales six per-layer costs
and the once-per-step head cost, preserves original row indices, and matches
router dispatch TNs to selected grouped bodies. It partitions the Cartesian
product by dispatch context before exact k-best merging. Forty randomized
brute-force comparisons include ties and context incompatibility; both MoE
host CTests pass. This does not qualify missing costs, implement the complete
CLI, grant512 exceptions or select an accepted model. Raw:
`moe_canonical_domain_match_cpu_v1/ACTUAL_DISPATCH_CONTEXT_PROGRAM_RANKING_CPU_COMPLETE.json`.

Verified router timing coverage is currently the explicit TN64/64 subset.
The benchmark now accepts compile-time gate/up and down dispatch TNs; default
values stay64/64. CPU-only normal builds for the other11 combinations are
running with frozen sources, per-kernel resources/MIR and lossless archived
compiler logs. They have no cost/model acceptance yet. Raw:
`moe_router_contexts_cpu_v1`. No pending measurement input was modified.

Verified strict raw-byte tests now pass for each of the three canonical
fixtures in five fresh processes,15 distinct child PIDs total. Router/combine,
router GEMM and grouped fixtures all report raw_bytes_equal; original
float-equality records remain unchanged. This is representative unit evidence,
not the model, routing or synchronization litmus gate. Raw:
`moe_byte_exact_after_costs_v1/ACTUAL_THREE_FIXTURES_FIVE_FRESH_BYTE_EXACT_COMPLETE.json`.

Verified recovered27 grouped keys now have three fresh rounds and pass source-
bound aggregation. The canonical896 grouped coordinates are covered by799
measured fitting rows and97 normal-build resource rejections (45 gate/up,
52 down at minimum prefetch). CPU outcome coverage is not full-model acceptance.
An initial CPU audit counted12 CSV fields as13 and stopped before producing a
catalog; its failed record is retained. Corrected v2 matches all896 coordinates
exactly. Raw: `moe_grouped_complete_outcomes_cpu_v{1,2}`.

Verified native cost-row import checks one requested batch/thread shape,
executor/body identity, three rounds, resources, grouped n_active and router
TN context. Attention requires the complete five-point grid and matching
source/binary/resources before averaging. Fifteen negative contracts and both
host CTests pass. All922 existing grouped/head aggregate rows import through
ten actual batch/thread requests. Import itself does not verify files, qualify
missing domains or authorize execution. Raw: `moe_cost_row_import_actual_cpu_v1`.

Verified the576-thread MoE head shape has130 fitting coordinates and20
resource rejections. Historical jobs21–27 contain those fitting keys but have
no timing rounds; this is outside the150-row512 reuse scope. A new frozen
130-key source/normal-command/resource-branch check passes, with three-process
head timing queued after regular795 costs on card7. No costs are backfilled.
Raw: `moe_head576_cost_scope_v1`.

Verified the host coverage checker enumerates all six GEMM domains, all12
router dispatch contexts and the20 attention grid coordinates per request.
It reports missing coordinates and rejects duplicated/conflicting outcomes,
measurement fields in rejections and rejections with fitting resources.
Synthetic complete/mixed domains and both host CTests pass;949 actual fitting
grouped/head rows plus97 actual grouped rejections pass through ten CPU
requests and correctly retain missing other classes. This is geometry coverage,
not file/branch qualification or plan authorization. Raw:
`moe_cost_coverage_actual_cpu_v1`.

Verified actual cucc dependency preprocessing for grouped, router, regular
GEMM, head and attention identifies current adopted project header bytes
exactly matching the frozen computation bodies. This uses the include graph,
so unrelated host solver edits do not falsely invalidate body-source identity.
No GPU launch or cost/model gate follows from preprocessing. Raw:
`moe_body_source_closure_cpu_v1/ACTUAL_FIVE_BODY_HEADER_CLOSURES_MATCH_ADOPTED_CPU.json`.

Verified TN64/64 router220 fitting keys now have three fresh timing rounds
and pass the frozen source-bound aggregator. This remains one dispatch context;
all11 additional contexts still require their own build/query/cost records.
Raw: `moe_router_cost_aggregation_after_complete_v1/complete220.json`.

Verified the first manual B1 model check stopped in serving/memory.py:48
because its dtype-size table omitted pair_u32 (two uint32 values). No model
kernel ran in that attempt. Adding its8-byte size passes13 CPU accounting
tests and four actual plan-manifest checks; binaries and GPU bodies unchanged.
The next attempt completes512-step C-2 with zero L1-separate/loop mismatches,
then HF C-1 fails because MetaX torch's grouped_mm is unsupported. That
unsupported backend is also recorded in torchcompile_baseline.py:70–87;
no external baseline is remeasured or changed. Both failed guards/logs remain.
Raw: `moe_manual_correctness_after_router_costs_v1`,
`moe_manual_correctness_memory_recovery_v2`.

Stated user amendment explicitly permits experts=eager for this round's HF
correctness and routing reference. The CLI now accepts an explicit choice,
asserts HF uses it and records it; grouped_mm stays the default when no choice
is passed. mx_check_modes forwards MX_MOE_HF_EXPERTS only when explicitly set.
No automatic implementation fallback or threshold changes were added.
Thirty-two CPU reference/routing/pair tests pass. The pending old route CPU
controller was superseded before changing its bound helper, with no GPU job
terminated. Fresh eager-reference manual C1/C2 and routing are queued; existing
cost and isolated rank-experiment queues retain their frozen inputs.
Raw: `moe_hf_eager_authorization_v1`, `moe_manual_correctness_eager_v3`,
`moe_manual_routing_eager_v2`.

Verified fresh guarded manual controls now pass B1 and B16 C-1/C-2,512steps
each. Native L1-separate/loop mismatches are0/0 for both batches. HF eager C-1
gap≤.5 ratios are1/1, maximum gaps .0625/.109375, on512/8192 positions;
required ratio .99 and allowed maximum3 remain unchanged. Four actual rows
with round/executor/implementation/result/guard hashes are appended to
`MX5_moe_controls_c550.csv`, explicitly manual_control_unselected. These
results neither choose top4 nor grant per-plan512 or final G-MOE acceptance.
Raw: `moe_manual_correctness_eager_v3`.

Verified manual B1 routing audit fails:1536 positions,1382 exact sets,123
boundary ties,31 non-tie failures. B16 also fails:24576 positions,21826 exact,
2282 ties,468 failures. The two-BF16-ULP criterion is unchanged; C1/C2 success
does not satisfy this separate gate. Diagnostic token prefixes match the
watchdog controls. For B1,22 of31 failures have an earlier same-token set
mismatch,9 do not; the maximum HF rank8/rank9 gap is10ULP. This correlation
does not establish a cause. Origin remains unlocalized; failing gate:
`python/tilemega/moe/native_routing_check.py:128–139`.
Both failed guards and full mismatch lists remain, with two append-only rows
in `MX5_moe_routing_c550.csv`. Raw: `moe_manual_routing_eager_v2`,
`moe_manual_routing_b16_eager_attempt_v3`.

Verified the waiting diagnostic rank CPU controller was superseded before
any of its GPU jobs. Its replacement waits for both actual route attempts
to be recorded, then runs independent frozen unit/microbench work; it does
not turn the failed model route gate into a pass. The rank candidate stays
off and subsequent resource/cost queues remain diagnostic. Raw:
`moe_router_rank_predicate_after_route_attempts_v2`.


Verified the equivalent integer rank predicate passes two legacy-comparison
fixtures in five fresh processes each. Its isolated microbench improves B1
TileRange/StreamK medians14.80%/14.55%, B16 by23.26%/23.62%; these alone
were not adoption evidence. Eight normal whole-program candidate plans and
all eight MIR reports pass with frame/static0. Candidate B1/B16 C1/C2 pass
512 steps and match original control tokens. Same-card ABAB P-PAIR, each
arm two fresh processes with warmup1/timed2, gives5.215872→5.025410ms at B1
and20.010283→19.264265ms at B16 (−3.65%/−3.73%), both discernible. All1024
tokens match. The measured predicate is adopted in source; eight process
rows are appended to MX5_candidate_ppair_c550.csv. These are manual controls,
not selected plans, a512 exception or G-MOE acceptance. Target defaults stay
unchanged. Raw: moe_rank_whole_program_candidate_v1.

Verified the old11-context router GPU queues were stopped while waiting,
before dispatch. Their completed CPU artifacts and original source costs are
retained. The adopted predicate requires a new source epoch:1296 normal CPU
libraries/6480 kernel outcomes cover all12 dispatch contexts, then6480 actual
device attribute queries and three fresh cost rounds per fitting coordinate.
The new queues retain R_cap220/248 and explicit TN-dependent U_gu/U_dn.
Raw: moe_router_rank_all12_contexts_cpu_v1,
moe_router_rank_all12_queries_v1, moe_router_rank_all12_cost_scopes_v1.

Verified regular795 stage costs now complete three fresh processes and the
source-bound aggregator. Together with165 actual resource rejections they
cover960 regular/attention coordinates. The grouped896 geometry similarly
contains799 measured costs and97 resource rejections. Neither partial-domain
coverage nor manual P-PAIR authorizes final model selection.
Raw: moe_regular_complete_outcomes_cpu_v3,
moe_grouped_complete_outcomes_cpu_v2.

Verified matched-input router diagnosis finds1507 exact sets,29 boundary ties
and0 non-tie failures on1536 B1 positions when the native router receives the
same HF pre-RMS inputs. This localizes a distinction from whole-model route
failure without proving its upstream cause. Installed HF eager casts routing
weights to BF16 and uses BF16 index_add destinations; the exported contract
keeps F32 routing weights and FP32 rank-order combine. Controlled diagnostics
using FP32 combine leave25 non-tie failures, and additionally export F32
routing weights leave23, versus31 in the formal eager reference. These are
diagnostic reference variants only; no formal reference or threshold changed,
and neither explains all failures. The first counterfactual attempt's failed
dtype assumption and logs remain. Raw: moe_matched_input_router_D7_v1,
moe_HF_routing_dtype_combine_counterfactual_D7_v2.


Verified complete-geometric-outcome candidate serialization now connects the
importer, coverage checker and context-aware exact top4 merge. Each candidate
retains seven class costs and original aggregate row indices (all five length
points for attention), explicit native pin geometry and a mandatory matching
prompt-pair marker. Partial outcomes or fewer than four legal programs are
rejected. Two host CTests pass, including serialization/source-index mapping
and layer/head cost checks. Output is diagnostic and explicitly grants no
file qualification, execution, thread exception, selection or default policy.
Actual1744 existing costs plus262 rejections pass all ten batch/thread coverage
requests; router contexts and head576 are correctly missing. Raw:
moe_regular_grouped_head_coverage_cpu_v2.


Verified the normal native compile command now routes a MoE decode bridge and
an explicit geometry-outcome table to the host candidate preview, preserving
moe_batch. Six CPU CLI tests use synthetic costs with the actual24-layer
bridge, assert145 stages and exact four candidates, retain attention's five
source points, reject incomplete domains/results/interpreted flags and preserve
an existing preview. No device compiler, kernel or model acceptance is implied.
Both host CTests pass. Ten actual partial1744-cost/262-rejection requests reject
without writing a plan or preview. The CPU domain tool also emits structured
coverage counts/missing coordinates for the pending complete table producer.
Raw: moe_regular_grouped_head_coverage_cpu_v2,
moe_actual_partial_cli_preview_rejection_cpu_v1.

Verified the adopted-source dependency check passes again for all five body
families, including the new router header. Context32/16 has540 passing actual
zero-launch queries and220 fitting cost keys now measuring. Context pipelines
use one guarded card6 job at a time while card7 completes130 head keys; source
and round protocols are unchanged. Twenty head576 terminal rejections are
verified against their actual queries and normal compiler records. CPU-only
collectors wait for all head/router contexts, then verify exact raw round rows,
quantiles and complete geometry for ten batch/thread scopes (5396 outcomes).
These queued collectors do not predeclare cost/model/thread-policy acceptance.
Raw: moe_rank_adopted_body_source_closure_cpu_v2,
moe_router_rank_context_pipeline_v2, moe_head_complete_outcomes_cpu_v1,
moe_router_rank_complete_outcomes_cpu_v1, moe_complete_geometry_table_cpu_v1.


Verified the eight P-PAIR summary rows initially left nested source-head/digest
and E2E metadata blank. MX5_moe_rank_provenance_c550.csv separately reconciles
those fields from actual identity/measurement files, with original CSV line
and hashes. Original measurement rows and values remain unchanged; this is
metadata reconciliation, not additional timing or a backfilled old row.


Verified shared dense-host regression checks exposed changed validation order:
the new MoE dispatch initially attempted bridge loading before rejecting an
unqualified dense table. The fix preserves dense early rejection and only
bypasses its parser for the explicit MoE geometry-table kind. The original
failure log remains. The dense host suite passes on rerun, as do seven MoE CLI
tests including rejecting a MoE table applied to a dense bridge. Device bodies,
cost snapshots, reference numerics and thresholds are unchanged. Raw:
moe_regular_grouped_head_coverage_cpu_v2/shared_dense_host_ctest_v2.log,
moe_regular_grouped_head_coverage_cpu_v2/cli_preview_pytest_v3.log.


Verified head576130 completes three fresh processes and the source-bound
aggregator. Together with the unchanged512 reuse150 and20 actual576 terminal
rejections, all300 head outcomes are complete. Combined regular/grouped/head
coverage is2156 outcomes (1874 measured costs,282 resource rejections). Ten
structured CPU audits each report exactly324 missing router coordinates and
no missing non-router coordinates. Raw: moe_head_complete_outcomes_cpu_v1,
moe_nonrouter_complete_geometry_cpu_v3.

Verified the new MoE-only occupancy helper validates exact native symbols,
Plan geometry, ELF/same-compile MIR, identity/receipt and host shared-byte
query before querying MACA attributes. It reuses the existing runtime ABI and
guard checker without changing the dense helper. Seventeen CPU tests pass,
including sixteen actual program/context combinations, negative bindings and
recording an injected loader failure without overwriting its result. A guarded
card7 job queries all eight normal adopted-rank manual programs with zero
kernel launches: MT235–249, frame/static0, resident1. This is not model timing
or a512 exception. Raw: moe_plan_occupancy_helper_cpu_v1,
moe_whole_plan_queries_after_head_v1.

Verified four normal adopted-rank minimum576 controls (B1/B16 prompt/decode)
compile successfully at the device-compiler level, then are rejected by the
zero-stack gate: B1 prompt/decode frames512/464B, B16 prompt/decode464/480B,
allMT168/static0. The corresponding actual zero-launch queries report
resident1 and identical attributes for all four. Occupancy success does not
override their positive-frame rejection; these observations do not claim
occupancy0 or a global576 impossibility. Twelve actual adopted-rank program
resource/query rows are appended to MX5_resources_c550.csv; no old row is
modified. Minimum controls are not selected plans or granted512 exceptions.
Raw: moe_rank_adopted_min576_programs_cpu_v1,
moe_adopted_min576_zero_queries_v1. A missing CPU PYTHONPATH attempt stopped
before request creation/GPU dispatch and remains separately logged.


Verified new router context GU-TN32/DN-TN16 completes220 measured costs from57
fresh processes and50 actual resource rejections, covering270 outcomes. The
strict source-bound collector and CPU geometry collector pass; context32/32
continues independently. Raw: moe_router_rank_context_pipeline_v2/g32_d16,
moe_router_rank_complete_outcomes_cpu_v1/g32_d16_geometry_outcomes.json.

Verified an isolated raw-value snapshot diagnosis targeted the earliest formal
B1 non-tie failure, decode step24/layer1 (HF boundary gap3ULP). It reuses45904
of49152 debug bytes, disables ordinary route-debug writing and adds only raw
copies after existing stage releases; no adopted body/reference changed. The
first compile fails on a load-helper namespace error; its header/patch/logs
are retained. Correcting that name compiles, but the normal resource gate
rejects MT250/frame2016B at
runs/mx5/moe_failed_route_value_snapshot_D7_v1/tree/include/tilemega/Codegen/tasks/mx/MoeProgramV2.cuh:146.
No snapshot kernel or HF comparison ran, no plan-complete marker was written,
and no raw intermediate result or causal explanation is claimed. The failed
resource row is appended to MX5_resources_c550.csv. Root remains unchanged;
route failure and G-MOE remain open. Raw: moe_failed_route_value_snapshot_D7_v1.

Verified108 completed old-router MIR logs retain byte-identical readable paths
and persistent gzip backups, freeing736.515MiB. Current source/binaries,
measurements and guards are untouched by this archive. Raw:
moe_completed_old_router_mir_archive_v1.


Verified a diagnostic replay uses the normal full24-layer native B1 request's
actual checkpoint/KV history at decode step24/past88, then invokes the existing
native stage bodies for layers0/1 (12 calls). All actual queries report resident1,
frame/static0; both replayed routes equal the original full-program control.
The GPU capture completes, but its original job exits1 on a CPU-only ULP-stat
condition dtype error. That failure/guard remain unchanged. Separate CPU-only
recovery reads the saved tensors, fixes the boolean condition and reproduces
the formal layer1 failure (HF rank8/rank9 gap3BF16ULP). The new down wrapper's
MIR is checked after capture:10 functions/2 rings/0 failures. These are raw-value
diagnostics, not cost-protocol timing, a new full-model result or a route pass.
Raw: moe_first_two_layers_replay_D7_v1,
moe_first_two_layers_replay_cpu_recovery_v2.

Verified the layer0 embedding input is bitwise identical to HF. QKV is already
different: maxabs.015625, relativeL2.0025998894. A CPU FP32 diagnostic using the
prompt's folded-BF16-weight/consumer-RMS rounding reproduces native QKV with
maxabs2.3841858e-7; the HF activation-RMS/original-weight rounding reference
matches actual HF with maxabs.001953125. The two reference contracts differ by
relativeL2.0025997139, close to the actual QKV difference. This localizes the
earliest observed difference to existing RMS/folding/rounding placement at one
position; it neither replaces the native MMA summation tree nor attributes all
31/468 formal route failures. Arithmetic, oracle and acceptance remain unchanged.
Raw: moe_qkv_rounding_attribution_cpu_v1.

Verified a guarded card7 diagnostic feeds the same native post-O residual to
unchanged HF Qwen3MoeRMSNorm/Qwen3MoeTopKRouter components, with explicit eager
reference configuration. Layer0's set matches; layer1 remains a different set,
but HF rank8/rank9 gap is now2BF16ULP (1.421875 versus1.40625), within the
unchanged tie rule. The full-HF-input comparison at that position remains the
original3ULP failure; this isolated changed-input diagnostic cannot replace it.
No source/model/threshold changed. Raw: moe_same_native_input_HF_router_D7_v1.

Verified317 completed grouped MIR logs retain byte-identical readable paths
and persistent gzip backups, freeing386.736MiB. No source, binary, cost sample
or guard result changes. Raw: moe_completed_grouped_mir_archive_v1.


Verified the reusable CPU cost-evidence checker validates original raw lines,
three rounds, aggregates, source/binary hashes, coordinates including router
TN context, actual resources, guard/PID records, working-set size and body
validation flags. The original512 head campaign has450 command records across
150 keys, without child-PID metadata; these remain explicitly labelled as
legacy command evidence and no PIDs are invented. Twenty-four CPU tests pass,
including changed body/source/scope, reused PIDs, altered raw rows, failed
body/guard checks and incomplete coverage. Actual existing2696 outcomes pass:
2314 measured costs plus382 resource rejections across regular/grouped/head
and the first two new router contexts. This file check alone grants neither
geometry completeness nor model/thread/default acceptance. The complete-scope
path additionally checks the adopted five-family header closure and actual
native geometry audit; ten real complete scopes are queued until all5396
outcomes exist. Raw: moe_cost_provenance_cpu_v1,
moe_complete_scope_file_verification_cpu_v1. The initial unit fixture's wrongly
calculated median and the first family-table schema rejection remain logged;
the fixture arithmetic and explicit supported family schemas were corrected,
without changing any measurement row or acceptance threshold.

Verified the final four router contexts (GU-TN128, all four DN-TNs) are assigned
to guarded card7, with byte-identical frozen query/ABI/measurement drivers and
the same source epoch. Its collector changes only the expected physical card.
Card6 retains the first eight contexts; all per-context results retain their
original paths for the existing outcome collector. The frozen original serial
controller will stop on its already-owned ninth context directory before any
duplicate GPU launch; its old all12 completion marker will not be claimed.
Completion is instead checked from all12 actual per-context aggregates. This
changes scheduling only; each key still uses three fresh processes on one
card,40960MiB admission,900s timing limits, at most cards6/7 concurrently.
Raw: moe_router_rank_card7_last4_pipeline_v1.


Verified original candidate preparation now replays the exact native host
ranking against the same bridge/table/producer and checks the complete file
and body-source evidence before writing jobs. It rejects edited pins, costs,
ranks, missing prompt pairs and changed table bytes. Every original candidate
gets its own prompt/decode pair at watchdog0 and1, sharing its actual pin.
The combined host integration/provenance suite passes37 CPU tests. No device
compilation or model acceptance follows from those synthetic host fixtures.
A frozen CPU queue waits for all real cost/preview/evidence markers, then
prepares and normally compiles80 diagnostic plans (5B×4 candidates×2phases×2WD),
records resource/MIR outcomes, validates actual prompt/decode bindings when
both fit and prepares zero-launch query inputs. Resource failures are retained;
no fallback, query/kernel launch,512 exception or final selection is applied.
Raw: moe_candidate_build_preparation_cpu_v1,
moe_original_top4_pairs_after_complete_cpu_v1. A missing parenthesis in the
initial queue script fails CPU parsing before queue creation; its source/log
remain. The corrected queue parses and waits without submitting any build.

Verified extraction of twelve existing normal-program compiler logs finds all
eight512-thread adopted-rank controls report staticMaxWarps/PEU2 with frame0;
the four576-thread minimum controls report3 with MT168/frame464–512B. This is
compiler-report evidence only, not proof of a hardware register limit, an
allocator defect, a minimum body requirement or a per-plan512 exception.
R_cap220/248 and the prompt's227/255 resource limits remain unchanged.
Raw: moe_576_effective_register_budget_probe_cpu_v1/
ACTUAL_TWELVE_NORMAL_PROGRAM_PEU_RESOURCE_RECORDS_CPU.json.


Verified a separate CPU-only synthetic control holds144/160/176/224 scalar
operands simultaneously with empty inline asm, using normal cucc flags and
no register-limit override. All four512 controls compile with frame0
(MT168/194/204/226);576 controls144/160 compile with frame0 (MT146/162).
The576 controls176/224 do not finish within the bounded diagnostic interval;
their exact own CPU mxcc children are terminated at about282s, with action
records, and cucc returns1. These two results are interrupted diagnostics,
not compiler resource-rejection or hardware-limit evidence. No kernel runs,
GPU process, model body or acceptance policy changes. Raw:
moe_576_effective_register_budget_probe_cpu_v1/
ACTUAL_SYNTHETIC_PROBE_OUTCOMES_CLASSIFIED_CPU.json.
