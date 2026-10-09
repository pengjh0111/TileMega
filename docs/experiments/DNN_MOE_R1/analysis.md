# Exact task access (CI-4, work in progress)

Status: static analysis implemented and verified; execution integration pending. This is not a
synchronization, model-correctness, or performance result.

stated: DM-1 requires flattened pixel ownership, exact window/floordiv read
relations, bounded virtual spaces, table/counted dependencies, and storage
anti-dependencies. Existing decoder representations and default output must
remain unchanged.

inferred: an opt-in `SemanticOp.task_space/task_map` separates ownership from
stored tensor coordinates. Projecting the bounded iteration domain constructs
task-to-element maps; composing consumer reads with inverse producer writes
gives RAW dependencies. Halo elements outside the producer image create no
wait. Pixel shuffle changes the stored indices while retaining the producer's
linear ownership. Nonzero reduction origins use an optional shift inside
floordiv. Default zero shifts add no legacy JSON/text fields.

inferred: shared-element volume is a function of `(consumer, producer)`, not
one constant tile width. Physical rereads are the sum of per-consumer read
sets minus their union. TaskWork and DRAM queries consume the physical sets.
Per-axis boxes are explicitly `over`; they contain the exact read relation and
never replace it when deriving waits. `read_box_exactness` records this in CG.

inferred: fully bound small task domains can be counted one fiber at a time
with ISL. Equal integer counts are grouped into exact polynomial pieces.
This avoids parsing large Barvinok expressions for floor/mod projections.
Symbolic or larger domains retain Barvinok, and legacy callers retain `Card`.
The finite-fiber limit is a compiler algorithm threshold, not a device limit.

inferred: the table encoder linearizes the same task coordinates as ownership,
enumerates producer IDs for each consumer, joins contiguous IDs into intervals,
and proves both containment directions. Rows are padded to the maximum interval
count of that edge, including empty rows.

inferred: storage reuse derives WAR from each old reader's physical read set
and the new writer's overwrite set. Old writes whose elements have no reader
also contribute WAW. These relations preserve task granularity and require no
whole-stage fallback. The memory planner must first place all accesses in the
same physical storage coordinate space and separately enforce halo/layout
compatibility. These hazards transfer ordering, not tensor data.

verified: `results/CI4_exact_analysis.json` seals 11/11 host checks and an
unchanged source snapshot, including window enumeration, table equivalence,
TaskWork, shifted split origins, I2 row gathers, and WAR/WAW. `check-policy`
passes. Earlier parser/domain mismatches and the superseded expensive counting
run are retained under `runs/dm1-ci4-*`; no expected values were changed.
verified: exact window/table selection passes six additional host checks
(`results/CI4_bound_encoding.json`). A clipped halo window is recovered by
solving integer constraints for its unclipped endpoints, then proving both
containment directions. Sparse predecessor sets retain exact interval tables.
verified: closed-form `min` and its disjoint polynomial pieces pass eight host
checks (`results/CI4_capacity_form.json`). Expert-group capacities retain
`ceil(T*K/BM) + min(E,T*K)` before binding, tested across four BM values and
T=1..4096. Nested minima and negative floor values keep their original criteria.

verified: virtual ownership and provenance pass nine host checks
(`results/CI4_virtual_binding.json`). Both binding strategies cover every
capacity slot, retain their symbolic capacity, and preserve affine gate-up/down
couplings. Gather reads retain their source buffer and widen only the indexed
axis under I2. Binding requirements also propagate through storage hazards.

inferred: `virtual_bindings` records the logical live extent as
`runtime_dynamic`, its capacity, source, and `prefix_sum`/`tensor_values`
requirement. Coupling extent attributes describe the scheduled capacity space
(`symbolic_static`), whose affine internal edges are Tier 0. This separates a
runtime binding's live rows from its static task/event capacity without changing
the legacy five-attribute Tier rules. Runtime binding writers and consumers are
still pending; these host checks do not establish their synchronization.

verified: table and counted runtime primitives and synthetic stage materialization
are sealed in `synchronization.md`; automatic CG transport is under validation.

Pending: real-body table/counted writes and waits; memory-planner integration;
poison checks and all required 50-process synchronization gates. No CI-4 item
is considered fully complete until its required integration is verified.
verified: sparse runtime table projection and flow geometry rebinding pass
9/9 host checks plus 2/2 empty-row/corruption checks
(`results/CI7_table_flow_host.json`, `results/CI7_table_projection_empty.json`).
The projection binds the plan's workload shape, keeps each table's exact
producer set, and counts grouped events after owner elision. Flow search
rebinds a table from L-sem when tile geometry changes, including conversion
to an exact window. Empty table rows request no event. These are CI-7
foundations; the remaining body classes and structural search are pending.

## Counted data-dependent writes

verified: `ProjectTaskWrite` retains affine dimensions and uses I2 only for
binding-dependent store axes. RAW derivation marks these edges Tier 3,
`tensor_values`, relaxed and predecessor-uncountable. The conservative write
image is not an issued-byte estimate.

verified: logical contribution thresholds derived from consumer reads pass
96 host geometries including token tails. All 24 permutations of a four-row
binding are contained in I2; weighted arrivals equal the derived thresholds
while the number of distinct predecessor tasks varies. Duplicate/unsorted
unit axes and missing binding sources are rejected
(`results/CI4_counted_write_host.json`).

A contribution unit needs exactly one writer. Separate column partitions
require a partition axis in the unit or compatible producer/consumer tiles.
The primitive proof does not establish a dispatch permutation or a real-body
synchronization gate; those checks remain pending.

verified: the aligned-scatter helper reconstructs reads/stores from L-sem,
checks the binding-indexed unit axes and proves that every producer covers
exactly the consumer's remaining column range. All 96 geometries pass;
misaligned N partitions, unrelated sources and missing tensor accesses are
rejected (`results/CI4_counted_partition_host.json`). Dispatch must still
establish a permutation of the logical units; static I2 cannot prove it.

verified: the per-consumer threshold view passes host offset, tail, uniform
compatibility and 64-bit epoch overflow checks
(`results/CI5_counted_threshold_host.json`). Threshold storage is immutable;
counter storage remains separate for each executor bank. Native integration
now passes 150/150 fresh processes (`results/CI5_native_counted_threshold_cuda.json`)
and the standard CUDA ctest (`results/CI5_native_counted_ctest.json`). Real bodies remain pending.

verified: exact task arithmetic now pulls constants, reduction work and model
extents onto an identity task domain. Symbolic tail reductions, width ratios,
legacy scalar output and nonidentity rejection pass host checks with assertions
enabled (`results/CI4_task_arithmetic_host.json`). Device/model gates remain pending.

verified: mapped physical outputs retain their separate logical ownership
geometry. All 32 NCHW/pixel-shuffle cases pass split-K cardinality, FP32
partial-byte and flow-count checks with assertions enabled
(`results/CI7_mapped_store_host.json`). Mapped device writes remain pending.

verified: 19/19 focused native ctests and check-policy pass after counted CG,
CUDA emission and runtime-projection integration. Counts retain logical
contribution semantics; static ordering tables retain I2, including affine
column partitions. The CG test covers 96 shapes x 12 grid/kappa choices and
rejects malformed metadata. Host page-resource checks cover 76 geometries
and exact/insufficient target budgets (`results/CI4_CI7_native_host.json`).
verified: native geometry rebinding passes 216 candidate transitions and 12/12
focused ctests (`results/CI7_counted_flow_host.json`). Synthetic contraction
semantics carry the ownership geometry; executable MoE bodies remain pending.

verified: dependency provenance roundtrips pass 45 cases and 81 corruption
rejections, with 7/7 native checks including existing fusion regressions
(`results/CI7_dependency_transport_host.json`). Retained table encodings and
counted thresholds are checked against their exact relation/unit image.
verified: counted fusion event transport passes 96 bound cases, including two
contracts on one stage pair, reversed target numbering, grids 1/3/8 and
kappa 0/1/4/16 (`results/CI7_counted_fusion_projection_host.json`).
Counted CG endpoint rebinding and body lowering remain pending.


verified: typed fused dependency transport passes seven native checks and 192
counted geometry cases, including 96 cases with an independent sparse table
on the same stage pair (`results/CI7_typed_fusion_host.json`). Both endpoints,
reverse task permutations, one/two tensor contracts, grid 1/3/8 and kappa
0/1/4/16 are covered. Bound descriptor reconstruction retains the rederived
thresholds and exact table intervals. Existing CG fusion lowering tests pass.

inferred: the counted I2 envelope is necessary for placement but does not
expand ordinary fine-event waits. Fusion composes these two relations separately;
counted targets retain weighted counters even when their task owner is local.
Counted CG endpoint rewriting is verified below; new fused physical bodies remain pending.


verified: logical binding request cardinalities pass twelve native checks and
nine capacity/token geometries (`results/CI4_binding_requests_host.json`).
A row request is keyed by virtual block and local row; an expert weight request
is keyed by virtual block, so channel and reduction coordinates retain their
within-task reuse. Two scatter axes from the same binding share the same request
keys. Full/tail rows and columns, frontier reads and reduction extents pass.
The extent-one enumeration fixture was corrected to use named coordinates;
its byte counts and numerical expectations were not changed.

inferred: request coordinates describe issued accesses rather than a unique
physical image. Different requests may name the same physical row. I2 still
widens only the data-dependent physical axes for dependency derivation. These
raw capacity counts require routing occupancy before expert pricing; DRAM
provenance and real expert-body integration remain pending. Request metadata
serializes only when present, preserving legacy default representations.

verified: typed binding traffic passes 368 full/tail tile byte and provenance
checks (`results/CI7_binding_request_traffic_host.json`). BF16/FP32 weight
widths, produced/external gathers, scatter stores and live input footprints
use independent expectations. Mixed indirect producer/output provenance
rejects until its actual binding image is supplied. The earlier ISL sum
failure came from seeding a coordinate polynomial with scalar zero; the
implementation now sums actual fibers. Numerical expectations are unchanged.
Routing occupancy and solver/model integration remain pending.

verified: independent expert binding blocks and MMA row tiles pass 384 host
geometry cases with five malformed-description rejections per geometry
(`results/CI7_virtual_partition_host.json`). Request tails, dense/SwiGLU
column packing and virtual/row/N executor numbering retain independent
expectations. The earlier oracle omitted a still-present extent-one task
coordinate; binding all named coordinates fixes that fixture, without changing
counts. Frontend/solver wiring and real expert bodies remain pending.

verified: the DM semantic key passes 48 layer-name substitutions, seven
semantic distinctions and input immutability
(`results/CI7_semantic_signature_host.json`). Ownership, side stores, shared
bindings, layout relationships and aliases are canonicalized together.
Legacy class keys remain untouched; DM consumer wiring remains pending.

verified: complete macro capture passes three host tests, including eight
actual nvcc preprocessing phases across sm_89/sm_100 and two translation
units (`results/complete_macro_capture_host.json`). Included defaults,
architecture conditionals and unit-local macros are preserved; unsafe or
incomplete dry-run command records reject. No binary was compiled or launched.
Identity generator integration follows the active PageStream cohort.

verified: the compiler/pricing overlay passes 72 typed traffic cases, 8
physical-image cases, 256 virtual geometries and 768 symbolic bindings.
Coordinate arithmetic, typed side-output FLOPs and memoization pass the cost
contract; the unchanged legacy regime-A test passes 18 bit-exact checks with
its CMake-specified target (`results/CI7_binding_pricing_integration_host.json`).
The failed private-header, shared_ptr and wrong-target attempts are retained.
The committed validation worktree still needs native checks before integration.

verified: the completed CI7 typed-fusion regression preserves all eight CUDA,
ptxas-resource, SASS and 64-step token comparisons. Full candidate ctest passes
133/134; full unchanged-reference ctest passes 94/97 after ignored fixtures
were restored. The common independent-attention mismatch is unchanged.
Reference also retains six missing wait_protocol schema fields and the missing
norm_prologue_gemm executable. These failures keep G-REG false; record-only
queue completion never overrides them (`results/T1_CI7_typed_fusion.json`).

verified: native multi-page PageStream passes 50/50 fresh forward processes and
50/50 fresh prefill processes after the timeout was moved inside the shared
flock. All ten binaries and their macro/resource identities are preserved.
sm_89 executed kernels have zero spills; the sm_100 loop kernel spills
8-byte stores/216-byte loads for forward and 4/4 bytes for prefill. Other
compiled kernels have zero spills. The original lock-wait timeout and its
50/50 forward, 8/8 prefill partial cohort remain a failed audit. No timing or
model gate is inferred (`results/CI5_native_multipage_phases_cuda.json`).

verified: `results/CI7_exact_fusion_host.json` seals five legal NCHW/pixel-shuffle fusion candidates, ownership aliases, phase arithmetic and eleven rejected geometries. The additional B=3 shuffle fixture was initially predicted to meet the existing single-producer fusion contract; independent element enumeration proves that output rows 112..127 span producer row blocks 0 and 1. The failed run is retained and that geometry is separately checked as a rejection. Four unchanged legacy written-price cases retain six bit-identical fields each; legacy rewriting retains 1890 edge identities. External static metadata is verified below; counted-scatter fusion and actual fused bodies remain separate work.

verified: `results/CI7_fused_dependencies_host.json` seals ten external edges around five fused pairs. A row permutation forces one exact dependency table. Independent element enumeration agrees with rebased shared/read relations, and regenerated geometry/table metadata passes both inclusions. Event storage uses the complete producer linear-ID range, including unused IDs. Unchanged legacy checks retain 1890 rewrite identities, 122 batch identities and four bit-identical written-price cases. Counted-scatter fusion is not claimed by this test.


verified: `results/CI7_conv_counted_host.json` seals the committed integration:
seven fresh focused native ctests, build and check-policy pass after the prior
25/25 host suite and linked cost overlay. Convolution splits use issued MMA K
iterations, while L-sem retains logical (r,s,c). C=24, R=S=2, TK=16 issues eight
iterations, rather than the six obtained by flattening logical K; split 8 is
legal. Exact relations, task work, CG transport, runtime projection and concrete/
symbolic chunk pricing use the same partition. Both relation inclusions and
ownership uniqueness are checked; no numerical/model or timing gate follows.

verified: `results/CI7_counted_endpoint_fusion_host.json` checks 81 fused
producer/consumer geometries, 567 rejections and 20,898 independently enumerated
contribution pairs. Counted targets retain exact static thresholds and the I2
placement envelope after either or both endpoints fuse. Duplicate arrivals,
non-bijective phase maps and foreign event-name collisions reject. This is host
proof only, with no fused TaskBody or §8.A synchronization claim.

verified: the integer device mapping probe compiles for sm_80/89/90/100/120;
sm_89 checks 189 geometries and 324,076 slots, with all ptxas spill counts zero
(`results/CI7_conv_iteration_device.json`). It validates indexing only.

verified: `results/T1_CI7_integrated.json` retains eight identical CUDA/resource/
SASS/token comparisons, candidate ctest 144/147 and reference 94/97. The two new
candidate multipage executables were omitted by a restricted native build command;
the next checkpoint builds all candidate default targets. The common independent-
attention numerical failure remains. Record-only queue completion is not G-REG.
