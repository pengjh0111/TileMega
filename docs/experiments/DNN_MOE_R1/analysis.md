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
