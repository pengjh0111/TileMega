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

Pending: executable table and counted waits; memory-planner integration;
poison checks and all required 50-process synchronization gates. No CI-4 item
is considered fully complete until its required integration is verified.
