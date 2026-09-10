# A7.3 production attention chunk validation

✅ 800/800 fresh BF16 processes: gqa2/mha4 × seq 4/128 × chunk 1/2/4/8 ×
50 rounds. The runner rotates the complete state list each round and refuses
inherited `TILEMEGA_*` overrides. Every row has a fresh log, hash, timing,
resource and schedule records; 800 logs and an immutable build manifest are
retained. `ctas_per_sm=2`, `block=128`, and `task_smem=24576` in the archived
resource records. No sm120 claim is made.

The production host plan expands attention only when `chunks>1`; chunk 1
uses the existing fused TaskBody. Chunked plans carry explicit per-stage
records through import, codegen, ModelSpec, host allocation and symbolic
projection. Internal dependencies are the exact three phase edges. The host
checks chunk scratch capacity before launch and rejects invalid plans.

The complete-model runner currently records timings and correctness. A paired
rank/CI analysis and a large-context attention-share comparison are separate
follow-up measurements; this result deliberately does not infer them from
the 50-round medians. The unchanged BF16 comparison criterion is used; no
tolerance was modified.

Source locations: `include/tilemega/Codegen/AttentionPlan.h`,
`include/tilemega/Codegen/tasks/AttentionPhasedTaskBody.h`,
`include/tilemega/Codegen/tasks/ModelHarness.cuh` host expansion and checks,
`lib/Codegen/Codegen.cpp` runtime table emission,
`lib/Solver/RuntimeProjection.cpp` phase/task projection.
