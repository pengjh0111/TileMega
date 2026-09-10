# Standalone Fusion Writeback

Verified: `tilemega-opt --tilemega-fuse-task-pair='producer=l0.s05.rope
consumer=l0.s06.append'` replaces the two production L-tasks by one
`tilemega.fused_task_space` for both gqa2 and mha4. One internal edge is
deleted, external edges are composed with the phase maps, event tensors are
recreated, old implementation/placement records are invalidated. A second
standalone optimizer invocation verifies the resulting IR.

Code: `lib/Dialect/CouplingGraph/FusionPass.cpp` (`Rewrite`, `FuseTaskPair`),
`CGDialect.cpp` (`FusedTaskSpaceOp::verify`),
`test/unit/fusion_rewrite_test.cpp`. The final all-in-one test receipt is in
`../rewrite_verified/transaction_and_incidence.txt`.

Verified: 1890/1890 new-graph wait/fanout incidence checks over the original
A1 seq={1,4,128,512,2048}, past={0,3,512} matrix; six rejection branches
leave zero references. Rejection clones the module before mutation, so a
nonadjacent request leaves the original text unchanged. False fused task
count is independently rejected. Changed edges also pass symbolic
sum(wait)=sum(fanout) checks in the pass.

Not complete: interval DP does not yet select this pass, and fused runtime
projection/TaskBody lowering is not implemented. Codegen and concrete
ModelDescription reading reject the new operation explicitly. These guards
prevent an old model_plan from silently executing the unfused program;
they are not an implementation fallback. No GPU fusion is claimed.

## Corrections

The first runner looked for the wrong dialect prefix (`cg` instead of
`tilemega`) and stopped despite successful writeback. `../rewrite/` retains
that failure. No numerical expected value changed.

An additional whole-graph symbolic IsZero check failed on untouched edge c3.
Its difference contains floor(S/128) and floor((S+127)/128); ISL's structural
zero query does not establish the mathematical identity. This was a failed
proof attempt, not measured bad coupling. The changed edges retain exact
symbolic checks; the complete graph uses the original A1 parameter matrix.
An initial test hand-bound only a subset of export parameter names, leaving
mha4 aliases free. It now uses ModelDescription::MetricBindings, so every
export-specific alias follows the same checked source as production.

Do not generalize this pair to GEMM/add/RMSNorm by removing the residual add.
The existing residual epilogue remains identified as already fused.
