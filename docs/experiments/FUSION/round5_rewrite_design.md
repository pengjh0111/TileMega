# L-task rewrite contract

Implementation design, not completed GPU acceptance. A solver decision names
adjacent logical tasks. The pass must not select that decision heuristically.

The replacement is a real fused task-space operation, not a placement flag.
It carries ordered phase semantics, consumer-to-phase coordinate maps, exact
external physical reads/writes and task count. Keeping separate phase
semantics preserves MMA versus SIMT arithmetic and each BF16 rounding boundary.
External users of the producer output retain that output and its write.

Rewire incoming/outgoing edges by composing the old relations with the phase
coordinate maps; remove only the internal edge. Recompute wait and inverse
fanout on every changed relation, and verify their summed cardinalities.
Event tensors and placement attached to removed tasks are invalidated, not
silently reused. Runtime projection and lowering must consume the replacement
before a fused module can execute; a module lacking that implementation is
rejected explicitly. The standalone pass is separately testable with
tilemega-opt, including transactionality on rejection.

The existing first-layer GEMM residual epilogue is already fused at runtime.
Rewriting its logical pair alone is not a new event or traffic benefit.
Likewise a GEMM/add/RMSNorm chain cannot be tested by dropping the residual
add. The interval solver and execution composition must retain every phase.
