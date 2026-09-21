# BE-5 — the barrier inventory, and what role granularity would change

`cta_sync()` is `__syncthreads()`: it requires every thread of the CTA. Put one
inside a warpgroup's branch and the other warpgroup never arrives. This is the
inventory §5.2 asks for — every CTA barrier the harness executes, what it
protects, and what would have to take it over if a TaskBody specialized its
warps.

## The fifteen barriers

| # | site | line | protects | under role specialization |
|---|---|---|---|---|
| 1 | `WaitDependencies` | 644 | the acquire: every thread has observed its producer's epoch before any proceeds; paired with the `__threadfence()` on the next line | unchanged — no TaskBody code runs here, so every role is present |
| 2–3 | `WaitTaskDependencies` | 682, 686 | the same at task granularity, and the reuse boundary of the smem union between two consecutive tasks (§8.6) | unchanged, same reason |
| 4–7 | `NotifyTask` | 857, 865, 872, 930 | F-1's release sequence: every writer converges before one thread publishes the event | unchanged; this is the barrier the litmus re-validates at role granularity |
| 8–10 | `GridBarrier` | 958, 972, 982 | §8.2's stage barrier: monotonic epoch, publish, backoff poll | unchanged |
| 11–15 | `tilemega_l2_kernel` | 1085, 1149, 1161, 1222, 1233 | slot-window bookkeeping, the trace stamp, and the prefetch issue boundary | unchanged |

✅ **Verified by construction: none of the fifteen sits inside a TaskBody.**
Every one of them is in the harness, at a point between tasks or inside the
wait/notify protocol, where all threads of the CTA are executing the same code
whatever the body did. Role specialization does not make any of them
unreachable.

The barriers that *would* break are the ones inside the bodies —
`TILEMEGA_PHASE_SIMT_BARRIER()` in the norm and attention bodies, and the two
`__syncthreads()` inside `BlockReduce` (R8 BE-3/BE-4). Those are CTA-wide today
because every body is CTA-wide today. A body that splits into a producer and a
consumer warpgroup would need each of them replaced by a named barrier over
the participating role, which is what `TaskRoles` (BE-1) exists to declare.

## What the litmus decided

See `README.md`: the compliant arm passes every cell, the missing-fence
control fails every cell, and the missing-barrier control **does not fail** on
sm_89 under any construction tried. §5.3 requires both controls to fail before
§8.5 may be rewritten, so B-a is not met and, per §8.3, **§8.5 is left
unchanged and no TaskBody is specialized this round**. The `TaskRoles` ABI
stays declared and unused; nothing in the harness was converted to named
barriers, because converting a barrier whose replacement has not been
validated would be exactly the change the litmus exists to gate.
