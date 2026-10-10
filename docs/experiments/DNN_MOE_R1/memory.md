# DNN reusable storage

verified: the host planner implements none, greedy and L2-budget arena policies,
256-byte offsets, layout/fill compatibility, lifetime exclusion of external,
state and observed output buffers, and exact reader-to-writer / writer-to-writer
couplings. All layouts and budgets come from the materialized plan and target.
A smaller occupant retains earlier readers/writers for any uncovered tail.

verified: `runs/dm1-memory-plan-host-v3.log` independently enumerates BF16 element
addresses for ordinary reuse and large-small-large reuse. The original storage
hazard oracle also passes (`runs/dm1-storage-hazards-host-v3.log`). Both contain
exact relation equality checks. These checks do not prove CUDA synchronization.

implemented, device validation pending: CG alias offsets, runtime arena binding,
initialization and single-owner destruction, and generated anti-dependency waits.
No GPU poison check, 50-process result or complete model reuse gate is claimed.

stated: no performance measurements are requested. `--selection predicted`
retains the solver's prediction path and forbids a measurement command. The
original measured selection remains the default for existing LLM builds.

verified: the generated seven-GEMM arena fixture passes a fixed synthetic
numerical check in L1 and L2, including three exact anti-dependencies and
zero-error memcheck (`runs/dm1-memory-generated-native-v1/events/memcheck.log`).
Maximum absolute output error is 0.0001220703125. This is an execution check,
not a 50-process synchronization claim. Further repetitions were cancelled
under the user's 2026-10-10 scope update. Full-model reuse generation still
requires work on exact-metric representation; prior timeouts are retained.
