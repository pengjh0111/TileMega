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

verified: the complete MobileNetV1 graph now generates with all 13 DW/PW fused
pairs and `--reuse l2`: 18 arena bindings, 13,559,552 arena bytes. The result is
in `results/DN_mbv1_full_reuse_host.json`. It is host generation evidence; its
single fixed synthetic native replay is separately frozen and queued under the
shared lock in `runs/dm1-mbv1-full-reuse-native-v1`, with no performance task or
process matrix. The earlier full-generation timeouts remain preserved.
