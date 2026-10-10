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

verified: CG alias offsets, runtime arena binding, initialization, single-owner
destruction and generated anti-dependency waits execute in the fixed cases below.
No GPU poison check, 50-process result or complete model reuse gate is claimed.

stated: no performance measurements are requested. `--selection predicted`
retains the solver's prediction path and forbids a measurement command. The
original measured selection remains the default for existing LLM builds.

verified: the generated seven-GEMM arena fixture passes a fixed synthetic
numerical check in L1 and L2, including three exact anti-dependencies and
zero-error memcheck (`runs/dm1-memory-generated-native-v1/events/memcheck.log`).
Maximum absolute output error is 0.0001220703125. This is an execution check,
not a 50-process synchronization claim. Further repetitions were cancelled
under the user's 2026-10-10 scope update. Full-model reuse generation and its fixed native smoke now pass; prior
timeouts are retained.

verified: the complete MobileNetV1 graph now generates with all 13 DW/PW fused
pairs and `--reuse l2`: 18 arena bindings, 13,559,552 arena bytes. The result is
in `results/DN_mbv1_full_reuse_host.json`. Its single fixed synthetic
native replay now passes in L1/L2 with identical output bits (minimum cosine
0.9999978542, maximum FP32 error 0.0005717725). Identity and all resource/spill
lines are in `results/DN_mbv1_full_reuse_native.json`; no performance task or
process matrix was run. The earlier full-generation timeouts remain preserved.

verified: the seven-GEMM split-K reuse fixture executes in L1/L2 with maximum
error 0.0001220703125 (`results/CI5_reuse_split_endpoints_native.json`). Its
storage edges retain main/combiner ownership separately from RAW edges. The
L2 kernel has 255 registers and aggregate spill stores/loads 68/200 bytes.
