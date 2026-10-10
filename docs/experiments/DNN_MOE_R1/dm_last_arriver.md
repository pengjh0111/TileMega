# DM reduction handoffs

verified: global pooling, MoE selection/dispatch and counted combine can be
elided in L1, L2 and paged executors. The host planner reverses the proved
window/table dependencies into producer arrival lists. It only selects a
handoff when every owner task already guarantees all other producer stages
are complete. Stage order alone never satisfies this condition. The host
checks cover image-segmented pool arrivals, the multi-stage dispatch chain,
weighted combine and rejection of an independently unready input.

inferred: counted down-to-combine arrivals contribute the number of live
rows for each distinct (token block, column block) target. All compute lanes
traverse the same targets so the final arrival can run a cooperative reducer.
Empty binding tiles contribute nothing. A bounded recursive handoff publishes
elided events in L2; L1 omits their stages and barriers. Binding-aware page
loaders resolve an elided dispatch event to its physical owner. Epoch tickets
use separate L1/L2 banks and existing release/acquire last-arriver primitives.

verified: one fixed 17-token grouped MoE input passes nonpaged and paged
L1/L2 execution, with maximum error 0.0078125, complete route agreement and
same-binary bit equality. Each build selects two reducers. A separate B=2,
TM=16, split-K=3 pool fixture crosses image boundaries and selects one reducer;
maximum error is 0.0016505123 in both modes. These results and ptxas spill
records are joined by identity in `results/CI5_dm_last_arriver_generated.json`.
The initial native builds failed because the new view was in the wrong
structure; the failed v1 artifacts remain preserved. No tolerance changed.

verified: `--global-la auto|0` controls pool handoffs. Explicit
`--nonpaged-la 0` or `--paged-la 0` disables the respective generic handoffs;
opaque MoE also disables them. DNN configuration accepts these controls and
`paged_la_splitk`; MoE serving configuration enables nonpaged handoffs by
default while preserving explicit overrides and dense LLM defaults.

These are single-process functional checks, not 50-process race evidence.
No latency was measured. Structural solver pricing and full-graph handoff
coverage remain unfinished; no joint-search or complete model gate is claimed.
