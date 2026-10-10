# MoE binding and occupancy integration

verified: production model flow consumes explicit runtime-binding contracts.
Private rows, binding tables, expert intermediates and dispatch workspaces retain
capacity/provenance declarations; they do not acquire fabricated physical
write images. Gathered hidden/statistic reads use the complete producer image
justified by K assignments per token. Active expert FLOPs use exactly T×K rows.
Without a profile, unique expert weights retain the explicit K-expert lower
bound. With a profile, nonzero expert occupancy is an exact rational expectation;
cardinality kind and binding provenance remain separate in the floor metadata.

verified: five binding families (slot and group BM16/32/64/128) rebuild the same
FX graph and rescore geometry, placement, residency and page coordinates. The
shared soft budget, errors and retained top-M predictions enter the sidecar;
the selected structure and hash enter the manifest/identity. Binding/BM is a
global deployment choice. Host tests use synthetic resource coefficients;
this does not establish measured latency or a native selected-plan result.
`CI7_moe_binding_profile_host.json` includes independent actual scalar body
resource compilation: RMSNorm, small dispatch and combine, maximum58 registers,
2560 shared bytes, zero spills; RMSNorm has a48-byte stack frame.

verified: the public compiler verifies profile content/file SHA256 and the
recorded HF capture identity chain. Token/layer/expert domains must match the
plan. Profile content enters the build cache key and manifest/identity.

verified: conditional expert pricing is connected to production flow. Pieces
refine static ownership by complete virtual-row occupancy histograms and row
subtiles. Two-way containment and complete counts prove coverage. Conditional
traffic/service and isolated cost are averaged before taking expectations.
The cache includes complete histogram contents, avoiding equal-mean aliasing.
650 unit cases plus production split-K/final-plan checks pass
(`CI7_moe_occupancy_flow_host.json`). The initial whole-axis coordinate rejection
and null incremental-module attribution crash were repaired and retained.

inferred: with timing work removed, absent empty-body fits use an explicitly
labelled full-capacity isolated-cost surrogate for fixed cost. Empty tasks issue
zero priced compute/DRAM; this surrogate is neither a new calibration nor a
latency prediction verified on hardware. The strict public pricing API still
rejects absent fits unless the caller explicitly selects this policy.

verified: binding families now cross MMA/GEMV and four independent dispatch/
combine LA masks, for 40 host-tested structures (`dm_reduction_search.md`). Actual
hybrid resource compilation reports 128 registers and 12288 shared bytes; its
out-of-line body has 296-byte spill stores and 416-byte spill loads. Scalar
GEMV pricing is analytic and labelled inferred, not a new empirical body fit
(`moe_gemv.md`, `CI7_moe_gemv_family_host.json`).
