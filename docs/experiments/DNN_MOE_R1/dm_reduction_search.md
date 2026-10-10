# DM reduction structural selection

stated: DNN structural search compares pooling LA with a separate stage. MoE
search crosses slot/group BM16/32/64/128, MMA/GEMV, and independent dispatch and
combine LA bits (40 families). Explicit no-LA and opaque controls constrain the
domain. The selected plan carries pool=1, dispatch=2, combine=4 in an optional
`dm_reduction_mask`; absent fields preserve archived LLM code generation.
Generated macros, manifest and build identity retain this selection. The solved
module also records the number of reducers admitted by the proof.

stated: the solver and runtime use the same templated qualification proof.
Every candidate binds its own exact task dependencies, including runtime
window/table widening, counted I2 contracts and split-K stage ownership.
Incremental flow candidates reconstruct these relations from their local
couplings; they do not require a missing full projection. Phase-gated paths
remain ineligible for this ordinary completion proof. An independent incoming
producer cannot be dismissed by stage order.

verified: the host FIFO simulator retains body setup/barriers, memory and
compute for DM handoffs. The final producer's worker remains reserved until the
body completes; multiple handoffs serialize and take precedence over a newly
ready ordinary sibling. Legacy simulator defaults are preserved. The stage
flow model keeps the same body work and a bounded CTA pool; it is a capacity
approximation, rather than an exact worker replay.

inferred: arrival atomic/fence cost uses the target's existing publication
latency as a proxy. Static arrival multiplicities come from the reverse proved
relation. Counted target multiplicity uses the conservative I2 envelope, with
a maximum per price piece; routing observations do not establish legality.
This can overprice a counted handoff. It is not an empirical body fit. Base
piece prices are restored before applying each mask so incremental reuse does
not accumulate arrival charges.

verified: six host combination checks pass: shared proof, stage flow, production
MoE flow, 40 MoE structures with final CUDA, DNN structures with final CUDA, and
policy (`CI7_dm_reduction_selection_host.json`). A H128 regression selects TN32
from a TN128 seed. Down-column changes reimport combine ownership, counted
thresholds and statistics storage; other incremental candidates remain enabled.
The original public-CLI alignment failures are retained under
`runs/dm1-la-gemv-search-cli-v3` and `v4`.

verified: the repaired public CLI completes selection, final materialization
and actual kernel residency query. It selects group BM16, conditional GEMV/MMA,
dispatch LA and independent combine (mask2, one proved reducer), grid128 and
residency1. Source/manifest/resource hashes are in the host receipt. The v5
query correctly rejected an artificial two-SM fixture; v6 binds actual device
resources without weakening that guard.

verified: one fixed synthetic T17/H128/I64/E16/K8 input passes the selected sm89
megakernel, max error0.0078125, complete route agreement and L1/L2 bit equality
(`MO_selected_reduction_gemv_native.json`). Direct L1/L2 entries have no spills,
2480-byte stacks and236/232 registers; the auxiliary stage kernel spills8/8 bytes.
The full artifact is therefore not labelled zero-spill. The solved source's
architecture guard rejects80/90/100/120 recompile attempts; these are retained
failures, with earlier five-architecture TaskBody receipts reused. No new device
architecture branch was introduced by this host proof/selection change.
No performance measurements, repeated real-weight checks or new synchronization
pass-rate claims are part of this work. Earlier fixed-input LA runtime receipts
remain in `results/CI5_dm_last_arriver_generated.json`.
