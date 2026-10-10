# Full-depth construction and execution entry

stated: the user removed performance measurements, real-weight gates and repeated
input/process matrices. The full-model entry remains available for a future device;
it collects no latency data and is not executed with full weights in this round.

verified: `python -m tilemega.moe.full {dry-build,preflight,build,check}` exports the real
configuration, binds batch and past ranges, checks checkpoint headers/recipes,
and emits plan and source identities. Nine CPU unit checks cover deployment
accounting, compiler arguments, native architecture/resource binding and rejection
when the allocation estimate exceeds available memory. `--target auto` probes
under the shared lock and overlays device resources onto its architecture profile;
it records that the retained calibration is not newly measured on this device.
The execution script defaults to this binding and checks memory after dry construction,
before native compilation. `scripts/run_qwen3_moe_full.sh` passes
shell syntax validation; `TILEMEGA_DRY_ONLY=1` stops after host construction.

verified: `runs/dm1-moe-full-dry-v2` generated the full 48-layer E=128 B=1 decode
and prefill CUDA and validated their checkpoint recipes. B=16 decode subsequently
generated in `runs/dm1-moe-full-dry-v3`. Full B=16 prefill now generates in
`runs/dm1-moe-full-b16-prefill-host-v12` (241 GEMMs, 1594 buffers). A new packed
decode uses its shared TN/TK in `runs/dm1-moe-full-b16-decode-shared-host-v1`.
All four plans pass header-only FQN/shape/dtype/packing-extent checks; their
paired allocation reports pass (`results/MO_full_four_plan_host.json`). The
prior row-major decode/tiled-prefill pairing is retained as an incompatible
recipe failure, and older host timeouts remain. This completes host construction
coverage, not a native full-model build, execution or G-MOE.

verified: the first dry build exposed incompatible router and gate/up recipes:
decode folded the post-attention RMSNorm gamma, while prefill did not. Both MoE
phases now use the same fold when deferred normalization is enabled. Eight full
prefill plan/producer-semantic host cases pass. One fixed synthetic two-layer
paged-prefill case passes native-input component numerics and L1/L2 bit equality
(`results/MO_prefill_shared_norm_smoke.json`). Whole-decoder HF errors remain in
that receipt and do not become a successful end-to-end gate.

inferred: B=1 at capacity 1088 needs 66,781,605,888 bytes of recipe-deduplicated
packed weights and an estimated total allocation of 67,347,137,584 bytes. The
estimate includes both plans' workspaces and request state, but excludes CUDA
context/event/queue tables and runtime-inserted split-K storage. It cannot certify
fit. Attention/head recipes can remain phase-specific; capacity-limited automatic
shared-layout constraints have host coverage described below.

The 80 GB script downloads/checks the original checkpoint, constructs plans,
then invokes the serving ABI directly for prefill and decode, compares same-binary
L1/L2 tokens, and optionally runs the existing teacher-forced HF check. It does
not call the timing-enabled `ServingEngine.generate` method. Full-weight execution
and complete script execution remain unverified.

verified: the public serving CLI accepts `features.prefill_pg=measure` and
`features.prefill_executor=measure`, with independent prefill paging/executor
selection. The prefill winner's resolved mode is recorded in the decode serving
sidecar. Existing defaults retain their previous nonpaged prefill choice.
Four host tests cover the candidate domain, fixed selections, content-keyed
cached winners and invalid configuration; mocked measurements launch no GPU.
No prefill latency selection was executed in this round.

verified: packed-only DM dense weights now replace their row-major buffer slot
when every consumer uses the same packed geometry and no activation, epilogue
or exported output observes the source. Legacy LLM packing is unchanged.
The router plus expert region passes one fixed synthetic paged L1/L2 execution
with exact reference output/routes and equal mode bits; identity/resources are
in `results/MO_packed_only_weights_generated.json`.

verified: the full B=1 manifest estimate reaches the memory rejection on this
device without loading model weights or compiling another binary.
`runs/dm1-moe-full-memory-preflight-v1/memory-preflight.json` records estimated
allocation 67,347,137,584 bytes versus 50,480,283,648 available bytes. The estimate
is inferred; available memory is queried under the shared lock. This does not
prove that a smaller allocation would fit.

inferred: shared layout constraints are keyed by canonical complete packing
recipes, not by tensor names. `--shared-weight-layout REF.plan.json` limits
matching GEMMs to the reference TN/TK in fixed builds and all solver candidates,
including projected paged seeds. An expert recipe mismatch is an error; a
phase-specific dense projection with different norm folding retains a separate
allocation. The full-model entry applies this restriction when its target's
DRAM capacity is below the inferred two-layout budget, and records the decision
plus reference-manifest SHA256.

verified: host candidate/search checks and generated fixed packed-layout
comparison pass in `runs/dm1-shared-layout-host-v1` and
`runs/dm1-shared-layout-cli-host-v2`. The first paged search failed because its
projected seed ignored the constraint; the repaired search passes all four
forward/token-axis and nonpaged/paged cases in
`runs/dm1-shared-layout-forward-host-v2.log`. Fixture preparation v1 failed on
a null FX shape before invoking the compiler; its correction is retained as v2.
No GPU numerical or performance matrix follows from these host checks.
verified: the final candidate-domain and forward-search checks also pass in
`runs/dm1-shared-layout-host-final-v1`; eight full-entry Python checks pass in
`runs/dm1-full-entry-python-final-v1.log`, including known/unknown capacity and
the exact capacity boundary.

verified: the regular serving CLI also selects decode first when the checkpoint
bytes, an additional BF16 expert layout and KV/token/RoPE state already exceed
known target DRAM capacity. The capacity decision is inferred and records its
config/index hashes; workspaces are omitted from this lower bound. The selected
decode manifest constrains all prefill candidates and their cache keys. The
final sidecar uses the subsequently selected prefill executor. Other models,
unknown capacity and sufficient capacity preserve prefill-first ordering.
Three host tests pass (`runs/dm1-shared-deployment-cli-host-v2.log`), including
the exact memory boundary, selected-artifact propagation and the existing
three-round selection/cache protocol with mocked measurements. No timing ran.

verified: the full-model C-2 checker now keeps one engine/weight allocation for
both L1/L2 launches of its fixed prompt. It clears request KV/tokens before each
mode and rejects unwritten/out-of-vocabulary tokens. Nine full-entry host tests
pass (`runs/dm1-full-single-weight-check-host-v1.log`), including state reset,
supported-mode checks and invalid tokens with mocked launches. This change has
not triggered another full real-weight or GPU replay.
