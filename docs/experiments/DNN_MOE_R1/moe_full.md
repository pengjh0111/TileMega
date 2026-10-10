# Full-depth construction and execution entry

stated: the user removed performance measurements, real-weight gates and repeated
input/process matrices. The full-model entry remains available for a future device;
it collects no latency data and is not executed with full weights in this round.

verified: `python -m tilemega.moe.full {dry-build,build,check}` exports the real
configuration, binds batch and past ranges, checks checkpoint headers/recipes,
and emits plan and source identities. Three CPU unit checks cover the deployment
allocation estimate and compiler arguments. `scripts/run_qwen3_moe_full.sh` passes
shell syntax validation; `TILEMEGA_DRY_ONLY=1` stops after host construction.

verified: `runs/dm1-moe-full-dry-v2` generated the full 48-layer E=128 B=1 decode
and prefill CUDA and validated their checkpoint recipes. The B=16 decode phase
then exceeded the enclosing 900-second host timeout. This is partial dry-build
coverage, not a completed full-model check. The new B=16 attempt uses the repaired
disjoint cardinality implementation and is recorded separately.

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
fit. Attention/head recipes can remain phase-specific; general automatic shared
layout constraints are still pending.

The 80 GB script downloads/checks the original checkpoint, constructs plans,
then invokes the serving ABI directly for prefill and decode, compares same-binary
L1/L2 tokens, and optionally runs the existing teacher-forced HF check. It does
not call the timing-enabled `ServingEngine.generate` method. Full-weight execution
and complete script execution remain unverified.
