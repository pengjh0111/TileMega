# A7 numerical primitive — not production integration

✅ 50/50 fresh GPU processes. Each tests 32 BF16 cases: grouped ratio 1/2,
seq 4/128, past 3/512, chunks 1/2/4/8; independently seeded input tensors.
All 1600 primitive cases pass the unchanged `1.6e-2 + 1.6e-2*abs(expected)`
criterion against the existing direct attention TaskBody. All 400 chunk=1
cases are bit-identical to that TaskBody; maximum absolute difference across
all chunks/seeds is .00390625. The frozen executable and sources are hashed
in `verification.json`; all 50 raw logs are retained.

This is not the requested 800-process complete-model/persistent-queue gate.
No L0.5/L1/L2 timing, race-free task-queue claim, complete-model resource
comparison or DP chunk ranking is inferred from the primitive test.

## Why four phases

The old `AttentionCombineTaskBody` wrote an integer iteration marker, not
an attention result. `AttentionTaskBody` has two dtype-rounding boundaries:
scaled QK scores and normalized probabilities before PV. Merging ordinary
online `(max,sum,weighted-value)` partials would lose the latter boundary.

The implemented prototype therefore computes chunked QK scores, normalizes
the complete query in the original order, computes chunked PV with those
same rounded probabilities, and sums FP32 partials before the final dtype
conversion. Chunk=1 will keep the original direct production path. Both
normalization boundaries and the full-query causal mask are preserved;
chunk>1 changes only FP32 PV summation association. The standalone combine
ABI now takes actual partials and an explicit shape, and the architecture
cross-compile fixture supplies live inputs instead of constant null pointers.

Four phases add a query-by-total FP32 score/probability buffer and
query-by-chunks-by-head-width FP32 partials. They also add three internal
dependencies and extra traffic. Those costs must be modeled explicitly;
the existing Stream-K combine rate does not cover score normalization.
Chunk scratch is sized for its KV interval in the probe, not the full total.
It is not yet the production `TaskSmem` union/occupancy result.

Code: `AttentionPlan.h:12`, `AttentionPhasedTaskBody.h:15`,
`AttentionCombineTaskBody.h:21`, `test/unit/attention_phase_test.cu:34`.
Reproduce with `ninja -C build-portable attention_phase_test` followed by
`run_attention_phases.py --out NEW_DIRECTORY`.

⚠️ Next required integration: per-attention plan choice, host stage expansion,
shared A2 projection, dependency and queue validation, compiled scratch and
global residency input, arithmetic/work signatures, pricing/DP enumeration,
then the complete-model acceptance matrix and paired timing.
