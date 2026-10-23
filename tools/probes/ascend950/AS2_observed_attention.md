# Observed-operand attention replay

`as2_attention_snapshot.py` extracts actual independent eager QKV and prior KV
from a captured model forward. Current-position KV is cleared before replay;
the native Body computes it. Canonical Q/K gamma and position-only RoPE tables
retain their original hashes. Auxiliary partial/LSE buffers remain present.

The generated probe queues are explicitly **outside C-0**. Full-model plans
still require serving import, CTG/ISL and actual scheduling/materialization.

```bash
source scripts/as2_env.sh
python -m tilemega.bench.as2_attention_snapshot prepare \
  --binding "$REPO/runs/as2/models/qwen3_0_6b/native_request_newton_fold_v1" \
  --reference "$REPO/runs/as2/correctness/qwen06_first_cache963_eager_diag_v1" \
  --layer 11 --softmax-fold --out "$REPO/runs/as2/probes/NEW_CASE"
# Replace reference/binding above with the actual independently captured pair.
# Build one target with nice10/parallel1 in AS2_WORK/build/probes:
nice -n 10 cmake --build "$AS2_WORK/build/probes" \
  --target as2_observed_attention --parallel 1
# Run both the helper's reference subcommand and the native binary through
# as2_run, sequentially, on the same selected physical card.
python -m tilemega.bench.as2_attention_snapshot compare \
  --case "$REPO/runs/as2/probes/NEW_CASE" \
  --native "$REPO/runs/as2/probes/NEW_NATIVE_RESULT" \
  --reference "$REPO/runs/as2/probes/NEW_REFERENCE_RESULT" \
  --out "$REPO/runs/as2/probes/NEW_COMPARISON.json"
```

Verify actual reference names in the saved `case.json`; use a new output path.
Reference reuse requires `--reference-case` and matching every operand hash,
config/weights, epsilon, position, layer and observed expected context. It is
recorded as reuse, never counted as another device reference execution.

**verified-run**, 2026-10-23, physical0 Ascend950PR_9579, CANN9.1.0,
CATLASS `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`:

- At actual past1025/layer11, Q/K/V and scores match bitwise. Baseline has8114
  differing FP32 probabilities,596 PV values and one BF16 context value.
- Row max/exp match. Standalone `exp.sum` differs from the fused softmax's
  reduction and is not substituted as its expected denominator.
- Installed arch35 softmax uses a register fold before summation. Explicit
  variant6 uses the pinned device-level `CalculateReduceSum(...,1088,1024)`;
  original FP32 exponential/division and output cast remain unchanged.
- At actual past1025 and181, all Q/K/V/scores/probabilities/PV/context match
  bitwise after that change. These are numerical replays, not new C-3 claims.
- Raw cases/results: `runs/as2/probes/observed_attention_p{1025,181}*`.
  Exact binary/source/headers: `tilemega-as2/build/retained/observed_attention_*`.

The subsequent full-model Qwen B1 1024-token and teacher512 results are indexed
in `docs/results/AS2_950pr.md`. No replay or hooked reference admits performance.
