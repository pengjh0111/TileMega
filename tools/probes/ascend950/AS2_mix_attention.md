# ABI4 cooperative FP32 attention probe

`python/tilemega/bench/as2_mix_attention_probe.py` constructs bounded **probe
queues**, not a model import or Solver result (`tilemega_pipeline=false`). It
runs the same generated `as2_llm`, `ExecuteLlm`, CATLASS tile bodies and AIV
phases used by the experimental `cube-fp32` candidate. Whole-model acceptance
separately requires `as2_compile_llm emit-diagnostic/emit-priced` output.

The probe uses recorded, independently computed Qwen0.6B first-layer QKV,
GQA=2/head128, G=8 and verified position-only NPU RoPE constants. Prefill owns
four different query tiles per group; decode reuses workspace for four epochs
with growing valid KV. Each owner alternates BF16 K=14336 tiles with FP32 QK/PV
using one CATLASS resource bank. Only one MMAD object is alive per phase.

Independent reference: `torch_npu.npu_rms_norm`, explicit BF16 RoPE multiply/add
cast points and eager `scaled_dot_product_attention`. Future public KV remains
zero and is checked in full. Decode's initialized reference prefill cache is
**probe-only**, not evidence of full TileMega generation.

ABI4 ready/done values reserve four tokens per owner ordinal. Phases are
prepare → QK → softmax → PV → output, with three GM handshakes and reverse
acknowledgement. AIC alone publishes logical completion, once. AIV0 participates;
unused AIV1 exits without waiting. No device locks or hardware flag assumptions
are introduced; AS-1's conservative DMA/cache visibility protocol is preserved.

Reproduction uses new canonical paths below `runs/as2`:

```bash
source scripts/as2_env.sh
python -m tilemega.bench.as2_mix_attention_probe prepare \
  --source "$REPO/runs/as2/correctness/qwen06_prefill_reference_fused_diagnostic" \
  --rope "$REPO/runs/as2/probes/qwen06_rope_table_v1" \
  --root "$REPO/runs/as2/probes/NEW_MIX_PROBE"
# Run the helper's reference command through as2_run on physical1, then:
python -m tilemega.bench.as2_mix_attention_probe seed-decode \
  --root "$REPO/runs/as2/probes/NEW_MIX_PROBE" \
  --reference "$REPO/runs/as2/probes/NEW_REFERENCE"
```

For each of five sequential fresh processes per phase, wrap the retained
`as2_llm PLAN INPUTS NEW_OUTPUT past forwards diagnostic` with `as2_run`.
Prefill arguments are `0 1 1`; decode arguments are `64 4 1`. Record each image
SHA before execution. Build completion must be awaited before snapshotting the
executable and its exact generated source/headers. An earlier incomplete dump
attempt used the preceding image; it is retained and excluded, then replayed.

**verified-run**, CANN9.1.0/CATLASS `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`,
physical1 Ascend950PR_9579, retained image
`01f4faa87f1d815e254faef90137b94285b3c517f43161b50cf6df836df644c3`:

- Prefill/decode repeatability: **5/5 fresh processes each**, all payload and
  control files bitwise equal within the same algorithm, no abort/guard violation.
- Full K/V and interleaved BF16 large-K outputs: bitwise equal to independent
  reference; decode context is also bitwise equal.
- Prefill context: **13/131072** BF16 differences, all position37/head6,
  maximum absolute error `0.0009765625`. Strict bitwise reference check **fails**;
  the unchanged predeclared Body tolerance passes. It is not relabelled bitwise.
- Same-input norm probe isolates one normalized query value, head6/element76:
  existing Body matches CPU FP32 normalization, differs from NPU fused RMSNorm.
  This is a numerical diagnosis, not a synchronization failure.

Raw evidence: `runs/as2/probes/mix_fp32_{protocol_v4,reference_v4}/`,
`mix_fp32_c3_v4b_policy.json`, `mix_fp32_qnorm_numeric_v1.json`, job receipts and
`tilemega-as2/build/retained/native_mix_fp32_v4b/identity.json`.
This probe does not pass a complete-model, long-generation or performance Gate.
