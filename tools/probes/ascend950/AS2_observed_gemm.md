# AS-2 observed large-K GEMM reduction

Scope: `verified-run` on physical950PR1 / logical0, CANN9.1.0, pinned CATLASS
`2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`. Component results do not admit a
complete model or performance cell. No baseline, seed, weights or tolerance changed.

## Same-operand evidence

The first old Qwen1.7 prefill divergence was layer1 Norm1. Layer0 activation
and canonical down weight reproduced52 BF16 output differences in an isolated
CATLASS body. L0K32, L0K64 and a reversible NK→KN physical transpose all give
the same52 differences. The isolated L0K64 M1/K8192 Llama case matches2048/2048
BF16 values; that alone does not prove the full Llama model.

An independent, one-op eager trace records `aclnnMatmul/MatMulV3`, BF16 inputs,
full FP32 math, 28 blocks and MIX1:2. Child-scoped DEBUG logs expose actual
tiling key4162 and the88-byte installed `MatMulV3BasicTilingData`:

| M,N,K | baseM,N,K | skSingleCoreK | Reduction |
|---|---|---:|---|
| 64,2048,6144 | 64,256,64 | 2048 | three FP32 K partials |
| 1,2048,6144 | 16,256,64 | 2048 | three FP32 K partials |
| 64,2048,8192 | 64,256,64 | 2731 | K2731/2731/2730 |
| 1,2048,8192 | 16,256,64 | 2731 | K2731/2731/2730 |
| 64,3072,8192 | 64,224,64 | 4096 | K4096/4096 |
| 1,3072,8192 | 16,224,64 | 4096 | K4096/4096 |

The installed arch35 template key and scheduler identify Stream-K. The
installed epilogue adds increasing K slices in FP32, then casts BF16 RINT.
Deterministic eager and default eager match for the observed M64 operands.
This is an observed algorithm choice, not a hardware peak or cost calibration.
Llama1B's same-input M64 down differs50 BF16 values from ordinary accumulation;
the three uneven CATLASS partials repair131072/131072 values. Its M1/tail case
also matches2048/2048. The subsequent explicit Llama binding rederives actual
CTG partial/combine queues and passes full64/1024 against SDPA/fused Norm.
Evidence: `probes/llama1b_down_{residual_debug_v1,partial3_tail_numeric_v6}` and
`llama1b_down_m1_partial3_tail_numeric_v6.json`; unaligned K2731 is not padded
or silently changed to a different reduction domain.

`as2_gemm` implementation4 writes three disjoint FP32 partial matrices in one
G2 launch. Each AIC computes multiple owned output tiles; this probe serializes
its K slices and does **not** claim cross-AIC split-K scheduling. The CPU-only
`as2_gemm_partial_compare` then verifies reduction order. M64 matches all131072
observed eager values; its difference from old native remains52. M1 matches
2048/2048 observed eager values and differs from old native in one value. CPU combine
is solely a diagnostic and is excluded from the native model path.

Raw evidence under `runs/as2/`:

- `probes/qwen17_observed_down_tile_v1/case.json`: exact operand/reference hashes.
- `probes/qwen17_observed_down_inner_k_v1.json`, `qwen17_down_layout_numeric_v2.json`.
- `probes/qwen17_down_eager_dispatch_v1`, `qwen17_down_tiling_debug_v2/tiling.json`.
- `probes/qwen17_down_partial3_numeric_v4.json`.
- `probes/qwen17_observed_down_decode_l1_v1`, `qwen17_down_decode_tiling_v1`,
  `qwen17_down_decode_partial3_numeric_v5.json`.

## Actual CTG decomposition

`as2_compile_llm emit-decomposed` consumes a checked-in, explicitly experimental
family binding. It applies `split_k=3` **before** semantic instantiation. Existing
CTG/ISL projection creates partial and combine task spaces; native scheduling
consumes their actual dependencies and owns placement/FIFO order. Materialization
checks every combine's three required partial predecessors, shape/counts,
role and the wait/FIFO graph. No final task queue is authored by hand.

- Partial: AIC-only CATLASS BF16→FP32, K2048 or K2731/2731/2730, original strides.
- Combine: independent AIV task, fixed FP32 sum order → BF16 RINT → BF16 residual.
- Logical partial `[row,col,part]` maps explicitly to physical `[part,row,col]`.
- Private named FP32 storage and BF16 raw-output storage have separate guards.
- Existing single-writer frontier publishes each transformed task once. No new
  hardware flag, cache shortcut, lock or Host operator fallback is introduced.
- Generated split capability and Host preflight reject old binaries/illegal
  shapes. Ordinary model defaults remain unchanged; split bodies are opt-in.

The Qwen1.7 M64 plan has228 stages/29029 tasks,28 partial/combine pairs,
3,760,446,464 arena bytes. It actually runs one persistent forward. Its198 output
checks,151936 logits,first greedy token and task receipts pass;140 saved
intermediates across28 layers are bitwise equal. Applying the observed decode
decomposition also repairs layer1 down raw. Full64/1024 greedy,teacher512 and
all-layer1087-position KV history match SDPA/fused Norm bitwise. Five fresh
reset processes pass two requests each (485 payload/control files/request).
The empirically priced EFT queues also pass1024/teacher512; actual CANN logs
show four MIX launches for four forwards. No final fused-eager/TPOT Gate is
admitted. Evidence: `correctness/qwen17_split3_*`,
`probes/qwen17_split3_reset_c3_v1.json`, `qwen17_actual_runtime_launches_v1.json`.

The Llama1B prefill/decode plans contain18476/4877 tasks,16 partial/combine
pairs and a2,636,313,088-byte shared arena. New image `3ef7b9e0c7ae8cf5…`
admits the observed uneven slices without changing the ordinary K loop.
Natural64/1024 tokens, full128256 logits and all16 layers/1087 KV positions
match the independent SDPA/fused Norm model bitwise. Last-forward saved
intermediates and teacher512 are also bitwise equal. Five fresh reset processes
pass281 files/request; fast/diagnostic217 files match. Empirical prices select
structural prefill and EFT decode; selected queues pass1024/teacher512. Actual
CANN logs show4 MIX launches for4 model forwards.
Evidence: `correctness/llama1b_split3_*`, `probes/llama1b_actual_runtime_launches_v1.json`.
The shape binding is experimental; final eager/TPOT admission remains open.

Llama3B's independently observed prefill activation is exact, but ordinary down
accumulation differs56 BF16 values; its next Norm differs8. Actual eager tiling
uses two K4096 partials. The updated component probe repairs196608/196608 M64
and3072/3072 M1 values. M1 has no same-input historical native capture and makes
no comparison to one. Actual CTG split2 creates35116 prefill tasks and a
6796274176B arena; device combine and four-token full-model results, final
logits/KV and saved intermediates match SDPA bitwise. Full1024/teacher512 and
28-layer1087-position KV also pass. Five fresh reset processes pass485 files per
request;373 fast/diagnostic common files match. Sourced positive prices select
structural prefill/EFT decode; both selected queues pass1024/teacher512 bitwise.
Evidence: `probes/llama3b_down_partial2_*numeric_v7.json`,
`correctness/llama3b_split2_request4_numeric_v1.json`.

**verified-run**, physical1: Llama8B's actual M1/N4096/K14336 eager projection
selects tiling66, not the other families'4162 Stream-K path. The existing
CATLASS full-K Body matches all4096 observed BF16 outputs bitwise. Evidence:
`probes/llama8b_down_m1_tiling_debug_v1/dispatch.json`,
`probes/llama8b_down_m1_fullk_numeric_v1.json`. No split family is enabled from
this one-row probe; prefill/full-model numerical checks remain separate.

The new full H4096/GQA4 fold candidate still forks at output2. Same-input
prefill and first-decode diagnostics both locate the first drift in `l0.qkv`,
while `l0.norm1` is exact. Unpacked M64 Q outputs are all exact, but K differs17
BF16 elements. Its actual eager tiling4162 prints K slice586; seven native
FP32 partials (last580) plus CPU ordered sum repair65536/65536 values.
Evidence: `probes/llama8b_k_m64_partial7_numeric_v8.json` and
`correctness/llama8b_{prefill_first_difference,fold_first_decode}_v3.json`.
This is not device-combine or CTG admission. Q and KV require distinct
decomposition contracts; a uniform split of fused QKV is not justified.

The first G2 partial attempt timed out120s while runtime logs stopped at device
subprocess-open response. No kernel launch was observed. Its failure receipt is
preserved. A bounded retry with explicit Host phase markers completes4.07s;
no reset or unrelated-process operation occurred. Shared-load times are not
performance constants.

Reproduce the already observed component using private `scripts/as2_env.sh`:

```bash
python -m tilemega.bench.as2_run --physical-device 1 --required-free-mib 1024 \
  --out "$REPO/runs/as2/jobs/NEW_PARTIAL" --timeout-s 120 -- \
  "$AS2_WORK/build/retained/gemm_partial_v5/as2_gemm" 64 2048 6144 4 \
  "$REPO/runs/as2/probes/qwen17_observed_down_tile_v1" \
  "$REPO/runs/as2/probes/NEW_PARTIAL"
python -m tilemega.bench.as2_gemm_partial_compare \
  --case "$REPO/runs/as2/probes/qwen17_observed_down_tile_v1" \
  --partial "$REPO/runs/as2/probes/NEW_PARTIAL" \
  --out "$REPO/runs/as2/probes/NEW_PARTIAL_NUMERIC.json"
```
