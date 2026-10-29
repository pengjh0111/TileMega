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

**verified-run**, the subsequent complete Llama8B path now keeps independent
Q/K/V GEMMs and exact guarded packing in CTG.64 KV partial/combine pairs and32
pack stages yield388 actual stages;65580/15693 prefill/decode tasks. Device
combine writes BF16 once, without residual. Full4-token intermediates,1024
greedy/logits/KV and teacher512 match independent SDPA/fused Norm bitwise.
Actual CANN logs record4 MIX calls for4 forwards. Evidence:
`correctness/llama8b_kv_split7_*_v4.json`, `probes/llama8b_actual_runtime_launches_v4.json`.
Subsequent sourced-price/EFT queues also pass1024/teacher512; no TPOT admission.

**verified-run**, actual Llama1B B16 decode M16/N2048/K8192 uses input shape
`[16,1,8192]`. Three device FP32 partials and CPU ordered sum match32768 BF16
values bitwise (`probes/llama1b_down_m16_partial3_numeric_v9.json`). This
remains a component result, without device-combine or full-model admission.

**verified-code/run**, M1024 prefill uses mixed DP/SK, not uniform seven-part
reduction. The installed CANN Blaze scheduler SHA is recorded in
`probes/llama1b_b16_streamk_domain_v1.json`. All83 full-K output differences
are in columns1792:2048; actual native activation prefix262144B is bitwise
equal. Uniform-seven replay fails13 values; selective-tail replay matches all
32768 sampled values. The resulting explicit local MIX Body passes the complete
B16 four-output model numerics; full1024/teacher512 and TPOT are separate gates.


## Qwen1.7 B16 prefill slice, 2026-10-26

`verified-run`: after fresh ACL/PCI inventory, physical1 is55:00.0 and maps
to logical0 (driver version.info25.7.rc1, CANN9.1.0, unchanged CATLASS SHA).
The actual M1024 prefill activation's first16 rows match the saved native
activation prefix byte-for-byte (196608B). The G2 `as2_gemm` probe writes seven
FP32 K intervals878/878/878/878/878/878/876 for the same BF16 operands.
Uniform seven-way reduction differs in7/32768 values; it is not admitted.
A CPU guarded replay takes ordinary native full-K values outside N1792:2048
and the measured seven-way sum inside that tail:0/32768 differences.
This is a bounded arithmetic diagnostic. The full M1024 scheduler domain,
device hybrid/combine, CTG binding and complete model are not yet verified.
No K6144 local Body or B16 decode decomposition is enabled by this probe.
Evidence: `probes/qwen17_observed_down_m1024_slice16_v1/case.json`,
`qwen17_down_m1024_slice16_partial7_numeric_v16.json` (uniform failure),
`qwen17_b16_prefill_tail_hypothesis_v16.json` (guarded replay).
Retained image: `tilemega-as2/build/retained/qwen17_partial_v16/`; one selected
serial compile, one short kernel launch, no performance or model pass claim.

## Llama8B B16 Q prefill tail, 2026-10-27

`verified-run`: separate-QKV prefill has an exact layer0 Norm prefix, but six
Q values differ within N3584:4096. The actual eager M1024/N4096/K4096 linear
selects tiling4162, base256×256×64 and singleCoreK1366. Replaying seven K slices
repairs only three of those six values and is rejected. Three native FP32
slices1366/1366/1364, ordered CPU sum and BF16 RINT repair all65536 sampled
values when combined with the regular full-K region. Uniform three-way
reduction still differs in eight values outside that selective-tail contract.
Evidence: `probes/llama8b_b16_q_tiling_v28/dispatch.json`,
`probes/llama8b_b16_q_tail_partial{7,3}_numeric_v{27,29}.json`.
These are same-input arithmetic diagnostics; CPU combination is excluded from
the model hot path. The opt-in MIX device-combine candidate requires separate
CTG/Plan/full-model numerical and request-performance admission.

The Q-only candidate repairs the observed prefill Q/context prefix but fails
the four-token model. Its first remaining prefix drift is O raw, then down
raw. Actual O rows16:32 have exact context operands and four BF16 differences;
the selective1366/1366/1364 replay repairs all65536 values. Down's full
M1024/N4096/K14336 eager call selects4162/singleCoreK4779. Its first eight
activation rows are byte-identical, and selective4779/4779/4778 replay repairs
all32768 values (uniform three-way still differs15). Evidence:
`probes/llama8b_b16_o_tail_rows16_partial3_numeric_v36.json`,
`probes/llama8b_b16_down_tiling_v34/dispatch.json`,
`probes/llama8b_b16_down_tail_partial3_numeric_v35.json`.
The residual candidate preserves BF16 sum rounding before residual Add; Q's
store variant has no fabricated residual. Both use per-group scratch, existing
ready/done/ack and one owner. No CPU combine enters the persistent kernel.


## L1 pipeline256, 2026-10-29

`verified-run`: bounded `as2_gemm` implementation7 changes L1 K128→256 while
retaining16×128 output geometry, L0K32, BF16 NK input/output and FP32 accumulate.
Actual Qwen1.7 gate/up operands M16/N12288/K2048 and a seed20261016
M1/N2064/K8208 tail fixture both match implementation1 bitwise and pass the
frozen independent CPU FP32 tolerance. G2 on physical1; one launch per probe,
one warmup and three repetitions per tile. Median tile service8.672→7.443us
and34.330→29.570us respectively. These are component observations, not TPOT.
The probe keeps MMAD alive per worker; current model constructs it per task.
No native candidate/price/default is changed on this evidence alone.
Source: `as2_gemm.cpp`; one selected BiSheng target, parallel1/nice10.
Evidence: `runs/as2/probes/opt_l1k256_numeric_and_timing_v1.json`,
`runs/as2/bench/opt_l1k256_probe_v1/{identity,state}.json`.
