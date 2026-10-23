# AS-2 pipeline evidence and remaining integration

Status rechecked at 2026-10-23: **in_progress**, not G-PIPELINE pass.
`verified-code` refers to this branch; numerical/device facts are `verified-run`.

| AS-1 gap | Implemented entry | Remaining work |
|---|---|---|
| A01 | `as2_compile_llm`: inspect/prepare/emit-diagnostic/emit-priced and native requests | priced complete generation entry |
| A02 | `CandidateGenerator::SetAscendDomain/Analyze/TopK`; `as2_rank_gemm` | model-family prices and structural search |
| A03 | new LLM materializer consumes actual queues/frontiers | native empirical provider connected; long-past/model-family calibration pending |
| A04 | both AIV lanes compute; G2/G4/G8 payload probes | larger model G/domain validation |
| A05 | short-request fast/diagnostic outputs including KV bitwise | main-cell C-2 and admitted timing |
| A06 | device FIFO/frontier consumer and five-process lifecycle | priced queues execute with399 equal payload files; broader search pending |
| A07 | measured-counter finite wait; old DSB/cache protocol retained | no cache-removal optimization enabled |
| A08 | QK norm/RoPE/KV, blocked decode and causal prefill | numerical failures, faster shared KV preparation |
| A09 | 64-bit LLM ABI, >4096 tasks, >4GiB real device arenas,1024 launches | teacher512, all dense/B16 long requests and wider bucket coverage |
| A10 | GM AIC/AIV0 owner-ready-done lifecycle, five processes | dual-participant MIX, BF16 direct and split-K |

`as2_compile_llm prepare` calls existing BuildModelPlan, ImportSemantics,
InstantiateForGranularity, ModelDescription, PrepareSymbolicProblem and
BuildPlanSkeleton. It retains symbolic ISL relations instead of expanding TSVs.
The result explicitly says priced_solver_executed=false and device_executed=false.
There is no hand-written final tile queue in this importer.

ModelDescription previously populated physical_buffers only for DM. Serving
now retains the verified typed allocation table too. Empty/missing table coverage
fails preparation instead of producing a spurious zero-byte capacity result.
Initial Llama-3.2-3B metadata yields maximum offset 6,555,471,872 B. The later
full28-layer short request actually allocates6,735,441,920B on device; Llama8B
also executes with16,523,817,472B. These inherited run receipts prove large
device addressing, while their numerical failures remain independently recorded.

`ScheduleAscendTasks` takes task service/GM bytes, typed legal workers, semantic
predecessors and explicit shared GM/synchronization prices. It uses critical
rank + EFT, a single device GM calendar and separate role timelines. Unknown,
zero or nonfinite service prices fail. MIX occupies explicit same-group lanes.
Actual FIFO orders and owners are outputs; a CPU audit checks data+FIFO cycles.
Per-producer maximum predecessor ordinal reduces waits; own-worker waits are
covered by FIFO. `AscendProgressValue` checks epoch/ordinal overflow. These CPU
tests do not establish a MIX ready/done/buffer-reuse device protocol.

`AscendTasksFromSymbolic` derives predecessor sets from real ISL oracles.
It rejects affinity that has not been explicitly lowered. It now feeds the native LLM materializer through an explicit diagnostic
bootstrap scheduler; no synthetic service times enter that bootstrap. The CPU fixture service times are marked
CPU_TEST_ONLY_NOT_A_CALIBRATION and are never written into hardware targets.

`as2_rank_gemm` compares two actually measured implementations using the native
provider, with exact-shape lookup and numerical/launch receipts. Shared-load
profiles are rejected by --final. It selects the second implementation in the
Qwen down-projection test, so TopK is not a fixed-first selector. Its scope is
hot tile service, not whole-model cost or a full optimized model Plan.

Evidence index under ignored `runs/as2/`:

- `models/qwen3_0_6b/prepare_decode_b1_storage/`: 10049 tasks, 485 buffers.
- `models/llama3_2_1b/prepare_decode_b1/`: 9207 tasks.
- `models/llama3_2_3b/prepare_decode_b1/`: 16399 tasks, GQA3 and >4 GiB offsets.
- `search/schedule_provider_contracts.json`: price-to-TopK/queue causality,
  dual AIV, prefix coverage, cycle rejection, 5000 tasks and overflow checks.
- `search/gemm_qwen_down_rank_provider/search.json`: measured component choice.

Native entry: `lib/Codegen/Ascend/AscendLlmCodegen.cpp`,
`include/tilemega/Target/Ascend/LlmPlanABI.h` and `LlmExecutor.h`,
`lib/Codegen/Ascend/PersistentLlm.cpp.in`. Tables are derived from the verified
module, symbolic oracles and audited per-worker schedule; materialization does
not replace FIFO order with stage order. Epoch stride and every queue/wait/
byte range are checked. Full task counts exceed4096 on the real device.

`emit-diagnostic` defaults to unified KV1088; prefill query tiles contain16
whole tokens, including GQA3 (48 query heads), with token-tile-major/group-minor
coordinates. Max-past1087 dependencies conservatively cover runtime decode
masking; prefill uses its real past0 graph. No per-token solving/recompiling.
The argmax semantic read extent now uses the actual64-column partial count,
rather than a hardcoded32-column formula. CPU regression checks vocabulary65.

Verified-run: `as2_request.py` relocates both solved phase tables by buffer
identity into one weight/KV/token arena; queue/wait bytes are unchanged.
`LlmRequestHost.h` submits prefill then stream-ordered decode, with device token
feedback, separate phase epochs and request resets. Five fresh processes/two
four-token requests each match401 payload/KV/control files, including fast mode.
Neither short requests nor local same-input arithmetic establish the main Gate.
`emit-priced` consumes sourced Body + dispatcher + notification + shared-GM
prices and compares rank/EFT with structural placement using the same fixed-FIFO
cost evaluator. Both plans retain real ISL predecessors; the lower predicted
makespan wins, rather than a fixed first candidate. Qwen prefill predicted
45.907 vs58.475 ms; decode5.06560 vs5.06588 ms. These are conservative model
predictions, not measured speedups. Both selected FIFO/owner tables differ.

Verified-run: selected prefill/decode execute four natural outputs;399 payload/
KV files match the old queues bitwise. `correctness/native_priced_structural_payload_c2.json`
and `models/qwen3_0_6b/native_{decode,prefill}_priced_g8_v1/search.json` are the
traceable example. CPU price changes alter actual queue/candidate selection;
fixed-queue repricing preserves order and responds to cost changes.

`as2_native_prices.py` verifies measured job/model/config/shape/G identities.
Prices are contextual empirical data, not global calibrated target constants;
shared-load status remains contended. Older ABI3 decode pricing is bounded to
past<=66 and rejects OSL5 before ACL. New ABI4 prices profile actual past1086;
`pricing_bounds.bin` preserves that bound after relocation. Five fresh control
processes pass8192 payload samples each, on the same physical1 card as the new
Body profiles. They remain context-local empirical prices, not peak constants.

Remaining: other dense-model/B16 numerical failures, wider measured
Body/context domains and end-to-end optimization/search admission.
G-PIPELINE remains in_progress: one measured Body geometry per whole-model
search, two real placement/order alternatives; no claimed final tuning result.

ABI4 adds experimental `cube-fp32` attention with an actual MIX role, checked
per-group workspace and three-stage owner/participant protocol. Source and
tables still come from real serving import/CTG/ISL/native scheduling; the small
protocol test's synthetic queues are never counted as pipeline evidence.
Earlier ABI4 Qwen candidates execute actual generated plans but fail numerical
checks; the new fold-softmax candidate passes long requests. ABI3 prices cannot be
replayed into ABI4; profile/pricing/binder identities include `runtime_abi`.
Internal attention-phase coordination is included once in owner service time.
11 CPU binder contracts pass. Broader hierarchy/search and ten-cell
correctness/performance remain incomplete. MoE stays deferred.

## Scope and implementation reconciliation, 2026-10-23

Verified-code at `ec83e3a85`: original Prompt SHA256 remains
`673b6deb619cb02f1c3f2d0875af2bdafb835bca09e3875d7dbfb840b1be00d6`.
The user's later MoE deferral takes precedence: five dense models/ten active
main cells; two MoE cells remain explicitly deferred, not silently passed.
Earlier device results are inherited run evidence, not new runs on this date.

| Prompt requirement | Current implementation and remaining obligation |
|---|---|
| Real optimized pipeline | Import/CTG/ISL/native scheduling/codegen execute; `emit-priced` has real causal candidate selection. Latest ABI4 has actual full-context empirical prices and structural/EFT selection; wider geometry/family search and end-to-end admission remain required. |
| Full model and state | Five canonical weight sets, full layers/embedding/head and persistent prefill/decode exist. Qwen0.6 B1 now passes actual1024 greedy and teacher512; other models/B16 and optimized-pipeline admission remain incomplete. |
| Numerical/performance contracts | Policy remains frozen. Intermediate bitwise checks locate drift; final admission uses declared error metrics plus exact greedy. Hooked/diagnostic runs cannot admit performance. |
| Resource/handoff optimization | G8 and both AIV lanes have evidence; same-task MIX uses AIV0. An opt-in actual CTG split3/FP32 combine candidate passes Qwen1.7 prefill and short generation. C3/performance, G14/28, BF16 direct and broader small-M alternatives remain pending. |
| TPOT validation | Strongest legal eager configuration still needs final baseline selection; full paired processes, warmups/repeats and95% intervals are pending for all ten cells. |
| Shared-server isolation | Private roots, one own NPU job, one card/process, no locks/reset/other-process control, sequential nice10/parallel1 builds and manifest cleanup remain mandatory. |

The observed-operand replay isolates the past1025/layer11 difference to softmax
reduction: Q/K/V, scores, exponentials match; the installed register-fold sum
repairs probabilities and PV bitwise at two real masks. Explicit candidate
`cube-fp32-fold-softmax` then passes Qwen0.6 B1 natural64/1024 and teacher512.
The same Plan fast/diagnostic pair matches401 payload/KV/control files (C-2,
not new C-3). Teacher runs have separate receipts and cannot enter greedy or
performance admission. Cached `reemit` regenerates source from an already
materialized plan without rewriting its queues or claiming a new solver call.
Current ABI4 search selects structural prefill (49.8288 vs49.9315 predicted ms)
and EFT decode (7.41285 vs7.41310 predicted ms). Both real queues differ. The
selected request passes actual1024 greedy and teacher512 against SDPA/fused
RMSNorm, with source-identical compiler-cache reuse and newly solved tables.
Results: `models/qwen3_0_6b/native_{prefill,decode}_fold_priced_v1/search.json`,
`correctness/qwen06_fold_priced_{request1024,teacher512}_numeric_v1.json`.
Prediction differences are not measured speedups. New FIA-v2 eager inference
runs without Host fallback but changes greedy at output6 and fails the current
KV/numerical pairing; this stronger-baseline issue stays open. Next: remaining
dense/B16 families, broader legal Body search and prescribed paired TPOT.
Default implementations remain unchanged.

H2048 full-row binding and head64/GQA4 are enabled only for explicit experimental
candidates after observed-operand probes. Llama1B's four-token check still fails.
Qwen1.7's old prefill drift is explained by observed eager Stream-K K2048×3,
not ordinary K accumulation. `emit-decomposed` changes granularity before
CTG/ISL construction:28 logical down GEMMs become5376 AIC partial tasks and1792
AIV combines.228 stages/29029 tasks, all three partial predecessors checked;
actual eight-AIC/independent-AIV receipts and one-launch full prefill pass198/198
outputs bitwise. Decode uses the same observed decomposition; four natural
tokens, full logits,56 KV slices and338 captured values/files are bitwise equal.
Natural64/1024 and teacher512 pass against SDPA/fused Norm. All28 layers/1087
KV positions are bitwise exact. Five fresh serial reset processes pass485
payload/control files per request (two requests/process); fast/diagnostic373
common files match. Actual CANN logs record four MIX launches for four forwards.
Empirical partial/combine prices reach `emit-priced-decomposed`: prefill EFT
84.509812ms vs structural85.280521ms; decode EFT12.806523ms vs12.813727ms,
**predicted**, not measured TPOT. Both changed queues are selected; actual
1024 greedy and teacher512 pass. Wider structure/family search and final fused
eager/performance admission remain open. Source-identical binaries and checked
retained headers are reused; final queues are newly solved, never patched.
