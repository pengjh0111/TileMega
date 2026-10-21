# AS-2 pipeline evidence and remaining integration

Status at 2026-10-21: **in_progress**, not G-PIPELINE pass.
`verified-code` refers to this branch; numerical/device facts are `verified-run`.

| AS-1 gap | Implemented entry | Remaining work |
|---|---|---|
| A01 | `as2_compile_llm`: inspect/prepare/emit-diagnostic and native phases | priced complete generation entry |
| A02 | `CandidateGenerator::SetAscendDomain/Analyze/TopK`; `as2_rank_gemm` | model-family prices and structural search |
| A03 | new LLM materializer consumes actual queues/frontiers | full-model empirical provider; old AS-1 restrictions remain isolated |
| A04 | both AIV lanes compute in full LLM G2; G2/G4 payload probes | larger model G/domain validation |
| A05 | native trace-free branch; common outputs bitwise | complete fast KV C-2 and request timing |
| A06 | device FIFO/frontier consumer and five-process lifecycle | priced model queue selection |
| A07 | measured-counter finite wait; old DSB/cache protocol retained | no cache-removal optimization enabled |
| A08 | QK norm/RoPE/KV, blocked decode and causal prefill | numerical failures, faster shared KV preparation |
| A09 | 64-bit LLM ABI, >4096 executed tasks, capacity/epoch bounds | actual >4 GiB device allocation and complete request reset |
| A10 | GM AIC/AIV0 owner-ready-done lifecycle, five processes | dual-participant MIX, BF16 direct and split-K |

`as2_compile_llm prepare` calls existing BuildModelPlan, ImportSemantics,
InstantiateForGranularity, ModelDescription, PrepareSymbolicProblem and
BuildPlanSkeleton. It retains symbolic ISL relations instead of expanding TSVs.
The result explicitly says priced_solver_executed=false and device_executed=false.
There is no hand-written final tile queue in this importer.

ModelDescription previously populated physical_buffers only for DM. Serving
now retains the verified typed allocation table too. Empty/missing table coverage
fails preparation instead of producing a spurious zero-byte capacity result.
Actual Llama-3.2-3B metadata yields maximum offset 6,555,471,872 B. This proves
host allocation arithmetic, not device ABI support or an allocated 3B model.

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

Next: bind prefill/decode buffers by verified names into a shared weight/KV
arena; add complete request boundaries/reset in the native Host runner;
resolve Qwen numerical failures; then consume sourced fast-stage/dispatcher/
shared-GM prices through the existing native EFT scheduler. The new optimized
materializer rejects unknown prices or diagnostic-only schedules. The CLI
currently emits only unpriced bootstrap plans; G-PIPELINE remains in_progress.
Changing geometry must regenerate CTG ownership and LM-head partials.
