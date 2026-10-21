# AS-2 pipeline evidence and remaining integration

Status at 2026-10-21: **in_progress**, not G-PIPELINE pass.
`verified-code` refers to this branch; numerical/device facts are `verified-run`.

| AS-1 gap | Implemented entry | Remaining work |
|---|---|---|
| A01 | `tools/as2_compile_llm.cpp`: real serving `inspect/prepare` | priced full compiler and execution entry |
| A02 | `CandidateGenerator::SetAscendDomain/Analyze/TopK`; `as2_rank_gemm` | model-family prices and structural search |
| A03 | priced worker binding accepted; native `AscendSchedule` | PrepareSymbolicProblem still rejects native price_tasks; old materializer still rejects priced/large plans |
| A04 | optional dual-AIV domain; G2/G4 payload evidence | full kernel dispatch and later G validation |
| A05 | body-only diagnostic probes | model diagnostic/fast separation remains |
| A06 | heterogeneous EFT and worker frontier lowering on CPU | device consumer of selected queues/waits |
| A07 | measured-counter finite wait; old DSB/cache protocol retained | no cache-removal optimization enabled |
| A08 | native 32-key AIV partial/merge, GQA integer mapping | fused KV/QK norm/RoPE, prefill and decoder integration |
| A09 | checked projected task counts, serving storage metadata, 64-bit host offsets | new device ABI/codegen; old 4096/16 limits remain |
| A10 | single/dual-AIV joint resource occupancy in CPU scheduler | owner/participant device lifecycle, direct BF16, split-K |

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
It rejects affinity that has not been explicitly lowered. It is not yet wired
into the executable model materializer. The CPU fixture service times are marked
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

Next required integration: materialize selected EFT queues and prefix waits into
a 64-bit-address LLM ABI, implement full serving-stage dispatch and KV lifecycle,
and supply nonzero empirical prices for every selected hot stage. The current
past64 dependency projection must not be reused blindly at arbitrary past;
use proved buckets or conservative capacity dependencies plus runtime masking.
Changing GEMM geometry must reinstantiate CTG ownership and LM-head partial
geometry before scheduling; the component TopK result is not a kernel-only
tile substitution into the current 32×64 CTG.
Full-model, single-launch and TPOT gates remain unpassed.
