# MX-5 C550 pipeline and specification audit

Source audit updated 2026-10-26 through `2852cad5b`; measured checkpoint includes
GU+Consume controls and the accepted fixed8 Qwen comparison. Earlier audits remain in git history. **Verified** means inspected
source or executed evidence; **inferred** means interpretation. Authority is
`/root/Prompt/TileMega_MX5_prompt.md`, SHA256
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
The workspace actually uses `/root/tilemega` and `/root/Prompt`; this path
variation is recorded, not a change of implementation authority.

## Current pipeline (verified)

```mermaid
flowchart LR
  A[Checkpoint and export bridge] --> B[Frontend ModelPlan]
  B --> C[Static types and StageProgram]
  C --> D[cucc and actual resources/MIR]
  D --> E[plan.so and identity/manifest]
  E --> F[Recipe packing and shared request state]
  F --> G[ServingEngine L1 / device decode loop]
  G --> H[C-1 / C-2 and uninstrumented timing]
  H --> I[Qualified costs and exact top-4 selection]
  I --> J[Acceptance gates and defaults]
```

This diagram describes the intended native route. Current model controls use
explicit diagnostic pins; the final selection/default edge remains open.

| Step | Current implementation and limit |
|---|---|
| Export | `scripts/mx_serving_cell.sh` produces decode/prefill bridge JSON from random-weight checkpoints. Dense has five models; MoE-L24 export/checkpoint also exists. Export is not native execution. |
| Frontend | `compile.cpp` routes xcore1000 decoder/decode JSON to `mx_decode::TryCompile`. Dense uses `BuildModelPlan`, import and `AnalyzeMxDecode`; DM MoE analysis/native routing remains unfinished. |
| Solver inputs | `configs/targets/xcore1000.json` still points to the MX-4 decode cost table. An explicit v2 solve flag consumes qualified schema2 costs. Schema3 has an explicit diagnostic complete cost-or-rejection route; the C++ results route rejects it; an explicit Python ResolvedTop4 v2 consumer now checks genuine proofs, executes four candidates and publishes their measured winner. |
| Selection | `MxDecodeSolverV2` enumerates the prescribed geometry, averages interpolated attention cost across past64..1086, and merges exact top4. Preview IDs/ranks/evidence are retained. Whole-program resource failures and justified thread choice remain required before final measurement consumption. |
| Static code | `StageProgram` and `MxDecodeProgramV2` emit constexpr body types and direct calls, without runtime stage descriptors. Dense uses qkv(kStore), attention, O, gate_up, down and head; embed is folded into layer0. There are5L+1 counters/barriers per decode step. |
| Edges | DownLast prefetches head; head prefetches next-step layer0 QKV weights. Token broadcast happens after grid acquire; first token broadcast precedes entrance prefetch. Terminal packet lexical scope and stage-local AS4 binding reloads are recorded departures in the integration report. |
| Synchronization | Selected DT-W/R/P/X/KV =576 / role fence+atomic / P-B / X-A / block_fragments. `StageSyncV2` uses LDS generation release; `LastArriverV2` uses last-only acquire. Each barrier has its own64B counter, monotonic across steps. Only diagnosed per-plan512 controls currently execute outside the576 shape. |
| Bodies | GEMM/head/attention v2 are shared by generated programs and benches. `3843f6477` guards GEMM weight Prefetch for inactive/chunkless waves. `5c032c501` schedules statistics expansion after MMA consumption. Attention scalar sharing was rejected; ca678cbc8 uses fixed8 partial-O batches after a distinguishable2.16% E2E improvement;2dbf763ab shares packed gamma dependencies after a further2.44% gain. |
| Build identity | Native compilation retains actual compile receipts, source snapshots, CU, macro capture, resource log and binary identity. Normal resource gates require frame0/static-smem0 and MT≤227/255, plus actual one-CTA occupancy. Forced-register diagnostics never qualify for execution. |
| Weights | `serving/weights.py` executes exported recipes and `tile_pages(xcore_mma,TN,32)` packing, checks conflicting recipes and deduplicates by source/recipe identity. Actual packing coordinates must match the executed types. |
| Prefill and KV | Dense retains MX-4 prefill geometry, watchdog0 for timing and watchdog1 twins for checks. DT-KV requires full32-token fragment writes. Prefill/decode share request backing storage; `engine.py` now checks physical `kv_storage` as well as `kv_layout`, sequence and capacity. |
| Runtime | `DecodeRuntimeV2` supplies L1/launch_steps ABI. `execution.py` resolves mx_static to L1. Binding and launch controls enter the kernel by value; hot global accesses use explicit address spaces. Engine executes one prefill followed by separate or device-loop decode. |
| Validation | C-1 uses512 teacher-forcing steps against HF; C-2 requires exact separate/loop tokens. Timing uses64/1024, greedy BF16, fresh processes, warm1+timed2, step-events0 and TPOT=(E2E−TTFT)/1023. Diagnostic skips/step clocks are attribution evidence. |
| Defaults | Default dense remains MX-4 static decode. Explicit v2 resource failure throws rather than interpreting the plan. MoE/DNN still have legacy paths; their final native implementation and compile-time legacy rejection remain required work. |

## Evidence boundary (verified)

- G-A passes. G-B/G-PROGRAM/G-LLM remain open. Existing default target/pins
  have not been promoted to MX-5 final results.
- Five older native development controls pass512-step C-1/C-2. Old-source
  schema2 has4545 keys with three fresh measurements each. Neither fact
  substitutes for final source qualification and25-cell selection.
- Llama1B B1 has a real fitting minimum576 plan, so blanket512 is invalid.
  Its old-source complete domain has154 measured keys+35 resource rejections,
  with no missing key or invented rejection latency. Schema3 is diagnostic.
- Some576 plans spill at the toolchain's observed168 allocation budget despite
  the nominal AP-total227 bound. This discrepancy and candidate-bound D7
  evidence are recorded. It does not prove a body-level register lower bound.
- Eight admitted idle-Prefetch unit families each pass5 fresh processes
  (40/40), including full512 numerical matrices and512/576 chains. Rejected
  complete576 families were retained and never run.
- The real idle-fix QwenB1 WD0/WD1 plans are MT251/255,frame0,resident1;
  both have15 clean MIR rings. C-1/C-2 passes512 steps; generated tokens equal
  the old control exactly. Same-card P-PAIR medians old/fix are
  2.026787953/2.034753758ms (+0.393%, indistinguishable). The fix is still about
  2.58% slower than stored MX-4; performance acceptance remains open.
- Attention scalar reuse passes real512-step C-1/C-2 but P-PAIR increases
  0.384% (indistinguishable); it is not adopted. GU+Consume passes25/25 fresh
  complete512 unit-family jobs and QwenB1/8BB16 original1 C-1/C-2.
- GU+Consume fixes8BB16 original1/3 allPw=D to241/frame0 (WD0). The new1B576
  control is152/frame0;8BB1 originalGU8 fits245/frame0 without reduction.
  These are manual controls requiring individual root execution evidence.
- Qwen GU/Consume P-PAIR is2.034798772/2.053543765ms (+0.921%, indistinguishable),
  still above stored MX-4. Its fixed8 descendant is2.007555ms (+1.21% vs MX-4), so the gate stays open. G/H archives remain exact; new A costs are required.

## Whole-spec checklist

1. Keep order: stage0 → dense → MoE → LLM30 acceptance → DNN25. Development
   integration/calibration overlap is recorded; gates are not waived.
2. Dense covers five models×B1/2/4/8/16. MoE covers24-layer Qwen30B-A3B at the
   same batches. DNN covers ResNet18, MBV1/2, BERT S128 and NAFNet-w32 256².
3. Implement §3–§8 structures/layouts/rounding as specified. Only measured DT
   branches, bounded register reductions and DT-F provide the prescribed
   alternatives. Failed artifacts and source differences remain explicit.
4. Rebuild P0/derived widths and three-process stage costs for the actual body
   source. Every coordinate/row must execute in code. Select each final cell
   through exact top4 E2E measurements and the1% rule; report pred/meas.
5. Implement MoE native analysis, router/dispatch, grouped GEMMs/combine,
   prompt64 program, packing and routing check. Then accept25 dense+5 MoE
   cells against the stored baselines before starting native DNN execution.
6. Implement DNN native forward ABI/bodies/generator/solver/time command;
   pass all25 synthetic-weight checks and collect all25 forward timings.
   Only the two DT-F fusion experiments may be cut for budget.
7. GPU work stays guarded on7/6/5/4, at most two cards and one job per card;
   job≤1800s, timing≤900s, MoE needs40960MiB. No external baseline remeasure,
   clock/power/driver change, or use of cards0–3.
8. Budget60 card-hours; count all three job logs by canonical distinct output.
   At2026-10-26T12:44:22+08:00:685 closed/2 active, wrapper-wall upper bound
   20.798983 card-hours. Raw artifacts stay under ignored runs; committed
   measurements append round/executor/implementation without rewriting rows.

Current raw paths, failures, source identities, detailed stage ledger and the
remaining acceptance work are in `MX5_integration_c550.md` and the CSVs.

Verified continuation: all76 GU+Consume P0 libraries compile;8290 G/H zero-launch
queries complete.120 G/H primary libraries are authorized from their actual
references. Old A controls stay separate; all16 A8 P0 builds and800 zero-launch queries are complete.
ResolvedTop431 CPU tests pass; unknown previous576 reductions block512
consumption. Canonical576/D7/Resolvedv2 CPU protocols are implemented;90 related root tests
pass. Explicit runner/results/script consumption is implemented by2852cad5b;25
root integration tests pass. It still needs genuine current tables and all four
candidate proofs; no real final selection has run. G/H primary queries and
conditional retry queries are complete;576 final selection/campaigns are being
closed without changing failed rows. Gamma scheduling passes5fresh full A/legacy bitwise jobs,C1/C2 and P-PAIR:
2.007205873→1.958246937ms,−2.439%,distinguishable; adopted by2dbf763ab.
QwenB1 is now−1.276% versus MX-4; all5 development controls now pass C1/C2 and timing.
A16 P0 and8primary actual800 queries each give512400fit/576100fit+300reject;
Canonical per-candidate histories and final solver selection remain pending. G/H sources remain exact.
