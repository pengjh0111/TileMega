# R10 partial closure for the R11 handoff

**Closed by explicit user instruction on 2026-09-26; R10 acceptance is incomplete.**
All unfinished R10 tests and queues were terminated, including both historical
B16 prefill searches, the old EV-1 queue and the four new-source search chains.
The recorded 14 processes exited without SIGKILL; no selected process survived.
No further GPU tests were started after the stop instruction. Read-only report
and source-verifier regeneration followed. There is no background R10 queue to
resume automatically.

This report supersedes earlier statements that tests are still running.
[The previous report](closure_r11/previous_summary.md) and
[implementation correction audit](implementation_completion/audit.md) are
historical records. Missing measurements remain missing; cancellation is not a
PASS and is not a claim that the underlying mechanism has failed.

## 1. Provenance and commits

- Baseline: `f6c270ce09988f600a667dc2fc63f3527909c759`.
- Corrected implementation measured here: `aac391c3a960f080e52bc96034d6f34cd62bca2a`.
- Historical selected plans: `b22f9c3427cc5233f6c27139d26650896cf03285`.
- External prompt (including errata): `/root/Prompt/TileMega_R10_prompt.md`.
- SHA256: `655fe479331d1cf771de6ecbe6b1bfdc9957847544b7b21752692d60ec3a0367`.
- Complete implementation history: [commits.txt](closure_r11/commits.txt).
  Evidence closure commit: `c117afb6a2ad1eb8b4f13f7bc059da32429bc8dc`.
  The final documentation commit is identified by
  `docs: close r10 with the unmeasured gates explicit`; its hash
  are recoverable from `git log origin/tilemega`.

The new in-flight and TaskBody measurements are committed with their raw inputs
and fitted target. They supersede the old target for subsequent work, but do
not retroactively reprice or validate historical selected plans. Original
fits/observations remain in `closure_r11/calibration_before_completion/` and
`closure_r11/target_before_completion.json`. Source/fit provenance is separate
from numerical pass/fail.

## 2. Gates and complete verifier output

NOT CHECKED/INCOMPLETE/PARTIAL explicitly mean the gate is **not passed**.
Report-only gates retain partial evidence rather than inventing a binary result.

| Gate | Status | Measured coverage / result | Evidence (relative to this directory) |
| --- | --- | --- | --- |
| G-1 | FAIL | 15/16 structural checks; K-12 lacks current selected-plan SASS | closure_r11/verify.txt |
| G-2 | PASS (focused backend evidence) | Current serving CTest 15/15; GEMM 1344/1344; prefill KV32 and KV64 each 8/8; decode checks retained | implementation_completion/audit.md |
| G-3 | NOT CHECKED | Formal EV-1 0/10; four seed HF checks do not replace it | closure_r11/seed_checks/ |
| G-4 | NOT CHECKED | Formal mode/timed-run matrix 0/10; four seed mode checks pass | closure_r11/seed_checks/ |
| G-5 | NOT CHECKED | Current final plans 0/20; historical b22 audits 18/20, all zero FP64 | closure_r11/historical_b22/plans/*/fp64_audit.json |
| G-6 | INCOMPLETE | Current code incremental/full 40/40, max relative error 0 (before fresh fit); historical Llama pruning ratio 1.0, Qwen unpruned timeout 5400 s | implementation_completion/incremental_final/report.json; closure_r11/historical_b22/pruning_equivalence/ |
| G-7 | FAIL (historical); NOT CHECKED (current) | 18/18 completed b22 plans exceed 600 s; range 1145.983–5932.311 s; current final plans 0/20 | closure_r11/historical_b22/plans/*/result.json |
| G-8 | PASS (existing CG audit) | 4/4 prescribed model/batch/past endpoints; no new final-plan audit | floor_audit/verify_output.tsv |
| G-9 | NOT CHECKED | 0/10 same-session final comparisons; geometric mean unavailable | closure_r11/status.json |
| G-10 | PARTIAL | 8 historical prefill/decode floor pairs; no final E2E/floor or measured three-point ratios | closure_r11/historical_b22/report_tables/request_floors.tsv |
| G-11 | PARTIAL | Historical model estimates and one diagnostic trace; no complete measured ten-cell decomposition | checkpoints/20260926T044401Z/gaps.tsv; trace_probe.md |
| G-12 | PASS (compile only) | sm_80, sm_90, sm_120 compile; no execution on those target architectures | implementation_completion/arch_compile.json |

Complete `verify.py` output (exit code **1**, expected incomplete G-1):

```text
K-1 PASS
  PASS include/tilemega/Backend/ServingGemm.h:40: cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
  PASS include/tilemega/Backend/ServingGemm.h:36: cute::Swizzle<3, 3, 3>{},
  PASS include/tilemega/Backend/ServingGemm.h:47: using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  PASS include/tilemega/Backend/ServingGemm.h: no /SM80_CP_ASYNC_CACHEALWAYS<std::uint32_t>|DefaultCopy/
K-2 PASS
  PASS include/tilemega/Solver/BackendCostQuery.h:132: constexpr bool ServingBF16ShapeLegal(int m, int n, int k, int stages) {
  PASS include/tilemega/Solver/BackendCostQuery.h:92: return m > 0 && n > 0 && k > 0 && stages > 1 && m % 16 == 0 && n % 16 == 0 &&
  PASS include/tilemega/Solver/BackendCostQuery.h: no /ServingBF16ShapeLegal[^\n]*m\s*%\s*32/
  PASS include/tilemega/Backend/ServingGemm.h:77: solver::ServingBF16SmemBytes(TileM, TileN, TileK, Stages);
  PASS include/tilemega/Backend/ServingGemm.h:78: static_assert(sizeof(typename Mainloop::SharedStorage) <= kSharedBytes,
K-3 PASS
  PASS git diff f6c270ce0: 88 source files, no arch comparison outside ArchDispatch
  PASS no serving SM/smem literal from the forbidden list
K-4 PASS
  PASS include/tilemega/Backend/ServingAttentionWarp.h:15: cute::SM80_16x8x16_F32BF16BF16F32_TN{},
  PASS include/tilemega/Backend/ServingAttentionWarp.h:28: using LoadA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  PASS include/tilemega/Backend/ServingAttentionWarp.h:30: cute::Copy_Atom<cute::SM75_U16x8_LDSM_T,Element>,
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h:120: asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(ka),"l"(p.key_cache+offset));
  PASS include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h:44: float weight = exp2f(lse[base] - maximum);
  PASS include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h:46: float4 first = *reinterpret_cast<float4 const*>(partial + base * HeadDim + dim);
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h:41: alignas(16) Element key[2][kKvTile*kHeadDim];
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h:42: alignas(16) Element value[2][kKvTile*kHeadDim];
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h:224: PV::PV(score,score_coords,s.storage.pipeline.value[slot]+warp*16*kHeadDim,output);
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h: no /Element probability\[|float score\[/
  PASS lib/Codegen/Codegen.cpp:266: int kv_tile = ServingAttentionKvTile(integerField(item, "width"), gemm_shared);
  PASS serving TaskBody headers: no trig or double
K-5 PASS
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh: no /ResetBuffersOnly/
  PASS python/tilemega/serving/engine.py: no /ResetBuffersOnly/
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh:35: std::uint64_t next_iteration[2] = {0, 0};
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh:251: if (iteration != plan->next_iteration[mode_index]) return -2;
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh: no /selected_mode/
  PASS python/tilemega/serving/plan.py:95: self.iteration = {1: 0, 2: 0}
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh: tm_plan_launch has no copy/sync
  PASS python/tilemega/serving/engine.py: decode loop has no sync/host read
K-6 PASS
  PASS include/tilemega/Codegen/tasks/ModelRuntime.h:146: int batch = 1;
  PASS include/tilemega/Codegen/tasks/ModelRuntime.h:209: std::uint32_t per_batch = 0;
  PASS lib/Frontend/Frontend.cpp:631: "batch", builder.getStringAttr(liftOptions.batch_symbol)));
  PASS python/tilemega/serving/export.py:157: inputs = [torch.randint(0, config["vocab_size"], (batch_example, seq), dtype=torch.int32)]
  PASS lib/Frontend/ModelPlan.cpp:906: std::uint32_t const id_bits = TokenIdBits(ids_node);
  PASS lib/Frontend/Frontend.cpp: no /batch\s*==\s*(?:1|16)/
K-7 PASS
  PASS lib/Frontend/ServingSemanticLifting.cpp:265: const auto Ec = C(stage.attention_kv_block);
  PASS docs/experiments/SERVING_R10/frontend_import/llama_decode_B16_seed.mlir: serving CG dump present
  PASS docs/experiments/SERVING_R10/frontend_import/llama_decode_B16_seed.mlir: 32 KV position maps use literal E_c and affine c/z terms
K-8 PASS
  PASS include/tilemega/Solver/ServingPruning.h:29: inline bool PruneServingR1(GemmConfig const& g,
  PASS include/tilemega/Solver/ServingPruning.h:60: inline bool PruneServingR2(GemmConfig const& g,
  PASS include/tilemega/Solver/ServingPruning.h:71: inline bool PruneServingR3(GemmConfig const& g,
  PASS include/tilemega/Solver/ServingPruning.h:92: inline ServingSearchOrderR4 MakeServingSearchOrderR4(int resident_limit) {
  PASS include/tilemega/Solver/OperatorClasses.h:103: if(PruneServingR1(candidate,pruning)) {++domain.removed_r1;continue;}
  PASS include/tilemega/Solver/OperatorClasses.h:104: if(enable_r2 && PruneServingR2(candidate,pruning)) {++domain.removed_r2;continue;}
  PASS lib/Solver/SkeletonSearch.cpp:434: auto serving_order=MakeServingSearchOrderR4(fixed.estimated_limit);
  PASS lib/Solver/SkeletonSearch.cpp:155: PruneServingAttentionSmemR1(attention->width,
  PASS lib/Solver/SkeletonSearch.cpp:425: out<<"ATTENTION_COORDINATE\t"<<start<<'\t'<<pass<<'\t'
  PASS lib/Solver/SkeletonSearch.cpp:83: if(options.incremental_prepare && hit!=serving_structures.end()) {
  PASS lib/Solver/SkeletonSearch.cpp:82: auto hit=serving_structures.find(next_key);
  PASS lib/Solver/SkeletonSearch.cpp:221: point.candidate.score=(EvaluateFlow(low.flow->flow).makespan_ns+
  PASS lib/Solver/SkeletonSearch.cpp:222: 4*point.candidate.score+EvaluateFlow(high.flow->flow).makespan_ns)/6;
K-9 PASS
  PASS lib/Solver/FlowPreparation.cpp:56: int RuntimeReleaseEndpoint(int cg_last,int consumer_task,int producer_count,
  PASS lib/Solver/FlowPreparation.cpp:61: auto bounds=codegen::RuntimeDependencyBounds(consumer_task,producer_count,
  PASS lib/Solver/FlowPreparation.cpp:63: last=std::max(last,bounds.last());
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh:89: auto lower = RuntimeDependencyBounds(
K-10 PASS
  PASS include/tilemega/Target/TargetSpec.h:191: std::vector<double> inflight_curve_bytes;
  PASS include/tilemega/Target/TargetSpec.h:193: std::vector<double> cta_stream_curve_bytes;
  PASS lib/Solver/StageFlowModel.cpp:61: std::optional<InflightDramServer> inflight;
  PASS lib/Solver/FluidExecutionSimulator.cpp:33: std::optional<InflightDramServer> inflight;
K-11 PASS
  PASS lib/Solver/SkeletonSearch.cpp:541: ? eb.evaluation.makespan_ns>0.98*ea.evaluation.makespan_ns
K-12 FAIL
  PASS docs/experiments/SERVING_R10/seed_split_price/command.txt: no /MIDPOINT_REFINE=1/
  FAIL 0/20 current-source SASS audits; 12 historical audits, source=ce69b16de34b629183bea452a3a891ebca7b11ebdb76350f9b197ab87b6677ef
K-13 PASS
  PASS include/tilemega/Codegen/tasks/ModelRuntime.h:332: std::uint32_t eft_past_lo = 0;
  PASS include/tilemega/Codegen/tasks/ModelRuntime.h:333: std::uint32_t eft_past_hi = 0;
  PASS include/tilemega/Codegen/tasks/ServingRuntime.cuh:89: auto lower = RuntimeDependencyBounds(
K-14 PASS
  PASS python/tilemega/serving/vllm_baseline.py:41: temperature=0.0, max_tokens=count, ignore_eos=True,
  PASS python/tilemega/serving/vllm_baseline.py:41: temperature=0.0, max_tokens=count, ignore_eos=True,
  PASS python/tilemega/serving/vllm_baseline.py:42: detokenize=False,
  PASS python/tilemega/serving/vllm_baseline.py:22: from vllm.inputs import TokensPrompt
  PASS python/tilemega/serving/vllm_baseline.py: no /enforce_eager=True/
  PASS python/tilemega/serving/vllm_baseline.py:24: ids = json.loads(args.prompt_ids.read_text())
K-15 PASS
  PASS lib/Frontend/ModelPlan.cpp:965: std::string qkv_recipe = "{\"kind\":\"qkv_group_interleave\",\"sources\":" +
  PASS lib/Frontend/ModelPlan.cpp:1040: std::string gu_recipe = "{\"kind\":\"gate_up_interleave\",\"sources\":" +
  PASS python/tilemega/serving/weights.py:47: if kind == "qkv_group_interleave":
  PASS python/tilemega/serving/weights.py:61: if kind == "gate_up_interleave":
  PASS python/tilemega/serving/weights.py: no /if\s+.*(?:llama|qwen)/
K-16 PASS
  PASS include/tilemega/Codegen/tasks/ServingEmbeddingTaskBody.h:20: int token_row, int seq, int past,
  PASS include/tilemega/Codegen/tasks/ServingArgmaxReduceTaskBody.h:25: int output_position, SharedStorage* shared) {
  PASS include/tilemega/Codegen/tasks/ModelHarness.cuh:292: int(stage.width), p.dims.capacity, p.dims.past + p.dims.seq,
  PASS docs/experiments/SERVING_R10/token_sets.log: TOKENS_DISJOINT batch=2 past=64 seq=64 read_bytes=8 write_bytes=8
G-1 FAIL: 15/16 structural checks
```

## 3. Stop ledger, degraded forms and deviations

| Item | Scope stopped | Reason / observed state | Unlock and estimated remaining effort |
| --- | --- | --- | --- |
| Historical plan matrix | Both models, prefill B16; downstream old EV-1 | User cancellation; 18/20 complete | Historical results are archived; prefer rebuilding current implementation, not accepting old kernels. |
| Current-source matrix | Four active B1 chains plus 16 not-yet-started plans; downstream EV-1 | User cancellation; 0/20 finalized | Explicit future resumption; full solve/compile/measure matrix. Prior runtime suggests hours, not a validated current estimate. |
| G-6 Qwen control | Unpruned equivalence result, hence complete pruning claim | 5400 s timeout (return code 124); no final winner | Profile preparation/search and complete the control; roughly 1–3 engineering days plus runs, inferred. |
| Final EV-1 and diagnostics | Ten-cell C-1/C-2, repeat determinism, same-session vLLM, throughput/floor, remaining trace and prediction diagnostics | Queue cancelled before any formal cell completed | Resume only when final plan set exists; GPU runtime is not measured for this matrix. |

[Process identities, commands and termination results](closure_r11/stopped_processes.json)
and partial search logs are preserved. This is a user-directed partial closure,
not one of the prompt's environment/global-stop conclusions.

Degraded forms: historical G-7 exceeds the allowed 600 s (continuation was
permitted by §7.3); trace v2 combines fixed/mainloop work where not distinguished.
Seed checks reuse existing generated **seed geometries** compiled against current
headers, not final solved geometries; HF free-greedy diagnostics were skipped.
Qwen B16 first ran out of memory before generation; that log is preserved and a
fresh-process retry passed. No failed attempt was relabelled successful.

Deviation from the prescribed final matrix: it was not completed, at the user's
explicit stop instruction. No gate threshold or sample set was relaxed. The
prior scalar-merge omission and other backend omissions were corrected before
this closure; see the source audit for exact implementation choices. No new
algorithmic deviation or R11 implementation is introduced by this closure.

## 4. End-to-end results (R10 §10.5)

Formal EV-1 remains **0/10**. Therefore TileMega TTFT, TPOT mean/p50/p90, E2E,
throughput, winner-mode full generation and the ten-cell throughput geometric
mean are unavailable. The earlier SB-4 baseline below is real, but **not a
same-session comparison with final TileMega**. Candidate single-forward times
must not be interpreted as request latency.

| Model | B | vLLM TTFT s | vLLM E2E s | vLLM TPOT s | vLLM token/s | Final TileMega / ratio |
| --- | --- | --- | --- | --- | --- | --- |
| llama | 1 | 0.024175202939659357 | 3.97274966200348 | 0.0038597990802187883 | 257.75598442404527 | not measured |
| llama | 2 | 0.024665972916409373 | 4.863472284050658 | 0.004730015944412755 | 421.0983183180134 | not measured |
| llama | 4 | 0.04198740795254707 | 4.382582632009871 | 0.004243006084122506 | 934.6087327785435 | not measured |
| llama | 8 | 0.04437419504392892 | 4.438678314094432 | 0.004295507447752202 | 1845.5944360706194 | not measured |
| llama | 16 | 0.05811895604711026 | 4.940374171012081 | 0.004772487991168105 | 3316.3479997393774 | not measured |
| qwen3 | 1 | 0.0351656679995358 | 4.525899613043293 | 0.004389769252242187 | 226.25336122102908 | not measured |
| qwen3 | 2 | 0.035793218994513154 | 5.093767000944354 | 0.004944255896334155 | 402.0600077742687 | not measured |
| qwen3 | 4 | 0.06014536297880113 | 5.248572611017153 | 0.0050717763910443315 | 780.4026548860513 | not measured |
| qwen3 | 8 | 0.063141203019768 | 5.551038968027569 | 0.005364513944289151 | 1475.7597716722269 | not measured |
| qwen3 | 16 | 0.06565501797012985 | 6.171229843981564 | 0.005968303837743338 | 2654.9003058083063 | not measured |

Sources: `baseline/vllm_summary.tsv` and per-run token/measurement records in
`baseline/vllm/`. Full-generation per-step curves and final mode selection
validation are not available for EV-1.

## 5. Absolute position (R10 §10.6)

These are **CG-derived historical plan floors**, not new measured request times.
Both B16 prefill plans were cancelled, leaving eight paired request floors.
Final `E2E / ΣT_floor` for both engines and measured p64/p575/p1086 `T/T_floor`
comparisons are not claimed. In particular the new seed checks are correctness
checks and provide no absolute-performance evidence.

| Model | B | Prefill floor s | 1023-step decode floor s | Request floor s |
| --- | --- | --- | --- | --- |
| llama | 1 | 0.0025201575333437215 | 2.5955892021788984 | 2.598109359712242 |
| llama | 2 | 0.0025222953191087662 | 2.6152610110923864 | 2.617783306411495 |
| llama | 4 | 0.002779744412529583 | 2.6546046289193637 | 2.657384373331893 |
| llama | 8 | 0.005559488825059166 | 2.733291864573319 | 2.738851353398378 |
| qwen3 | 1 | 0.0035132275829130332 | 3.6551958461526737 | 3.6587090737355865 |
| qwen3 | 2 | 0.0035207068083844403 | 3.724044739657721 | 3.7275654464661057 |
| qwen3 | 4 | 0.004022770331201172 | 3.8617425266678156 | 3.8657652969990166 |
| qwen3 | 8 | 0.008045540662402343 | 4.137138100688006 | 4.145183641350408 |

Raw floor/point details: `closure_r11/historical_b22/report_tables/request_floors.tsv`,
`floor_points.tsv` and CG checks in that directory. Existing weight/KV endpoint
checks are `floor_audit/verify_output.tsv`.

## 6. Correctness (R10 §10.7)

Current backend, existing seed geometries, 1024 generated tokens per request:

| Cell | Positions | HF gap ≤0.5 fraction | Max gap | L1/L2 mismatches | HF mean NLL |
| --- | --- | --- | --- | --- | --- |
| llama_B1 | 1024 | 1.0 | 0.0 | 0 | 0.08093190938234329 |
| llama_B16 | 16384 | 1.0 | 0.125 | 0 | 0.06294780969619751 |
| qwen3_B1 | 1024 | 1.0 | 0.0 | 0 | 0.09046763181686401 |
| qwen3_B16 | 16384 | 1.0 | 0.125 | 0 | 0.061585210263729095 |

All four integration checks pass their HF threshold and same-instance mode
comparison: 34,816 generated positions per mode in total. `seed_checks/*` retain
both token sequences, HF gap=0/p99/p99.9/max, position buckets, NLL, commands and
stdout/stderr. Free-HF-greedy first-divergence positions were **not measured**
(`--skip-free-greedy`); an empty list is not an assertion of no divergence.
Three timed-run determinism is also not established by these checks. They do
not close formal G-3/G-4. The four earlier vLLM HF self-checks remain in `baseline/`.

Focused current-source backend results are 15/15 serving CTests, 1344/1344
GEMM/PyTorch cases, prefill 8/8 for each KV width, and the recorded decode checks.
Legacy CTest 80/80 coverage is historical, not newly rerun at closure.

## 7. Solver configurations and time (R10 §10.8)

18 completed **b22** plans include solve, top-3 candidate timing, selected plan
and SASS audit. They do not establish full-request correctness or performance.
All 18 exceed 600 seconds. Current-source plans have partial search/materialization
files only; no final winner is claimed.

| Historical cell | Solve s | Selected mode | Grid | Residency | κ | Ec | Rq | Variants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_prefill_B1 | 1663.6373534370214 | L1 | 128 | 1 | 1 | 1088 | 16 | 5 |
| llama_prefill_B2 | 2990.9885147659807 | L1 | 128 | 1 | 1 | 1088 | 32 | 5 |
| llama_prefill_B4 | 3367.3451718050055 | L1 | 128 | 1 | 1 | 1088 | 64 | 5 |
| llama_prefill_B8 | 5932.310612546047 | L1 | 128 | 1 | 1 | 1088 | 128 | 4 |
| llama_decode_B1 | 1145.9828607119853 | L1 | 128 | 1 | 1 | 64 | 4 | 5 |
| llama_decode_B2 | 1445.3676811960759 | L2 | 128 | 1 | 1 | 128 | 4 | 5 |
| llama_decode_B4 | 1572.8573534750612 | L1 | 128 | 1 | 1 | 256 | 4 | 4 |
| llama_decode_B8 | 1419.7960633350303 | L1 | 128 | 1 | 1 | 1088 | 4 | 5 |
| llama_decode_B16 | 1417.5804216819815 | L1 | 128 | 1 | 1 | 1088 | 4 | 4 |
| qwen3_prefill_B1 | 1691.533057575929 | L1 | 128 | 1 | 1 | 1088 | 16 | 4 |
| qwen3_prefill_B2 | 2972.932516042958 | L1 | 128 | 1 | 1 | 1088 | 16 | 5 |
| qwen3_prefill_B4 | 3753.6718772719614 | L1 | 128 | 1 | 1 | 1088 | 32 | 3 |
| qwen3_prefill_B8 | 5383.106440728996 | L2 | 128 | 1 | 1 | 1088 | 64 | 4 |
| qwen3_decode_B1 | 1451.9026450649835 | L1 | 128 | 1 | 1 | 64 | 2 | 4 |
| qwen3_decode_B2 | 1971.1001926590689 | L2 | 128 | 1 | 1 | 128 | 2 | 5 |
| qwen3_decode_B4 | 2415.9502018699422 | L2 | 128 | 1 | 1 | 256 | 2 | 4 |
| qwen3_decode_B8 | 1879.4846540309954 | L1 | 128 | 1 | 1 | 1088 | 2 | 4 |
| qwen3_decode_B16 | 1973.134292148985 | L1 | 128 | 1 | 1 | 1088 | 2 | 5 |

Per-class tile/stages/split configurations and measurements are in
`closure_r11/historical_b22/report_tables/plans.tsv`; complete manifests,
search rows, top-3 modes, materialization metrics and phase logs are archived
under `closure_r11/historical_b22/plans/`. Domain counts and timings are in
`pruning_domains.tsv` and `solver_phases.tsv`. Existing top-3 timings are
single-forward candidate measurements, not EV-1. Current-source cache changes
have an independent 40/40 incremental/full equality check with zero relative
error (`implementation_completion/incremental_final/`), preceding the fresh fit.
A full new-calibration pruning control was not run.

## 8. Backend and resources (R10 §10.9)

[Coverage table](backend_coverage.md) and
[implementation audit](implementation_completion/audit.md) enumerate the paths.
The merge vectorization omission is fixed. New epilogue/attention implementation
checks, seed architecture compilation and seed SASS records are in
`implementation_completion/`. New seed build commands and ptxas logs are in
`closure_r11/seeds/`; these eight seed libraries are not selected final plans.
All 18 historical selected-plan audits show zero FP64, but none substitutes for
the 20 missing current-source final audits. G-12 is compile-only for
sm_80/sm_90/sm_120.

The prior GEMM microbenchmark measured serving about 27–32 GB/s per CTA and
819–846 GB/s at one CTA per SM, compared with legacy 18–21 and 771–819 GB/s
(`collective_bench/result.tsv`). It predates the implementation correction and
is explicitly **historical**, not a new backend throughput measurement.
Likewise `trace_probe.md` is an old Llama B1 instrumented L2 diagnostic, not
current final-plan bandwidth. Full final registers/smem/spill/residency and
attention/GEMM bandwidth coverage remain incomplete.

## 9. Gap decomposition and R11 inputs (R10 §10.10)

Historical model counterfactuals cover eight decode plans in
`checkpoints/20260926T044401Z/gaps.tsv`; they are model estimates. The single
historical trace in `trace_probe.md` must not be generalized to ten new plans.
No new complete measured decomposition is available. Consequently:

1. RMSNorm→GEMM prelude fusion: per-layer measured two-link saving is unknown;
   model-only chain costs are retained in the historical gaps table.
2. PG: historical model-only `T − max(T_floor,T_np0)` is retained in that table;
   no current ten-cell bound has been validated against measurement.
3. Protocol constants: historical model publication/wait/hop contributions are
   retained, with no new synchronization/race claim.
4. Device multi-step loop: measured step-gap ×1023 is unavailable because
   final event curves were not collected. No numerical saving is invented.

These are R11 inputs with explicit missing evidence, not completed research
conclusions. The unmeasured serving acceptance remains a prerequisite to
comparing future optimizations against a trustworthy R10 performance baseline.

## 10. Cost model (R10 §10.11)

Fresh current-source in-flight calibration (three repeats) and TaskBody
calibration (nine repeats) completed before cancellation, under the shared GPU
lock. There are **66 TaskBody observations over 12 kinds**, each at most 12
shapes. Raw curves, target fields, exact commands and fit errors are in
`closure_r11/` and the refreshed `calibration/`. New/old coefficient comparison:
`closure_r11/calibration_comparison.tsv`. Fit errors are recomputed from raw
observations (`calibration/analyze_task_bodies.py`), not inferred from a PASS flag.

The calibrated in-flight and per-CTA curves remain measured inputs, not proof
of final-plan prediction quality. `FlowPreparation.cpp` uses the shared runtime
window helper; final before/after release prediction changes and measured
p64/p575/p1086 errors were not collected. Current seed correctness checks do
not validate cost-model ranking.

## 11. Unmet gates: cause and next action (R10 §10.12)

- **K-12/G-1/G-5:** the source-fingerprint guard correctly excludes old binaries.
  `docs/experiments/SERVING_R10/verify.py` needs the complete current selected-plan
  audit set. Rebuild/select/audit all plans when authorized; do not weaken the guard.
- **G-6:** Qwen unpruned search timed out at 5400 s; no winning configuration
  exists to compare. `lib/Solver/SkeletonSearch.cpp` serving search and
  `Prepare`/`SetServingStructure` remain the profiling targets. The new
  `(Ec,Rq,argmax tile-N)` cache fix is implemented, but complete pruning
  equivalence with the fresh fit is not measured. Complete that control before
  asserting pruning retains the optimum.
- **G-7:** historical preparation dominates and top-3 full compilation adds
  hundreds of seconds (`historical_b22/.../solver_phases.tsv`). Inspect
  `SkeletonSearch.cpp:142` (`Prepare`), `:190` incremental preparation,
  `FlowPreparation.cpp` and `PiecePricing.cpp`; measure cache hit/miss time and
  compilation reuse. The existing fix does not retrospectively close the budget.
- **G-3/G-4/G-9/G-10/G-11:** missing final plan/matrix evidence after user stop,
  not an observed numerical or throughput failure. Resume the final protocol
  without substituting seed checks. A budget/quality conclusion is unsupported
  until that evidence exists.

## 12. Exclusions and handoff

No PG; no tile-direct TF or RMSNorm prelude fusion; no synchronization primitive
or barrier redesign; no TMA/WGMMA/tcgen05 implementation; no sampling, speculative
decode, continuous batching, block table, tensor parallel or quantization;
no MPK scheduler or device multi-step loop; no CUTLASS submodule change; no
MIDPOINT_REFINE enablement. Legacy solver/harness feature changes remain outside
scope. Closure changes only evidence, calibration data and documentation.

Raw evidence over 500 KB is losslessly gzip-compressed, with original SHA256
in `closure_r11/compressed_raw.json`; uncompressed local copies remain under
`/root/r10_work/r10_closure_raw/`. See `closure_r11/README.md` for reconstruction.
The stop ledger is authoritative; all earlier running/queued wording is historical.
