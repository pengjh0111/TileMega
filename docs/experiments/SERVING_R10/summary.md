# R10 implementation and experiment checkpoint

**Partial checkpoint at 2026-09-26T04:44:01.317825+00:00; not final R10 acceptance.**
The user requested that completed work be recorded and pushed, then that the
agent stop processing while the existing experiment processes continue.
The final ten-cell EV-1 matrix has not started at this checkpoint. No process
was stopped, restarted, or added for this checkpoint.

## 1. Provenance and implementation status

Baseline: `f6c270ce09988f600a667dc2fc63f3527909c759`. Source HEAD before this evidence commit:
`0a1abb3620d73b242a6ba7aca3677e5921641dc3`. Prompt: `/root/Prompt/TileMega_R10_prompt.md`; SHA256:
`655fe479331d1cf771de6ecbe6b1bfdc9957847544b7b21752692d60ec3a0367` (including the user's R10 errata).
The pre-checkpoint commit list is in [checkpoints/20260926T044401Z/commits.txt](checkpoints/20260926T044401Z/commits.txt);
the checkpoint commit is identifiable by its message
`docs: checkpoint the r10 implementation and running tests`.

The serving frontend, batch/past semantics, real-weight packing, external-buffer
C ABI, per-mode iteration counters, Python generation engine, GEMM collective,
fused attention, in-flight pricing, pruning/search, measurement and reporting
paths are implemented. This does **not** establish complete conformance:
`AttentionMergeTaskBody.h` still reads scalar FP32 partials instead of the
specified 16 B vectors, and the ten-cell acceptance is pending. Legacy tests
remain separate from serving acceptance.

## 2. Gates at this checkpoint

PENDING means the required formal sample set is incomplete. The executable
checker reports G-1 FAIL while K-12 lacks eight final plan SASS audits; no
existing audit contains FP64. G-7 is an observed budget failure.

| gate | status | value | evidence |
| --- | --- | --- | --- |
| G-1 | FAIL | 15/16 code checks | report_tables/verify_output.txt |
| G-2 | PASS | GEMM 1344; attention 60+8; CTest 80/80 | task_body_tests/; ctest_partial_interrupted.txt; ctest_cpu_*.txt |
| G-3 | PENDING | 0/10 | ev1/*/B*/tilemega_hf/check.json |
| G-4 | PENDING | mode 0/10; timed 0/10 | ev1/*/B*/mode_check/mode_check.json; ev1/*/B*/result.json |
| G-5 | PENDING | 12/20 audits; max FP64=0 | plans/*/fp64_audit.json |
| G-6 | PENDING | incremental=True; pruning=False | incremental_equivalence/report.json; pruning_equivalence/report.json |
| G-7 | FAIL | 12/20 plans; 12 over 600 s; max 2991.0 s | plans/*/result.json; report_tables/plans.tsv |
| G-8 | PASS | 4/4 endpoints | floor_audit/verify_output.tsv |
| G-9 | PENDING | 0/10; geomean=? | report_tables/requests.tsv |
| G-10 | PENDING | request floors 4/10; points 0/30 | report_tables/request_floors.tsv; report_tables/step_ratios.tsv |
| G-11 | PENDING | counterfactual gaps 8/10 | report_tables/gaps.tsv |
| G-12 | PASS | sm_80=0, sm_90=0, sm_120=0 | arch_compile/results.tsv |


Gate evidence paths in the table are relative to this directory. The immutable
checkpoint copies of report tables and full output are under [checkpoints/20260926T044401Z/](checkpoints/20260926T044401Z/).

## 3. Complete verify.py output

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
  PASS git diff f6c270ce0: 86 source files, no arch comparison outside ArchDispatch
  PASS no serving SM/smem literal from the forbidden list
K-4 PASS
  PASS include/tilemega/Backend/ServingAttentionMma.h:33: cute::SM80_16x8x16_F32BF16BF16F32_TN{},
  PASS include/tilemega/Backend/ServingAttentionMma.h:36: using LoadA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  PASS include/tilemega/Backend/ServingAttentionMma.h:38: cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>,
  PASS include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h:53: // The cache path uses one 16-byte cp.async.cg per lane and never issues an
  PASS include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h:41: float weight = exp2f(lse[base] - maximum);
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
  PASS include/tilemega/Solver/ServingPruning.h:59: inline bool PruneServingR2(GemmConfig const& g,
  PASS include/tilemega/Solver/ServingPruning.h:70: inline bool PruneServingR3(GemmConfig const& g,
  PASS include/tilemega/Solver/ServingPruning.h:91: inline ServingSearchOrderR4 MakeServingSearchOrderR4(int resident_limit) {
  PASS include/tilemega/Solver/OperatorClasses.h:103: if(PruneServingR1(candidate,pruning)) {++domain.removed_r1;continue;}
  PASS include/tilemega/Solver/OperatorClasses.h:104: if(enable_r2 && PruneServingR2(candidate,pruning)) {++domain.removed_r2;continue;}
  PASS lib/Solver/SkeletonSearch.cpp:393: auto serving_order=MakeServingSearchOrderR4(fixed.estimated_limit);
  PASS lib/Solver/SkeletonSearch.cpp:112: PruneServingAttentionSmemR1(attention->width,
  PASS lib/Solver/SkeletonSearch.cpp:384: out<<"ATTENTION_COORDINATE\t"<<start<<'\t'<<pass<<'\t'
  PASS lib/Solver/SkeletonSearch.cpp:147: SolverPhase phase(timing,"incremental_prepare");
  PASS lib/Solver/SkeletonSearch.cpp:178: point.candidate.score=(EvaluateFlow(low.flow->flow).makespan_ns+
  PASS lib/Solver/SkeletonSearch.cpp:179: 4*point.candidate.score+EvaluateFlow(high.flow->flow).makespan_ns)/6;
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
  PASS lib/Solver/SkeletonSearch.cpp:500: ? eb.evaluation.makespan_ns>0.98*ea.evaluation.makespan_ns
K-12 FAIL
  PASS docs/experiments/SERVING_R10/seed_split_price/command.txt: no /MIDPOINT_REFINE=1/
  FAIL 12/20 solved-plan SASS audits, counts=[(PosixPath('/root/TileMega/docs/experiments/SERVING_R10/plans/qwen3_decode_B1/fp64_audit.json'), 0), (PosixPath('/root/TileMega/docs/experiments/SERVING_R10/plans/llama_decode_B8/fp64_audit.json'), 0)]
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
  PASS include/tilemega/Codegen/tasks/ModelHarness.cuh:289: int(stage.width), p.dims.capacity, p.dims.past + p.dims.seq,
  PASS docs/experiments/SERVING_R10/token_sets.log: TOKENS_DISJOINT batch=2 past=64 seq=64 read_bytes=8 write_bytes=8
G-1 FAIL: 15/16 structural checks
```

## 4. Stops, degraded delivery, and deviations

- **Agent pause:** requested by the user. Background experiment chains remain
  active; process identities and exact commands are in
  [checkpoints/20260926T044401Z/snapshot.json](checkpoints/20260926T044401Z/snapshot.json). The EV-1 queue waits for all 20
  plans, relinks the plan tables, then runs full generation, HF/mode checks,
  reports and the optional four-cell trace. It does not finalize repository
  documentation or commit future results; that remains for the next resume.
- **G-7 degradation:** all 12 completed plans exceed 600 s.
  R10 §7.3 explicitly permits continuing while reporting G-7 FAIL.
  No budget or search domain was relaxed for this checkpoint.
- **Implementation gap / declared deviation:** §4.3(c) requests 16 B vector
  reads for LSE merge. The actual implementation parallelizes output dimensions
  but reads FP32 partials one scalar at a time and serially reduces live KV
  blocks (`include/tilemega/Codegen/tasks/AttentionMergeTaskBody.h:27`). A
  validated vectorized merge has not been implemented. The running plan matrix
  measures this implementation. BE-11's full vectorization claim remains open;
  implementing it later requires repeating affected TaskBody and plan/EV-1
  validation. This is not a prompt-imposed exclusion.
- The trace combines TaskBody fixed and mainloop time because trace v2 does
  not distinguish these boundaries. It is a separate instrumented L2 diagnostic,
  not a substitute for selected-mode uninstrumented timing.
- No global-stop condition has been established. Qwen3's unpruned G-6 control
  is still running; its possible timeout is not pre-labelled a pass or failure.

## 5. End-to-end performance and absolute position

Formal EV-1: **0/10 complete**, so no R10 throughput geometric mean, C-1/C-2
matrix result, or TileMega/vLLM end-to-end conclusion is claimed.
SB-4's earlier vLLM ten-cell baseline and four HF self-check cells are recorded
in `baseline/`; EV-1 will remeasure the baseline in the new session.

CG-derived request floors currently cover four complete prefill/decode pairs:

| model | batch | prefill_floor_s | decode_floor_s | request_floor_s |
| --- | --- | --- | --- | --- |
| llama | 1 | 0.0025201575333437215 | 2.5955892021788984 | 2.598109359712242 |
| llama | 2 | 0.0025222953191087662 | 2.6152610110923864 | 2.617783306411495 |
| qwen3 | 1 | 0.0035132275829130332 | 3.6551958461526737 | 3.6587090737355865 |
| qwen3 | 2 | 0.0035207068083844403 | 3.724044739657721 | 3.7275654464661057 |


Final `E2E/ΣT_floor`, p64/p575/p1086 measured ratios, TTFT, TPOT mean/p50/p90,
throughput and per-step curves remain pending EV-1. No pilot result replaces them.

## 6. Correctness evidence

G-2: GEMM **1,344/1,344** PyTorch cases; decode attention **60/60**;
prefill attention **8/8**; all **80/80** registered CTest cases in disjoint
logged groups. See `task_body_tests/` and `ctest_partial_reason.md`.
G-8: weight/KV floor **4/4** endpoint checks. G-12: compilation succeeds for
**sm_80, sm_90, sm_120** using the local nvcc; this is not execution on those GPUs.
Incremental/full preparation agrees for **40/40** configurations with zero
relative error (`incremental_equivalence/report.json`). Llama pruning and
unpruned flow scores both equal **3,889,647.611 ns**, with the unpruned winner
admitted by the pruned domain; Qwen3's matching control is pending.

Formal C-1 distributions, HF free-greedy divergence/NLL, C-2 mode equality and
three-run token determinism are pending. Earlier B1 pilots and short sequence
checks are diagnostic evidence only.

## 7. Completed solver plans and resources

These 12/20 rows mean solving, top-3 candidate timing and selected
plan audit completed; they do not mean full-request correctness passed.

| cell | solve_seconds | mode | grid | residency | kappa | ec | rq | variant_count | fp64_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_prefill_B1 | 1663.6373534370214 | L1 | 128 | 1 | 1 | 1088 | 16 | 5 | 0 |
| llama_prefill_B2 | 2990.9885147659807 | L1 | 128 | 1 | 1 | 1088 | 32 | 5 | 0 |
| llama_decode_B1 | 1145.9828607119853 | L1 | 128 | 1 | 1 | 64 | 4 | 5 | 0 |
| llama_decode_B2 | 1445.3676811960759 | L2 | 128 | 1 | 1 | 128 | 4 | 5 | 0 |
| llama_decode_B4 | 1572.8573534750612 | L1 | 128 | 1 | 1 | 256 | 4 | 4 | 0 |
| llama_decode_B8 | 1419.7960633350303 | L1 | 128 | 1 | 1 | 1088 | 4 | 5 | 0 |
| llama_decode_B16 | 1417.5804216819815 | L1 | 128 | 1 | 1 | 1088 | 4 | 4 | 0 |
| qwen3_prefill_B1 | 1691.533057575929 | L1 | 128 | 1 | 1 | 1088 | 16 | 4 | 0 |
| qwen3_prefill_B2 | 2972.932516042958 | L1 | 128 | 1 | 1 | 1088 | 16 | 5 | 0 |
| qwen3_decode_B1 | 1451.9026450649835 | L1 | 128 | 1 | 1 | 64 | 2 | 4 | 0 |
| qwen3_decode_B2 | 1971.1001926590689 | L2 | 128 | 1 | 1 | 128 | 2 | 5 | 0 |
| qwen3_decode_B4 | 2415.9502018699422 | L2 | 128 | 1 | 1 | 256 | 2 | 4 | 0 |


Full per-class tile/stages/split selections are in
[checkpoints/20260926T044401Z/plans.tsv](checkpoints/20260926T044401Z/plans.tsv); raw plan manifests and command/results are
in `plans/<cell>/`. Timing phases, pruning domain counts, top-3 predicted/actual
rankings and selected L1/L2 register/shared/spill counts are in
`solver_phases.tsv`, `pruning_domains.tsv`, `top3_rankings.tsv` and
`kernel_resources.tsv` under the same immutable checkpoint directory.

## 8. Backend evidence

`backend_coverage.md` maps every serving operator to its implementation and
tests, with the merge vectorization gap above. The separate GEMM mainloop
microbenchmark (`collective_bench/result.tsv`, three fresh processes per point)
measures about **27–32 GB/s per single CTA** for serving versus **18–21 GB/s**
for legacy; at one CTA per SM, serving reaches **819–846 GB/s**, legacy
**771–819 GB/s**. This is not end-to-end speedup evidence.

The Llama B1 diagnostic trace at past575 gives effective QKV/output/gate-up/
down/lm_head rates **254/676/864/831/929 GB/s** and attention **86 GB/s**.
See `trace_probe.md`. Its instrumented L2 event mean is 4.316 ms, while the
selected uninstrumented plan uses L1; the trace path reconstruction is 7.84%
short of that instrumented mean. Three remaining trace cells are queued after
EV-1. All 12 completed selected plans have FP64 count zero.

## 9. Gap decomposition and R11 inputs

Model counterfactuals currently cover eight completed decode plans in
[checkpoints/20260926T044401Z/gaps.tsv](checkpoints/20260926T044401Z/gaps.tsv). They contain synchronization, fixed,
contention and chain-delay terms; PG upper bounds; RMSNorm chain costs;
attention shares; and protocol constants. They are model estimates, not
measured decompositions. The complete ten-cell measured comparison and
launch-gap ×1023 bounds await EV-1 and the remaining trace cells.

## 10. Cost model

The calibrated in-flight and serving TaskBody profiles and raw observations
are in `calibration/`. `FlowPreparation.cpp` and the serving runtime use the
shared runtime dependency-window helper. Selected-plan three-point predictions
and flow pieces are retained in `plans/<cell>/winner.*`; post-measurement
prediction errors and rankings will be recomputed from EV-1.
The earlier attention workload-unit error and superseded search outputs remain
marked in `price_audit/` and are not accepted as final performance evidence.

## 11. Failed/open items: location and next action

- **G-7:** `SkeletonSearch.cpp:61-92` rebuilds the serving structure and clears
  flow/price reuse when argmax tile-N or attention geometry changes;
  `:99-179` prepares candidates at three past points.
  `FlowPreparation.cpp:281-353` derives/prices uncached spaces and maps their
  pieces; `PiecePricing.cpp:60-81` handles causal singleton fibers.
  Top-3 full compilation adds about 300–390 s in completed plans. Preserve
  separate structure caches for `(Ec,Rq,argmax_tile_n)`, reuse unaffected class
  price/release data, and reuse compiled variants across related plans; validate
  score equivalence before treating the budget as closed. Estimated work:
  1–3 engineering days plus the required plan rerun; this is an estimate.
- **Merge:** implement vector partial loads and shared LSE normalization while
  keeping lowest-risk existing numerical semantics; rerun attention unit and
  full-request checks. Estimated 0.5–1 engineering day plus experiments.
- **Remaining gates:** no conclusions until queued samples finish. G-6 Qwen3,
  remaining plans, full C-1/C-2, G-9, G-10 and G-11 remain explicitly open.

## 12. Scope and resume contract

No PG, tile-direct TF, synchronization primitive redesign, RMSNorm-to-GEMM
fusion, TMA/WGMMA/tcgen05 implementation, sampling/continuous batching/block-table/tensor-parallel/quantization work, MPK scheduler/device multi-step loop,
CUTLASS submodule change or MIDPOINT_REFINE enablement is claimed. Legacy
continues to be checked by the existing CTest suite.

On resume: inspect running chain/queue outcomes, regenerate report tables,
inspect every failed cell, finish the required final report and documentation,
then commit and push the actual results. This checkpoint does not certify R10
as fully completed.
