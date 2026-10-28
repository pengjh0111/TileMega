# AS-2 pipeline and Prompt reconciliation — 2026-10-26

Status: **in_progress; G-PIPELINE is not passed**. Latest source recheck at
`c44d98f959e15ae31273e205e4bdf558e1531651`; historical runs retain their own SHA.
`verified-code` means inspected source; `verified-run` means retained actual
execution. Earlier runs retain their original date/device/software identity.
Prompt SHA256: `673b6deb619cb02f1c3f2d0875af2bdafb835bca09e3875d7dbfb840b1be00d6`.
The user's subsequent instruction defers MoE: five dense models, ten B1/B16
main cells; two MoE cells remain deferred, never passed or removed silently.

## Actual compilation and execution path

```text
local full model config + canonical random weight/input manifests
  → serving export / BuildModelPlan / serving semantic lifting
  → ImportSemantics / InstantiateForGranularity
  → ModelDescription / CTG task spaces / ISL element dependencies
  → bounded implementation or decomposition binding
  → symbolic preparation / typed legal WorkerMap
  → task prices + structural versus rank/EFT placement/FIFO selection
  → dependency/FIFO/participant audit + producer-frontier waits
  → MaterializeAscendLlm / POD tables / implementation contracts
  → LowerAscendLlm / generated PersistentLlm / CANN native build
  → ACL load / named-arena phase binding / one persistent launch per forward
  → TileMega-owned CATLASS tile + native AIV Body execution
  → logits/greedy token + per-layer KV state
  → independent eager comparison / C-2/C-3 / request timing
```

1. **Identity and import — verified-code.**
   `python/tilemega/bench/as2_models.py::{export,weights}` reads pinned
   `configs/ascend/as2_models/`, seed20261016, stable per-FQN generation and
   tied aliases. Canonical weights are generated once, with tensor hashes.
   `as2_native.py` packs by name/layout; no values are specialized or cached.
   `tools/as2_compile_llm.cpp` consumes an actual serving export, calls
   `BuildModelPlan`, `ImportSemantics`, `InstantiateForGranularity`, MLIR
   verification and `ModelDescription::FromCouplingGraph`. Generic/degraded
   semantics fail. Full layers, GQA3, Qwen Q-width, RoPE scaling and tied heads
   are part of the imported model, not model names attached to an Encoder.

2. **CTG and implementation binding — verified-code.**
   Geometry and cross-task split-K are bound before CTG instantiation; task
   ownership and dependencies are regenerated. `PrepareSymbolicProblem` and
   `BuildPlanSkeleton` retain the existing symbolic/ISL machinery.
   `AscendTasksFromSymbolic` queries its reverse dependency oracles; final
   queues are not handwritten. Serving explicit ownership survives phase
   binding, ModelDescription reconstruction and FlowPreparation.
   Local K loops inside one indivisible MIX Body are distinguished from CTG
   partial/combine tasks; their scratch/layout/traffic remain explicit.

3. **Finite optimization — verified-code, partial.**
   `CandidateGenerator::{SetAscendDomain,Analyze,TopK}` and `as2_rank_gemm`
   compare measured legal component candidates and reject absent native prices.
   The whole-model CLI currently takes an explicitly selected geometry/Body/
   decomposition. Its `emit-priced*` path reads positive contextual prices,
   checks model/config/phase/B/G/runtime-ABI/stage/shape identity, and compares
   **two real placement/order plans for that fixed structure**.
   `ScheduleAscendTasks` uses critical rank, heterogeneous earliest finish,
   explicit MIX participants and one device-shared GM calendar.
   `PriceAscendSchedule` evaluates both fixed FIFO orders under the same model.
   The lower predicted makespan wins; materialization preserves its queues.
   This is real native schedule optimization, **not completed outer
   CompilerSearch/JointSearch over geometry, decomposition, Body and G**.
   A TopK component test does not establish whole-model hierarchical search.

4. **Plan and waits — verified-code.**
   `AuditAscendSchedule` rejects illegal roles/participants and cycles in
   semantic dependencies plus all participating worker FIFO edges.
   Per-producer maximum predecessor ordinal lowers dependencies to
   `(worker, epoch, required_prefix)` waits. Same-worker order is covered by
   FIFO; no cross-role same-group relationship is treated as worker identity.
   `MaterializeAscendLlm` checks sourced prices for optimized plans and emits
   checked 64-bit arena offsets, stage/task/queue/wait tables, implementation
   contracts and bounded epoch values. Large addressing and >4096 tasks have
   real NPU evidence. This does not waive future fanout/reuse-lifetime audits.

5. **Codegen and build — verified-code / verified-run.**
   `AscendLlmCodegen.cpp::LowerAscendLlm` emits `PersistentLlm.cpp.in` with
   supported Body traits. `reemit` reuses an already solved Plan and explicitly
   reports no new solver call; it does not rewrite queues.
   `cmake/AscendAS1.cmake` and `tools/ascend/CMakeLists.txt` provide native
   host/CANN targets. The top-level CMake returns before archived CUDA targets.
   BiSheng/CANN uses `--npu-arch=dav-3510`; selected targets build serially with
   nice10. Cached binaries are reused only with retained source/header/flag
   identities. No NVCC, CUTLASS scheduler or ACLNN per-op fallback is used.

6. **Request and device execution — verified-code / verified-run.**
   `as2_request.py` relocates separately solved prefill/decode by buffer name
   into a shared weight/KV/token arena, preserving queue/wait bytes.
   `LlmRequestHost.h::RunRequest` resets requests, launches prefill for output1
   then1023 stream-ordered decode forwards for outputs2..1024. No Host solve,
   repacking or per-layer launch occurs in that loop. Runtime past is real;
   KV1088 is capacity, not the effective attention length.
   `LlmExecutor.h::ExecuteLlm` traverses each static worker FIFO within the
   forward. G MIX blocks yield G AIC +2G AIV workers; numBlocks is G, not3G.
   Current admitted G is at most8. Both AIV lanes run independent tasks;
   cooperative MIX tasks currently use AIC+AIV0, not both AIV participants.
   Owner ready/done/reverse-ack protects scratch reuse; only the owner publishes
   logical task completion. Diagnostic/fast use the same Body/Plan contract.

7. **Body and synchronization — verified-code / verified-run.**
   `GemmTile.h` wraps CATLASS BlockMmadTla for coordinates supplied by TileMega;
   it does not ask BasicMatmul/DeviceGemm to schedule the whole operator again.
   `LlmCubeBody.h`, `LlmServingBody.h` and `LlmVectorBody.h` implement GEMM,
   cooperative QK/softmax/PV, Norm/QK norm, RoPE/KV, SwiGLU/residual and greedy.
   Cooperative attention uses explicit FP32 opmath/cast contracts and real GQA.
   `ProgressPrimitives.h`/`DevicePrimitives.h` retain conservative measured
   DSB/cache visibility; an AtomicAdd is not assumed to provide release/acquire.
   GM remains the general route. BF16 direct, UB→L1, Mode4, SIMT and larger G
   remain disabled until separately validated and evaluated.

8. **Verification and measurement — verified-code, admission incomplete.**
   `as2_compare_request`, `as2_teacher` and `as2_native_audit` distinguish natural
   greedy, teacher forcing, fast/diagnostic equality and multi-process reset.
   Independent `as2_eager`/`as2_native_reference` use the same canonical weights,
   not TileMega Bodies. Frozen tolerances allow stated legal FP32/BF16 error;
   intermediate bitwise replay locates drift but is not a performance objective.
   Main correctness still requires exact greedy and the declared KV/logit checks.
   These checks validate TileMega. The latest user override waives baseline
   numerical pairing; native engine performance collection is independent.

## AS-1 gap → present state → remaining AS-2 obligation

| Gap | Current verified code/run evidence | Remaining obligation |
|---|---|---|
| A01 | Real serving import and `emit-priced*`; full model requests | Complete cost-driven outer structure selection |
| A02 | Two regenerated geometries; native Analyze/TopK and measured component choice | Family/phase/B/past Body comparison inside whole-model search |
| A03 | Actual prices → rank/EFT versus structural queues → native materialization | Wider calibrated domains and end-to-end admission; no zero-cost unknowns |
| A04 | G2/4/8 and both AIV independent-task execution; five-process G8 payload tests | Evaluate larger G; distinguish hardware/validated/selected groups |
| A05 | Same-Plan fast/diagnostic numerical comparisons | Main-cell C-2 coverage and measured removal benefit |
| A06 | Frontier waits, exact dependency/FIFO/participant checks and real priced queues | Further redundant-wait/control-cost reduction with reuse proof |
| A07 | Conservative cache/DSB protocol preserved; finite measured timer-based waits | Justified minimal visibility optimization; control/device-time attribution |
| A08 | Real QK norm/RoPE/GQA3, blocked prefill/decode and cooperative FP32 attention | Faster legal attention/small-M alternatives and full-request evaluation |
| A09 | Full model/layers, 64-bit addresses, dynamic past and1024 request forwards | All ten cells, more bucket/context calibration and compact templates |
| A10 | Single-owner GM MIX lifecycle; true CTG split partial/combine; local tile reduction | Dual-AIV MIX and measured BF16 direct/GM choice, not an assumed gain |

## Evidence example and current numerical matrix

**verified-run** Llama1B B16 uses231434 prefill tasks/116 stages, original CTG
ownership,16 explicit local-tail Body bindings and57344B/group/stage scratch.
Decode has actual three-part CTG projection/combine tasks. The local seven-part
prefill reduction is a guarded internal MIX implementation, not a new CTG split.
Original failed no-split results remain under `failures/`; tolerance is unchanged.

`models/llama3_2_1b/native_b16_{prefill_local_tail7,decode_split3}_priced_v1/`
contains CTG/import/price/search/Plan/source identities. Actual regular/tail
service buckets are priced separately; neither uses a default zero.
Prefill selects EFT946.144483ms versus structural953.482356ms; decode selects
EFT31.539152ms versus structural31.645729ms. **Predictions, not measured TPOT.**
`correctness/llama1b_b16_local_tail7_priced_request1024_numeric_v1.json` passes
all16384 greedy outputs, logits and checked KV against independent eager.
The sourced selected queues also pass512-step teacher forcing
(`correctness/llama1b_b16_local_tail7_priced_teacher512_numeric_v2.json`).
Unpriced structural queues separately pass teacher512, C3=5/5 processes and
C-2=120 common files. These C-2/C-3 results do not prove five priced processes.
Actual CANN launch evidence records8 calls for8 model forwards across two
short reset requests (`probes/llama1b_b16_actual_runtime_launches_v1.json`).

| Dense model | B1 full1024 + teacher512 vs SDPA | B16 full1024 + teacher512 vs SDPA |
|---|---|---|
| Qwen3-0.6B | pass; sourced selected queues | priced full1024 passes; prior diagnostic teacher512 passes |
| Llama-3.2-1B | pass; sourced selected queues | pass; local-tail +split3; priced1024 also passes |
| Qwen3-1.7B | pass; sourced split3 queues | pass; sourced local-tail +split3 queues |
| Llama-3.2-3B | pass; sourced split2 queues | priced decode split2 full1024 passes; teacher512 pending |
| Llama-3.1-8B | pass; sourced separate-QKV/split7 queues | priced selective Q/O/down-tail3 full1024 passes; teacher512 pending |

**Ten numerical long cells, zero formal final correctness/performance cells.**
The latter still require final TileMega configurations and applicable
C-0/C-1/C-2/C-3 coverage;10/10 numerical runs do not complete those Gates.
Price and capture histories are retained in `runs/as2`; exact selected bindings,
binaries and next commands are indexed by `checkpoint.json`/`resume.json`.

## Prompt direction and corrections to the work order

- The objective is **complete dense-model requests with reproducibly better
  TPOT than the strongest legal torch_npu eager**, under the same BF16/random
  weights/greedy/KV contract. Numerical repair is a prerequisite, not the endpoint.
- Do not continue unbounded bitwise micro-probes as the main optimization loop.
  Each diagnostic must answer a failing model check or a quantified bottleneck,
  retain its failure evidence, then return to the complete model request.
- Preserve all ten numerical long cells without shrinking layers/vocab/output
  length. New teacher512/final coverage remains. Keep MoE deferred.
- Connect bounded Body/geometry/decomposition/G choices to the existing semantic
  and schedule chain; component TopK plus a fixed whole-model Body is incomplete.
  Reuse shapes and source-identical binaries; no large sweep or batch build.
- Use full-request Host/device measurements and controlled ablations to select
  the next optimization: dispatcher/sync, queue critical path, resources, Body,
  or handoff. Predicted objective reduction cannot admit a default or speedup.
- Latest user override: no baseline numerical comparisons. Collect eager,
  compile, vLLM and SGLang once on idle hardware for five dense models and
  B1/B16 (40 baseline cells), then collect ten TileMega cells for user review.
  B2/4/8 are deferred by the latest request. Keep
  ISL64/OSL1024, TPOT=(t_last-t_first)/1023, warmup1/timed3. Snapshot ratios
  are not contemporaneous paired confidence intervals; raw samples remain.
- Private paths only; one own NPU job/card, no device locks or external process
  control, no system changes/downloaded weights/CUDA regression. Manifest,
  finite timeouts and whitelist cleanup remain acceptance requirements.
- On2026-10-26, a prelaunch telemetry query returned DCMI-8005 and launched no
  model. A later query was healthy but NPU ID↔PCI correspondence had changed.
  Mapping is time-bound; re-probe ACL/PCI before new model runs. Never attribute
  a telemetry failure to an NPU kernel deadlock or silently reuse old PCI identity.

## Experimental FIFO observation reuse, 2026-10-26 continuation

`verified-code`: `ElideAscendFifoWaits` runs after priced plan selection. A wait
is omitted only if every current participant already observed the required
producer prefix earlier in its own FIFO (or is that producer). Observations
are not transferred between workers or inferred through another producer.
`AuditAscendSchedule` reconstructs coverage of the original semantic edges
under the combined dependency/FIFO graph. A newly participating AIV must wait.
The default remains per-task frontier lowering; only explicit `emit-priced-fifo`,
`emit-priced-decomposed-fifo` or `emit-priced-body-fifo` enables the experiment.

`verified-run CPU`: existing schedule contracts pass, including64 enumerated
small dependency graphs and negative missing-observation/MIX-lane cases. Real
Qwen0.6 B1/G8 CTG/ISL/pricing/materialization yields decode9267→6295 waits
(4158 tasks), prefill124083→45304 (16485 tasks). Queues, offsets, Stage/buffer
tables and task owner/ordinal/participants match the original selected Plans.
The same current codegen emits identical source for compressed/uncompressed
Plans; older archived sources also contain unrelated Host-template changes.
Prefill offline wall171.74s, user CPU58.87s, peak RSS240936KiB are retained;
these are compiler costs, not model latency. No payload acquire, publication,
MIX ready/done/reverse-ack, epoch encoding or Body code was removed.

`verified-run NPU`: ordinary `/usr/local/sbin/npu-smi info` succeeds (7.44s
in project env,3.66s before sourcing it); both cards are healthy. Earlier
15s timeouts are query observations, not evidence of a compute/device failure.
CANN changes libc_sec/libmmpa resolution. Management children now prefer driver
libraries; compute environments are unchanged. Intermittent timeouts also occur
with driver precedence, so that difference is not a proven root cause.
Query deadlines are configurable30s by default, at most60s, plus bounded1s
reaping. Seven CPU safety tests pass; no shared configuration is changed.

Qwen0.6 B1/G8 uses one identical retained binary for original/elided queues:
457 short diagnostic payloads per request match; fast/diagnostic401 common
files match. Five fresh fast processes /10 reset requests pass C-3,401 files
per request. Full1024 greedy tokens,151936 final logits and56 checked KV slices
are bitwise equal to the independent SDPA/fused-Norm reference. Teacher512
also passes,513 launches, zero prediction/logit/KV differences. Each model
forward remains one persistent launch. Evidence: `search/fifo_device_validation_v1.json`.
The wait mode remains experimental/off by default: no paired TPOT admission,
refitted prices or wider-model C-3 is claimed. At that checkpoint seven long numerical cells and
zero formal performance cells were admitted only against SDPA. Nine compute jobs used physical1
(55:00.0→logical0),54.72s own-child time, no abort/timeout/OOM; external load
is retained. Two query-only preflight failures launched no compute work.
CPU semantic tools still link no CANN DSOs; explicit Probe compilation passes,
not runtime Probe execution. C3 dedup preserves2286 paths/payloads and reclaims763637760
allocated bytes; re-audit remains5/5. Project sampling briefly reached
103300968448B, then102540890112B, below the96GiB soft budget. No new large
candidate was started while over budget. AS-1 and unrelated processes are untouched.

Latest continuation: Qwen1.7 B16 local-tail6144 + CTG split3 now passes full1024
and teacher512 with actual sourced EFT-selected queues; eight numerical cells,
zero formal cells. See `AS2_950pr.md` and `AS2_baselines.md`. Outer geometry,
Body/decomposition/G search and final TileMega performance remain partial.
Baseline scope now follows `AS2_baselines.md`; no further engine numerical pairing.

The next Llama3B B16 bootstrap genuinely imports/solves460810 prefill and11578
decode tasks,11,389,044,224B shared arena, G8/24 workers. A short four-output
request uses four persistent launches but fails5/64 greedy tokens; first
divergence is batch row3/output1 (zero-based). No long-cell pass is claimed.
Phase-sized Host-only diagnostic extents now survive arena relocation;17 CPU
tests pass, core Plan tables remain byte-identical. Device capture is blocked
at ACL device initialization507033 before launch, with TsdOpen/CreateSession
errors. A successful management query also reports physical1 Critical/HBM0/0;
no reset/retry/new NPU work is performed after this observation.

## TPOT continuation, 2026-10-28

`verified-run`: five-dense full forward semantics remain intact. The follow-up
[TPOT report](AS2_TPOT_optimization.md) separates semantic completeness from
remaining outer search and acceptance. Qwen0.6 B1 FIFO elimination has no
reliable gain in five fixed process pairs and stays non-default. Qwen0.6 B16
valid-key QK64 preserves original arithmetic/queues/waits and passes repriced
full1024 bitwise. Five fixed fresh process pairs now give geometric speedup
1.100785, paired-log-t95%[1.071267,1.131116]; medians50.683→46.248ms.
Teacher512, full1024 C2 and C3 five processes/ten reset requests pass on the
same measured image. This is local gain versus TileMega control, not eager
G-PERF; all shared-load samples remain retained.
All baselines remain frozen; wider Body/geometry/decomposition/G selection,
new final-config teacher/C2/C3 coverage and the other cells remain open.

The next CPU entry, `python -m tilemega.bench.as2_search --domain <private JSON>
--out <new private directory>`, bounds the outer domain at four sourced
candidates. Each candidate calls the same native `emit-priced*` semantic and
heterogeneous schedule path; predicted selection points at actual materialized
tables/source. Calibration identity, duplicate-domain and cost-to-selection CPU
guards pass. Same-card two-Body solves select valid-QK; execution tables/source
match the device-tested Plan byte-for-byte. Subsequent measured subphase
attribution identifies repeated BF16→FP32 KV conversion/tail initialization.
Body9 adds request-reset derived FP32 state, preserving authoritative BF16 and
mathematics. It passes repriced full1024, teacher512, C2 and C3 5/5. A second
actual native outer search selects Body9, predicted52.379→33.485ms; full-request
timing admission remains pending. Source/table receipts link both outer results
to their NPU evidence. This bounded family still does not complete multi-family
geometry/decomposition/G optimization or original G-PIPELINE. See the TPOT report.
