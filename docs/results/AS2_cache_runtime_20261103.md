# AS-2: cache retirement and bounded QKV DMA acquisition

Status: **in_progress**, five dense models/B1/B16; MoE deferred. Frozen
framework baselines, random parameters, model sizes and numerical policy are
unchanged. TaskBody/Pipeline prompt SHA256:
`ecec4e795defb8a467a653f7258852f7bd05890549137338478a3c329a2e11d1`.
Starting HEAD81f15f382; feature commitb48d3eb49, Junhui Peng
<pengjh0111@hnu.edu.cn>. No shared environment/toolchain upgrade.

## Cache retirement

`verified-run`: classify only AS2 `cache/pycache/*.pyc` and pip download cache,
register completed/regenerable identities, dry-run, then apply the existing
manifest/path/stamp/link/process-reference guards. Remove31507 files,
578534643 logical bytes,650686464B of allocated file blocks. The223 build
executables inventoried before deletion retain their SHA256 values afterwards.
All builds, including the most recent two batches, weights, numerical/failed
evidence, selected images, installed environments, baseline wheels and uv
package archives remain. AS1 dependencies are outside this cleanup selection.

Process observation finds no unfinished own job or readable open/mapped cache
reference. Other users' inaccessible `/proc` entries are not claimed inspected;
only regenerable bytecode/download files are retired, never loaded code,
installed packages or processes. No account-wide cache scan/removal occurs.

Generated usage sampled after cache retirement is140.376316GiB. Repeated C3
captures are subsequently shared only after completed-job registration,
byte-identity checks and dry-run; all paths/payloads remain and C3 is re-audited.
The first storage-sharing attempt correctly rejects unregistered captures
before mutation, then succeeds after explicit retained-result registration.
The older150.121319GiB excursion/failing disk Gate remains recorded; this
cleanup does not retroactively pass it. Budget remains150GiB.

Completed within/across-cohort sharing reclaims10309988352B without losing
any path or payload; all four C3 cohorts pass the post-sharing audit. Latest
sample143.459900GiB generated,8231.567215GiB free on the shared filesystem.

Evidence: `runs/as2/cleanup/development_cache_20261103_v1/`,
`cleanup/qkv_dma_{control,candidate}_{preview,apply}_20261103_v1.json`, and
the corresponding priced-C3 sharing receipts.

## Actual optimization path

`verified-code/run`: full Qwen3-0.6B config/random inputs → serving import →
ModelDescription/CTG/ISL → sourced implementation prices → native
structural/EFT placement/FIFO/frontier audit → Plan materialization → Ascend
Codegen → CANN9.1.0 → one persistent launch per model forward. No final queue,
Host operator fallback or alternate mathematical model is handwritten.

GEMM variant11 is bounded to28 QKV projections of **Qwen0.6 B1 decode/G8**:
M1/N4096/K1024,32 tiles/stage, BF16 input/output and unchanged FP32 accumulate.
These operands are consumed only through CATLASS MTE2. Its producer completes
MTE3/FIX before publishing readiness. The first task on each worker retains
the conservative acquire; immutable descriptors may then be read before the
payload-specific boundary. Only the registered AIC recipe replaces entire
DCache maintenance with the measured DMA-consumption DSB path. Scalar payload
consumers, AIVs, Attention, readiness/owner/epoch/FIFO and completion stay
conservative. This is not a general acquire/release claim for Scalar reads.

Actual bound tables contain896 affected tasks, eight first-queue tasks still
conservatively acquiring:888 whole-cache operations omitted per decode. These
are operation counts, **not** TPOT shares. Bootstrap tables/ownership/queues
are byte-identical to control except28 Stage implementation IDs. Explicit
source capability, shape/topology/count checks and old-image rejection keep
the experiment fail-closed. ABI layout/version4 is unchanged.

Entry points: `QkvDmaContract.h`, `DevicePrimitives.h`, `LlmExecutor.h`,
`LlmCubeBody.h`, `LlmRequestHost.h`, `AscendLlmCodegen.cpp`,
`tools/as2_compile_llm.cpp`, `as2_request.py` and `as2_native_prices.py`.
Configuration: `configs/ascend/as2_qkv_dma_qwen06_b1_decode_experiment.json`.
The existing pinned CATLASS Block/Tile GEMM remains the mathematical Body;
CATLASS SHA2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52.

## Price and pipeline evidence

`verified-run`: five fresh finite native-cost processes measure the new
dispatcher path, and each actual recipe is profiled at past95/319/799.
Context counts64/384/575 cover1023 decode forwards. Generic AIC dispatcher
789.796875ns versus DMA-policy449.156250ns are **component measurements**,
with contended provenance retained. One first acquire is amortized over32
probe tasks; this is a context-local approximation, not exact FIFO startup
or a portable hardware constant. Unknown/legacy dispatcher prices reject
variant11 instead of becoming zero cost.

Two complete request recipes cause three genuinely solved phases with a
shared identical priced prefill. Native search selects the DMA candidate:

| Recipe | Predicted prefill,ms | Weighted decode,ms | Request,ms |
|---|---:|---:|---:|
| Conservative |30.992305|5.176756|5326.813578|
| QKV DMA |30.992305|5.148389|5297.793960|

Predictions are not speedup evidence. Full config/CTG/ISL/price/schedule/Plan/
source/image hashes are linked by
`search/qkv_dma_priced_pipeline_20261103_v1.json` and its source-commit receipt.
Image SHA256:
`1e2ef30f631c32cceabbe84125f62133cc8532d0e05222617e3beeb7c13b55c5`.

## Correctness and performance admission

`verified-run`, physical1/55:00.0, logical0:

- Both bootstrap and actual priced arms pass full64/1024 natural greedy,
  teacher512 (67 captures, observed max-abs0, zero prediction differences),
  fast/diagnostic C2, and C3 five fresh processes/two reset requests each.
- Bootstrap control/candidate457 common full-request numeric files are
  bitwise equal. Full request receipt records1024 launches for1024 forwards;
  teacher records513 forwards, each one launch. No baseline engine is rerun.
- Source/Host guards,31 request tests,14 pricing tests and two new policy
  tests pass; explicit native compiler/CANN builds pass at nice10/parallel1.
  Missing capability, wrong geometry and legacy unknown costs reject.

Evidence: `correctness/taskbody_qkv_dma{,_priced}_20261103_v1/`,
`search/qkv_dma_context_calibration_20261103_v1/`, and
`bench/taskbody_qkv_dma_priced_pairs_20261103_v1/`.

Five alternated fresh-process pairs, one warmup/three complete64/1024
requests per process, record the following process-median summary:

| Metric | Conservative | QKV DMA |
|---|---:|---:|
| TPOT,ms |5.002505|5.005739|
| TTFT,ms |34.061590|33.821300|
| E2E,ms |5151.669279|5154.920325|
| Output token/s |198.770524|198.645165|

Paired geometric ratio0.971904, log-t95%[0.892417,1.058471], CV0.54%/6.95%.
**No reliable gain/inconclusive**;8/10 receipts have external-PID contention
flags. The slow fourth-pair candidate remains in the results. All40 natural
token payloads match. Performance submissions follow two util0 observations
five seconds apart; idle observation reserves nothing. No default or formal
eager performance Gate passes. Microsecond dispatcher improvement does not
establish an end-to-end improvement.

The next bounded component, `as2_qk_phase_drain.cpp`, retains FP32 QK math,
physical score stride1088, one live CATLASS MMAD per phase and disjoint tile
outputs, but completes FIX once at phase exit. G1 passes5/5 new processes,
four reuse epochs, rows2/D128/keys129 and rows4/D64/keys1087. Control/candidate
are bitwise equal; independent FP32 max-abs1.788139e-7 passes the predeclared
2e-5 absolute/relative policy, and capacity suffix/guards remain intact.
These are standalone component results, **not** model/price/TPOT admission.
See `tools/probes/ascend950/AS2_qk_phase.md` and
`probes/qk_phase_drain_20261103_v1/`.

## Remaining scope and safety

This validates one bounded DMA consumer family, not all Scalar/DMA cache
paths or the whole TaskBody/Pipeline plan. CATLASS49 streaming/local Attention
under frozen numerical gates, GEMM start/II/drain/session overlap, credit-slot
retirement, dual-AIV compute cooperation, wider shape/G selection and fresh
five-dense B1/B16 performance admission remain. Other models do not inherit
this Qwen-specific admission. All new defaults remain off pending reliable
complete-request improvement.

One own NPU job at a time, physical1 only; physical0's Alarm remains a hold.
Functional jobs may coexist with external load; performance waits for observed
idle. No device/advisory lock, exclusive mode, reset, other-process operation,
CUDA regression, official-weight download or shared configuration change.
