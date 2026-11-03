# TaskBody/Pipeline: native local residual continuation

Status: **in_progress**, five dense models/B1/B16; MoE deferred. Branch
`tilemega-ascend`, author Junhui Peng <pengjh0111@hnu.edu.cn>. Frozen framework
baselines, model dimensions, weights, inputs and numerical policy are unchanged.
TaskBody prompt SHA256:
`ecec4e795defb8a467a653f7258852f7bd05890549137338478a3c329a2e11d1`.
Generated-artifact budget150GiB; no dependency/toolchain upgrade.

## Actual path and implementation

`verified-code/run`: full Qwen3-0.6B config/random inputs → serving import →
ModelDescription/CTG/ISL → legal resource/implementation binding → native
structural/EFT placement and FIFO → Plan materialization → Ascend Codegen →
CANN9.1.0/BiSheng15.0.5 → one persistent launch per model forward. No final
queue is handwritten and no Host operator/library fallback is introduced.

New GEMM variant9 is explicitly bounded to G8, Qwen0.6 B1 **decode**, its56
O/down residual projections (448 tile tasks/forward), M1/N1024/K2048 or3072.
`LocalGemmTile.h` composes pinned CATLASS Block/Tile components, with NO_SPLIT
Fixpipe output explicitly selecting same-group AIV0. CATLASS remains
2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52; its whole-operator scheduler is absent.

`LlmCubeBody` completes FIX before the existing ready publication. The AIV0
`LlmServingBody/LlmVectorBody` residual consumes BF16 directly from UB, adds
the BF16 residual with unchanged FP32 opmath/cast boundaries, completes MTE3,
then follows the original done/reverse-ack/unique-owner frontier protocol.
There is one blocking slot, no asynchronous task retirement or credit region.
Raw BF16 GM output is deliberately retained in fast/diagnostic paths for
teacher/C1/C2; this first experiment removes its **read**, not its write.

One vector TPipe reserves136192B, containing the106496B scratch and receive
slot131584..135680, plus guards/reserved separation. A second CATLASS Resource
is not constructed on the live vector worker. The source capability, shape,
budget, Host/device checks, implementation identity and GM-traffic pricing
all recognize this route; old images and incompatible prices reject it.
Target-wide flag/direct capability bits are not silently generalized.

Entry points: `LocalResidualContract.h`, `LlmCubeBody.h`, `LlmVectorBody.h`,
`LlmServingBody.h`, `tools/as2_compile_llm.cpp`, `AscendLlmCodegen.cpp`,
`as2_request.py` and `as2_body_profile.py`. Experimental configuration:
`configs/ascend/as2_local_residual_qwen06_b1_decode_experiment.json`.

## Numerical and lifecycle evidence

`verified-run`, physical1/55:00.0:

- Transport component: G2 and G8 each pass5/5 fresh processes, four reuse
  epochs, independent group inputs, M1/M16, N128/N129 and K8192/K14336.
  GM/local outputs are bitwise equal; independent FP32 references pass the
  frozen BF16 policy. Padding/guards and AIV1's private UB witness are intact.
  This proves AIV0 transport/isolation, not a dual-AIV compute epilogue.
- Both native bootstrap control/candidate pass full64/1024, all1024 natural
  tokens identical, and teacher512 with67 captures, zero observed max-abs
  error and zero differing predictions. C2 matches401 common files exactly.
- Each arm passes5/5 fresh C3 processes/two reset requests each,457 captured
  files/request. Byte-preserving storage sharing is followed by another
  successful audit. The new local route is actually selected in the device
  Stage table; it is not a compiled-but-unused Body.
- Native Host guards reject absent capability, wrong M/N/K, epilogue, task
  domain, split reduction and missing workspace.31 request/relocation tests,
  native schedule contracts and CANN compilation pass.

Image SHA256:
`3915063a696b319f059f960fbf85e53a0880502950ee3b97bd2da6c7728c1ad2`.
Evidence: `runs/as2/probes/local_gemm_witness_g{2,8}_20261102_v{5,6}/`,
`correctness/taskbody_local_residual_20261102_v1/`,
`search/local_residual_build_20261102_v1.json` and
`search/local_residual_fixed_queue_20261102_v1.json`. Native bootstrap
metadata/tasks/waits/queues/allocations are byte-identical between arms;
only residual implementation variants change. No other dense cell inherits
this admission.

## Sourced optimization and shared phases

`verified-run` CPU/NPU: both actual recipes are profiled at past95/319/799,
with context counts64/384/575 covering all1023 decode forwards. Positive
sourced native prices include dispatcher, control and logical traffic.
Interference remains in provenance; these are context-local measurements,
not portable hardware constants. Reusing cached-GM prices for local variant9
is rejected by the real native compiler before final Plan output.

The first request search exposes a real gap: identical prefill candidates are
rejected when only decode changes. `as2_search.py` now reuses an explicitly
identical **priced** phase, checking compiler/input/receipt/table hashes.
Different measured prices are not interchangeable. Changed-price and changed
native-table CPU guard checks pass. No queue is reconstructed in Python.
Two recipes use three genuinely solved native phases, retaining each phase's
structural/EFT comparison and full request objective. The selected prediction:

| Recipe | Prefill,ms | Weighted decode,ms | Request,ms |
|---|---:|---:|---:|
| Cached GM control |31.448328|5.318252|5472.020343|
| Local residual |31.448328|5.242454|5394.479103|

These are **predictions**, not request timing. New actual priced plans are
bound with full CTG/ISL/import/schedule/Plan/source/image hashes in
`search/local_residual_priced_pipeline_20261102_v3.json`.
Both actual priced plans independently pass full1024/teacher512/C2/C3 in
`correctness/taskbody_local_residual_priced_20261102_v3/`; the later admission
is linked without rewriting the original pipeline receipt. Five alternated
fresh-process pairs, one warmup/three complete requests each, give:

| Metric | Priced GM control | Priced local residual |
|---|---:|---:|
| TPOT process median,ms |5.153418|5.061302|
| TTFT process median,ms |34.504494|34.554740|
| E2E process median,ms |5306.497051|5212.441614|
| Output token/s |192.970992|196.453040|

Paired ratio1.022795, log-t95%[0.954411,1.096077], process CV3.33%/3.69%.
**No reliable gain/inconclusive**: the interval crosses1 and3/10 processes
are contended. All40 natural token payloads match. No slow samples are
discarded, no new default is enabled, and no formal eager G-PERF passes.
Evidence: `bench/taskbody_local_residual_priced_pairs_20261102_v3/`.

## Llama1B priced continuation

`verified-run`: both actual sourced uncached/cached plans pass full64/1024,
teacher512 (43 captures, zero observed error/differing predictions), C2 and
5/5-process C3. Full-request timing uses5 alternated new-process pairs,
one warmup plus three complete requests/process, same physical1 and image.

| Metric | Uncached control | Cached KV |
|---|---:|---:|
| TPOT process median,ms |7.690514|7.296456|
| TTFT process median,ms |35.063864|35.043954|
| E2E process median,ms |7902.391213|7499.361564|
| Output token/s |129.581031|136.544957|

Paired ratio1.080439, log-t95%[1.009846,1.155967]. **Inconclusive:** control
process CV5.08% exceeds the frozen5% bound;9/10 receipts are contended.
The interval does not override these failures. All samples are retained under
`bench/taskbody_llama1b_context_priced_pairs_20261102_v3/`; no default or
formal eager G-PERF pass follows. Frozen framework baselines are not rerun.

## Safety and remaining work

Physical0 remains Alarm with no own compute. Physical1 is healthy; functional
jobs may coexist with load when HBM permits. Performance starts only after two
util0 observations5s apart; idle observation reserves nothing. One own NPU job,
finite timeouts, no device/advisory locks, reset or other-process operations.
CPU builds remain nice10/parallel1; no CUDA regression or weight download.

The generated-artifact budget is briefly exceeded: sampled peak150.121319GiB
against150GiB, caused by accumulating the last correctness cohort before
storage sharing. **The disk Gate fails for this excursion**, despite no shared
filesystem exhaustion. Original evidence/peak samples remain under
`failures/disk_budget_excursion_20261103_v1.json`. Registered byte-identical
C3 storage is shared within/across completed cohorts; every path and payload
remains, followed by successful audits. Final sampled usage is140.929459GiB;
cross-cohort sharing releases7675219968B. The common AS2 runner now checks live
private-root usage plus declared output headroom before creating a child;
default1GiB, `--output-reserve-bytes` for larger captures. CPU safety/topology
tests and an actual no-child overcommit rejection pass. No device/disk lock
or shared-directory cleanup is introduced.

Earlier G1 local M16 Host timeout and priced valid-PV p4 timeout remain failures
with unknown original device phase; later recovery/cohorts do not rewrite them.
Source/image/minimal reproducers are protected. See the versioned local probe
report and [valid-PV record](AS2_valid_pv_20261102.md).

Raw-export elimination is a separate experimental GEMM variant10. It keeps
the reserved allocation but writes/reads no raw GM payload; both capture modes
omit exactly56 implementation-private buffers, checked against the actual
Stage table. Host/materialization reject any operand consumer or workspace
alias. Semantic outputs, full logits/KV, numerical tolerances and teacher
checks remain unchanged. Both raw9/no-raw10 bootstrap arms pass full1024,
teacher512, C2 and5/5-process C3 in
`correctness/taskbody_local_no_raw_20261103_v1/`; candidate C2/C3 compare
345/401 files with56 explicitly declared private omissions. Image SHA256
`6c51ab9f307983dbe32689fdffa4030f54cf768439a6c44cd26f4563d60a2e87`.
Its actual context-priced search also selects no-raw10, predicting
5259.892815→5259.296799ms/request, less than0.1% difference. Both new priced
plans independently pass full1024/teacher512/C2/C3 in
`correctness/taskbody_local_no_raw_priced_20261103_v1/`. A one-pair complete
request pilot (one warmup/three timed requests/process) records TPOT
4.893879→4.886538ms, ratio1.001502. This has no confidence interval or final
performance/default admission; the small difference is not a meaningful win.
Evidence: `bench/local_no_raw_priced_pilot_20261103_v1/`. Further work should
target dispatcher/cache and Attention rather than repeat raw-export tuning.
A DMA-consumer-only
message-passing probe passes5/5 new processes across four directions and
128KiB payloads, with stale Scalar prereads; Scalar consumption and the actual
model runtime policy remain conservative. See [the probe](../../tools/probes/ascend950/AS2_dma_message.md).

Still incomplete: raw-export elimination performance admission,
Task-internal/credit-slot overlap, dual-AIV compute cooperation, CATLASS49
streaming/local Attention under frozen numerical gates, family geometry and
start/II/drain pricing, and new five-dense B1/B16 performance admission.
The earlier ten-cell results retain their original identities. This local
route does not establish full TaskBody/Pipeline or AS-2 completion.
