# TaskBody and pipeline checkpoint — 2026-10-30

Status: **in_progress**. Five dense models/B1/B16 remain the target; MoE is
deferred. Frozen framework baselines were not rerun. This checkpoint adds code
and compile/CPU evidence; it contains **no new NPU numerical or TPOT result**.

Baseline local/remote HEAD: `a3226e1aa5c450dfcb42085ce37bec6a1253bf8a`.
Branch: `tilemega-ascend`; author: Junhui Peng <pengjh0111@hnu.edu.cn>.
Exact latest user instruction:45640 UTF-8 bytes, saved outside the repository at
`/home/d50064638/junhuip/tilemega-as2/Prompt/TileMega_TaskBody_pipeline_20261030.md`.
SHA256: `ecec4e795defb8a467a653f7258852f7bd05890549137338478a3c329a2e11d1`.
Original AS-2 instruction remains unchanged; final commit/push identities are
recorded in ignored `runs/as2/inventory/taskbody_r0_v1/delivery.json`.

## Pipeline and scope reconciliation

`verified-code`: local full model config/random manifests → serving export and
`BuildModelPlan` → semantic import/ModelDescription/CTG/ISL → implementation or
decomposition binding → symbolic task/role preparation → sourced positive
prices and structural/EFT comparison → audited worker FIFO/frontier waits →
`MaterializeAscendLlm` → `LowerAscendLlm`/native CANN → one persistent forward
launch → greedy output and independent per-layer KV. The existing priced path
is real; complete multi-family optimized search and final AS-2 gates remain
incomplete. See [the pipeline audit](AS2_pipeline.md).

The new implementation is selected in that materialization/codegen path;
there is no handwritten final queue, Host per-operator call or replacement
frontend. Bootstrap generation is explicitly unpriced. It cannot be reported
as a measured optimized selection or reuse the old argmax service price.

## R0: actual controls and source-derived work

`verified-run` CPU: `as2_taskbody_inventory.py` froze10 control identities and
116 projection-family rows from actual configurations and binary Plan tables.
The five B16 controls use their latest measured Body9+strided-SwiGLU images,
distinguishing pilot evidence from Qwen1.7's five-pair study and the older
declared-default snapshot. B1 controls preserve their independent identities.
Image/config/canonical-weight digests are checked against retained receipts.

Raw inventory: `runs/as2/inventory/taskbody_r0_v1/{controls.json,projection_families.csv}`.
Ledger entries include MMAD object lifetimes, row/micro-epilogue calls, argmax
scan upper bounds/partials, queue/wait/publication/Acquire calls and early/mid/
late logical payload bytes. These are source/Plan counts, not measured cycle
shares, physical HBM traffic or hardware utilization.

Example, Qwen0.6 B16 decode:8403 logical tasks,18992 argmax row calls and
2430976 elements as the old per-tile Scalar scan **upper bound**;1187 partials
per output row. Removing this scan does not imply a measured TPOT reduction.

## R1: implemented fast-line candidates

`verified-code`, **disabled by default**:

- `ArgmaxTile.h`: batched BF16 row load → FP32 max → matching-value mask →
  minimum global index. No per-element Scalar search. Merge bulk-loads
  value/index partials and uses the same stable tie rule. Vocabulary IDs are
  bounded below2^24, exactly representable in FP32. Reductions exclude padding;
  nonfinite values and invalid IDs return failure rather than publish success.
- GEMM variant6 and kind15 variant1 bind the complete LM head and final merge.
  BF16 rounding still precedes selection; full vocabulary computation, task
  ownership and existing owner/participant/ready/done/reverse-ack stay intact.
  Host, device, source-union and request checks reject missing capabilities or
  invalid shapes. Existing106496-byte vector scratch is borrowed, not enlarged.
- Existing strided SwiGLU variant4 now accepts model rows1/16/64/1024 and passes
  the actual final tile row count. It retains gate16/up16 packing, BF16 SiLU
  rounding and final BF16 multiply output. K256 variant5 remains rows16-only.
- `precision_contract_id=as2_llm_fp32_replay_bf16_casts_v1` enters binding,
  import/implementation/Plan identity, generated source, request, profile and
  prices. Unimplemented contracts and incompatible sourced prices fail closed.
  This does not admit a BF16-probability streaming Attention implementation.

`verified-run` CPU: actual full Qwen0.6 imports, all G8:

| B | Phase | Stages | Logical tasks | New bindings |
|---:|---|---:|---:|---|
| 1 | prefill | 200 | 16485 | 28 SwiGLU,1 head,1 merge |
| 1 | decode | 200 | 4158 | 28 SwiGLU,1 head,1 merge |
| 16 | prefill | 200 | 245955 | 28 SwiGLU,1 head,1 merge |
| 16 | decode | 200 | 8403 | 28 SwiGLU,1 head,1 merge |

For all four, CTG MLIR and symbolic ISL dependency files are byte-identical to
their corresponding frozen controls; logical Stage fields also agree. Only
implementation variants change. No Tensor values were generated or specialized
for this checkpoint. Bootstrap native FIFO is not the priced control FIFO.
Evidence: `inventory/taskbody_r0_v1/{bootstrap_v2,candidate_audit_v2}.json`.

## Source and compiler gates

`verified-code`: pinned CATLASS49 contains `MmadFAIQK`, `MmadFAIPV`,
`PackedTileCopyTlaToUB`, online softmax/rescale, BF16 P in L1 and FP32 score/
output buffers in UB. Its outer block/tiling scheduler remains outside TileMega.
Its actual source uses Mode4 and per-AIV flag offsets; these cannot be pasted
over executor/CATLASS flag7. Buffer lifetimes and small-query AIV partitioning
still require a separate extraction/protocol design and device validation.

`verified-run` compile only: native CANN9.1.0/BiSheng builds the new generated
`as2_llm`, shared-code `as2_argmax` probe and pinned CATLASS49 reference.
The reference copy removes device resets and caps its request to2 groups;
it is an optional separate target, never linked into the LLM runtime.
Compiler warnings about inferred reference kernel types and duplicate inline
are retained. Linking success does not establish MIX/flag/runtime correctness.

All targets built serially with nice10/parallel1. Source/header manifests are
checked around builds. The final probe-only status drain has a separate rebuild
receipt. Artifacts: `runs/as2/builds/taskbody_argmax_v2/` and
`tilemega-as2/build/taskbody_argmax_v2/`. Source/build failures are retained under
`runs/as2/failures/taskbody_argmax_contract_link_v1/` (device constexpr linkage,
fixed with an explicit AICore-inline declaration).

`verified-run` CPU tests:41 Python tests (inventory3/request24/search3/price11)
and3 Host capability executables pass. No ACL initialization in Host tests.
The12-case numerical probe covers ties, BF16 collapse, signed zero, stride/tail,
both AIVs, repeated calls and nonfinite/index errors; it is **compiled, not run**.

Source/commit/blob/license identities: `configs/ascend/as2_taskbody_sources.json`.
CATLASS remains `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`.
FlagGems `ae9e355c121b3fffb1d229ba56a7c0ed8398fe39` and DeepGEMM-Ascend
`3a35326271b069b8b5bf1a754435acd6cc91575b` are source/algorithm references only;
their schedulers/code are not copied or installed. No CANN/framework upgrade.
FlashMLA's remaining pipeline advice is still proposal, not a local port.

## Device boundary and remaining work

`verified-run` read-only `npu-smi info`,09:26 UTC: physical0/D7:00.0 HealthOK,
util100%,85523/131072MiB HBM; physical1/55:00.0 Alarm,util100%,15425MiB.
External processes are observed only. The inherited physical0 Host600s timeout
has no final-stream/device-exit receipt; physical1 has an earlier unknown exit
plus Alarm. These holds remain. This is not an external-process idle gate.
**Zero new NPU jobs**, no locks/reset/exclusive mode or other-process actions.
Evidence: `inventory/taskbody_r0_v1/{npu_readonly.txt,safety.json}`.

| New-roadmap gate | State | Evidence/remaining requirement |
|---|---|---|
| G-SOURCE | pass for reviewed subset | pinned source/license + native compile |
| G-BODY | in_progress | real code/compile; numerical probe not run |
| G-HANDOFF | blocked | inherited exit/health; no new protocol claimed |
| G-PIPE | in_progress | real bootstrap; new empirical prices/search pending |
| G-CORRECT | blocked | C1/teacher512/C2/C3/full1024 not run for new Body |
| G-PERF | not_started | no new complete paired request samples |

Still required: model-validated vector argmax/SwiGLU; batched residual and
descriptor caching; CATLASS49 streaming Attention with effective-KV, explicit
precision/participants and scratch accounting; GEMM session lifetimes; credit
slots/retired-prefix audits; legal same-group affinity/region lowering; measured
start/II/drain/context costs and finite whole-model selection. New G/direct paths
need their own device admission. Frozen tolerances/greedy equality are unchanged.

After device exit/health is established safely, run the bounded shared-code
probe through `as2_run`, then actual-dimension layer → short full model →
teacher512/C2/C3/full1024. Only then price/search and paired complete requests
against the frozen same-card TileMega control. No framework-baseline rerun and
no TPOT improvement claim are warranted by this checkpoint.

Storage audit: this phase's sampled peak/final generated footprint is
106116857856B (98.83GiB), within the existing100GiB budget. The prior historical
peak108181118976B is retained separately. Manifest cleanup dry-run and apply
both find0 eligible files, remove0 and release0B; failure evidence and both
recent builds stay protected. Filesystem free space changes from
10173754073088B before the final build to10202181730304B at cleanup; this shared
filesystem change is not attributed to our zero-byte cleanup. Raw tensors,
generated sources, binaries and detailed logs remain ignored, outside Git.
