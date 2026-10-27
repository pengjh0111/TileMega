# AS-2 baseline collection — frozen snapshot, 2026-10-27

The user added torch_npu compile, vLLM and SGLang comparisons, then explicitly
waived baseline numerical comparisons and requested one idle-only performance
dataset for reuse during TileMega optimization. The latest request narrows this
first comparison to **B1/B16**: five dense models × two batches × four systems
= **40 baseline cells**, plus ten TileMega cells. B2/4/8 and MoE are deferred.
**All40 baseline cells are frozen; original statistical G-PERF is not passed.**
Extension text is outside Git at `tilemega-as2/Prompt/AS2_baseline_extension_20261026.md`,
SHA256 `2b27dbb3b4d518e81837bf2bcb5f469019d4f4d0b0cc11a239ee7699ec94b5d7`.
The no-comparison/idle-only override SHA256 is
`a527203268aa0e819ffce2af77c01e3ed1cb980580aa60670a132859272cda26`;
the explicit batch correction SHA256 is
`aaddc0550559cae3ddcf88157200f2cfb36e221905db60d8e422d626c29ed7b4`.
Exact texts are preserved separately under the same private `Prompt/` directory.

## Collection contract

`configs/ascend/as2_focus_compare.json` fixes the active five models, two batch
sizes and eager/compile/vLLM/SGLang. Each cell uses BF16, ISL64/OSL1024,
greedy/ignore-EOS, one new process, one warmup and three complete timed requests.
Only collect when plain `npu-smi info` establishes idle state; unavailable
telemetry is not evidence of idleness. Defer busy cards without locks or process
intervention. Idle observations cannot reserve hardware against other users;
observed interference makes a sample contended, not an admitted idle snapshot.
The active idle policy is two snapshots15 seconds apart, utilization0%, healthy
devices and sufficient free HBM. Idle resident contexts are observed rather
than evicted; their presence conservatively marks timing contended.
Collection remains serial and bounded, not a bulk compile or stress sweep.

The exact latest user text is `tilemega-as2/Prompt/user_focus_20261027_zh.md`,
SHA256 `006a101838ca5b04bd780201d36a67be2f4ea3df34b43f4eedb062473b2ca1a8`.
The sole submitter is `as2_queue run --directory runs/as2/bench/focus_b1_b16_v1`.
Its comparison CSV/JSON retains TPOT/TTFT/E2E/throughput, physical device,
all failed attempts and interference. Current results: `AS2_focus_comparison.md`.

`verified-run`: compile uses NPUGraphEx; vLLM uses native mode3 decode graphs.
SGLang uses native FIA without graphs: this installed graph replay failed107000
with SQ/CQ binding diagnostics. Its page128 pool is B×2048; logical KV capacity
is1088 and scheduler context1090 preserves all1024 outputs. Collector2 consumes
CPU-ready delta tokens without an unused explicit parent NPU context. A child
SetDevice507033 failure remains recorded; the bounded collector2 retry succeeds.

Keep configuration/weight/input identity and requested output-count checks;
do not compare baseline tokens, logits, KV or teacher-forced traces. Historical
diagnostics are retained as provenance and are no longer collection blockers.
TileMega's own model/Body/pipeline correctness checks remain required.
Freeze the completed performance dataset and reuse it rather than rerunning
engines after each TileMega change. Such later comparisons are snapshot ratios,
not contemporaneous paired-process confidence intervals; do not claim the
original Prompt's statistical performance Gate from this dataset alone.

## Pipeline boundary

TileMega: fixed model/random identity → serving export/BuildModelPlan → semantic
lifting/ModelDescription → CTG/ISL → explicit Body/decomposition → empirical
prices → heterogeneous structural/EFT queue comparison → dependency/FIFO/MIX
audits/frontiers → ABI4 → generated Ascend C/CANN → ACL persistent forward.
One full-model forward is one launch;64-token prefill produces output1, then
1023 decode forwards produce outputs2..1024. Outer Body/geometry/G search is
still incomplete; a fixed-structure EFT choice does not complete G-PIPELINE.

Independent baselines consume the same canonical BF16 manifests/config/input.
Engine loaders stream/hash each tensor and call native engine weight loading
and packing; no TileMega TaskBody, model mathematics or executor is reused.
Tied embedding aliases are recorded. No official checkpoint/tokenizer download.
Full layers/vocabulary/KV/greedy and the requested output count remain required;
short pilots do not replace a complete1024-token performance request.

## Private software and reproducibility

All paths below are under `/home/d50064638/junhuip/tilemega-as2`; shared Python,
CANN,driver,rc files and package configuration are unchanged. CANN9.1.0 and
the current25.7.rc1 driver remain installed. No Docker/system installer ran.

| Environment | Actual selected packages / source |
|---|---|
| Existing `envs/as2` | torch2.7.1+cpu, torch_npu2.7.1.post7.dev20260707, transformers4.51.3; only private protobuf4.25.8 added |
| `envs/vllm-ascend` | Python3.12.3, torch2.10.0+cpu, torch_npu2.10.0.post4, transformers5.5.4, vllm0.23.0+empty, plugin0.23.0/A5 |
| `envs/sglang-npu` | Python3.12.3, torch2.10.0+cpu, torch_npu2.10.0.post6, transformers5.19.0; SGLang pinned development source, kernel2026.9.0 |

`scripts/as2_baseline_env.sh` selects one private engine and directs verified
cache/vendor/header settings locally. Dependency lockfiles/install logs are retained
under private `scripts/` and `logs/`; detailed receipts are `runs/as2/inventory/`.

- vLLM core0.23.0 source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`;
  native targets excluded with its `VLLM_TARGET_DEVICE=empty` build path.
  Core build isolation would request a different Torch; the consumer build uses
  the already matched CPU Torch and `--no-build-isolation --no-deps`.
- vLLM Ascend0.23.0 source `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`;
  official950 wheel SHA256 `b4798453e985f2d833d3be50c064c90e6dc04cbae797845e9d9f0656093a3d45`,
  `_build_info.__device_type__=A5`. Filename compatibility alias is a hardlink;
  wheel contents/RECORD are unchanged.
- Stock text dependency resolution conflicts: Triton3.2.2 requires NumPy1.26.4,
  core OpenCV≥4.13 requires NumPy2. Video support is explicitly excluded;
  core already handles absent OpenCV with PlaceholderModule. Optional Arctic
  speculative/suffix plugin is also excluded: its source build requests
  Torch2.7.0 and a separate extension. Speculation/video are not tested/enabled.
  Model/kernel source is unchanged; this is an explicit text-only dependency
  profile, not a claim that all stock extras are installed.
- SGLang source `d7f832d30ac3d88cf111ba242e3217dbf6ee6212`; uses its NPU
  pyproject profile, preserving the original GPU profile. No diffusion extras.
  Official2026.9.0.post10 Torch2.10/CANN9.1/950/x86 kernel archives are pinned;
  `.run` payloads were inspected and extracted locally without executing installers.
  `ASCEND_CUSTOM_OPP_PATH` points only to private extracted vendor directories.
- Missing NNAL is supplemented privately from official CANN9.1.0 NNAL,
  outer SHA256 `1a743a0957a77a0624c2ecfdf689b5c5937a9097fda1e3b3e8e0191a7d11c694`.
  Inner ATB9.1.0.B150 payload checksum is verified; only ABI1 runtime files
  (347223158B), headers/config and licenses are extracted. No installer ran.
  Private dynamic-link + actual KV-write probe passes bitwise at slots0/129
  on physical0, `probes/sglang_private_atb_cache_p0_v3`. ATB attention reports
  "950 only supports PAEncoder"; pinned native `ASCEND_USE_FIA=1` subsequently
  runs a four-token model request successfully. No system ATB change is made.
- Triton community3.2.0 is installed before Ascend3.2.2 as required by the
  official overlay packaging; no NVIDIA compilation or regression is run.
- Missing system Python.h was resolved by extracting only headers from the
  matching Ubuntu `libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb` into private
  `third_party/`; no package installation. SHA256
  `c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16`.

## Historical setup probes, superseded by the frozen dataset

Model runs below used physical1/PCI55:00.0→logical0, one visible950PR.
The private ATB follow-up probe selects physical0/PCID7:00.0 when physical1
has only1607MiB free versus physical0's125437MiB. Submissions are serial;
other load is observed only. No device lock, reset or other-process action.

| Path | verified-run historical evidence | Status at that earlier checkpoint |
|---|---|---|
| Both new Torch/NPU stacks | `probes/{vllm,sglang}_framework210_v1`: BF16 GEMM passes | Performance dataset pending |
| torch_npu NPUGraphEx compile | Qwen0.6 B1 full1024 completes; two captured graphs, no timed recapture | Idle warmup+three requests pending |
| vLLM/950 plugin | Canonical310 tensors consumed/226 packed parameter names; full1024 completes | Bounded optimized configuration and idle dataset pending |
| SGLang/private operators | Private imports/KV probe and native FIA four-token request pass | Full1024 count guard fix and idle dataset pending |

Compile implementation is `as2_compile.py` + `as2_eager.py --execution compile
--compile-backend npugraph-ex`. Preparation captures two real FX/NPU graphs;
no timed recapture or eager fallback. Three CPU audit tests pass. Installed
TorchAir GE separately fails on aten.index_copy; a Scatter adapter reaches GE
but fails real BF16 cache tiling. Those failures remain, not hidden as compile pass.

`as2_vllm.py` uses native LLM/uni, TP=PP=DP1, fixed cache budget, disabled
prefix reuse/tokenizer/speculation. Eager mode is a pilot, not a frozen fastest
engine choice. The1024 run is cold/unpaired/contended and changes token path;
its raw13.664693ms TPOT is **not an idle baseline measurement**. Historical
output/logprob comparisons are retained but no longer pursued or used as
collection gates. Single-rank engine
distributed bookkeeping is logged; no multi-device communication is configured.

`as2_sglang.py`/`as2_sglang_loader.py` implement the native engine adapter.
The initial missing `libatb.so` is resolved privately. Native FIA executes;
the pinned scheduler clips generation to `max_req_len-input_len-1`. The
harness initially requested context1089, which remained insufficient. Current
context1090 and B×2048 physical pool are verified by all ten full1024 cells;
both scheduler guard positions are preserved without truncation. No new
baseline numerical comparison was needed. A prior query-only timeout affected plain
`npu-smi info`; subsequent telemetry recovered. No system/runtime change.

The completed40-cell snapshot is `bench/focus_b1_b16_v1/frozen_baselines_v1.json`
under ignored `runs/as2`, SHA256
`4f1fa3729e3f74ce5c48e72257009447bee39e453dec93735d3da851a55b37ab`.
It retains identities, original samples, selected successful technical retries,
physical devices and contention. All failed attempts remain in queue history.
Further TileMega changes reuse this snapshot; B2/4/8 remain deferred.
