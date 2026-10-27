# Five dense models: first B1/B16 comparison, 2026-10-27

`verified-run`: **40/40 baseline cells and10/10 TileMega cells collected**.
Ten sourced priced TileMega plans pass complete64/1024 numerical requests.
All40 TileMega warmup/timed token payloads match the nominated numerical pass.
This completes the user-requested first snapshot, **not the remaining AS-2
optimization, final teacher/C2/C3 coverage or statistical G-PERF**. B2/4/8 and MoE
remain deferred; further optimization awaits user review.

BF16, seed20261016, actual full layers/vocabulary, ISL64/OSL1024, greedy,
ignore-EOS. One process/cell, one warmup + three full timed requests; process
medians below. TPOT=(last token ready−first token ready)/1023. No baseline
numerical comparisons. All original samples and technical failures are retained.

| Model | B | torch_npu eager ms | compile ms | vLLM ms | SGLang ms | TileMega ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1 | 7.405 | 2.925 | 2.954 | 8.930 | 6.716 |
| Qwen3-0.6B | 16 | 6.195 | 5.401 | 3.669 | 9.528 | 48.862 |
| Llama3.2-1B | 1 | 3.529 | 3.010 | 3.690 | 5.766 | 8.325 |
| Llama3.2-1B | 16 | 3.644 | 3.654 | 4.375 | 5.488 | 32.153 |
| Llama3.2-3B | 1 | 6.599 | 6.590 | 6.421 | 8.728 | 19.035 |
| Llama3.2-3B | 16 | 8.215 | 8.248 | 7.655 | 8.844 | 68.735 |
| Qwen3-1.7B | 1 | 6.227 | 4.532 | 3.613 | 9.528 | 12.782 |
| Qwen3-1.7B | 16 | 6.513 | 6.282 | 4.543 | 9.842 | 59.629 |
| Llama3.1-8B | 1 | 12.416 | 12.339 | 11.929 | 11.335 | 40.066 |
| Llama3.1-8B | 16 | 14.540 | 14.789 | 13.332 | 12.786 | 144.543 |

TileMega is descriptively 2.30–13.32× slower than the smallest of the four
observed baseline medians. This is an unpaired shared-machine comparison.
Idle admission uses two HealthOK/utilization0% observations15s apart and enough
HBM, without reserving hardware. Resident contexts conservatively mark all
current timings contended; later interference is possible. Llama8B B16 raw
TPOT samples are 144.543, 247.888, 110.868ms; none is dropped.
No paired confidence interval or reproducible hardware-speedup claim is made.

## Implementations, pipeline and fixes

- Eager: native FIA-v2, fused Norm, StaticCache. Compile: real NPUGraphEx;
  no timed recapture/eager fallback. Installed TorchAir GE failures remain.
- vLLM: native Ascend mode3/FULL_DECODE_ONLY with exact batch capture.
- SGLang: native FIA, graph-off. Current graph replay fails107000/SQ-CQ bind;
  page128/B×2048 physical pool and context1090 preserve all1024 outputs.
  Collector2 uses delta CPU-ready tokens, no explicit unused parent NPU context.
  Initial SetDevice507033, safety pause and successful bounded retry remain.
- TileMega: local config/random manifests → serving BuildModelPlan → semantic
  ModelDescription/CTG/ISL → actual implementation/decomposition → measured
  prices → structural/EFT placement/order comparison → wait/FIFO/MIX audit →
  ABI4 → generated Ascend C/CANN → one persistent launch/full-model forward.
  Each request has one64-token prefill +1023 decode forwards; no Host operators.
- Qwen0.6 B16: fixed missing empirical pricing; selected full1024 passes bitwise.
- Llama3B B16: actual CTG decode split2 + ordered device combine repairs down;
  selected full1024 tokens/final logits/KV are bitwise equal to the reference.
- Llama8B B16: separate Q/K/V, decode KV split7; prefill Q/O/down selectively
  use three local FP32 K intervals only in N3584:4096. BF16 RINT precedes
  residual Add. One AIC owner/AIV0, per-group scratch and existing ready/done/ack;
  no CPU combine or new synchronization primitive. Q-only and uniform-seven
  failures remain. Four-token intermediates and selected priced full1024 pass
  bitwise:16384 tokens,2052096 final logits and64 last-position KV tensors.
- Llama8B actual plans:804874 prefill/21018 decode tasks,23,928,201,216-byte
  request arena. Both phases select EFT from two real legal queues. Predicted
  EFT/structural: prefill3.937757/3.939860s; decode109.960/111.038ms. These are
  sourced context-local estimates, not hardware peaks. Source-identical request
  binary is reused after relocation/pricing, with a strict generated-source check.
- Fast mode omits diagnostic tensors/traces. `as2_queue audit` independently
  checks all40 timed/warmup output hashes against prior full1024 numerics;
  no new NPU work or baseline output comparison occurs.

## Identity, versions and resource boundaries

- Original AS-2 Prompt SHA256:
  `673b6deb619cb02f1c3f2d0875af2bdafb835bca09e3875d7dbfb840b1be00d6`.
- Latest exact user scope SHA256:
  `006a101838ca5b04bd780201d36a67be2f4ea3df34b43f4eedb062473b2ca1a8`.
- AS-1 baseline `2bb2194d3ab6e093a9d8fbc4dce05f29a389271c`;
  this continuation starts `e2184c93e523e1721ffaf41e57e90b2ad44c9dc9`.
- Code HEAD before this report commit: `432e1cfd2d374671f6a031fe6c3f8cf75ee6449f`.
  Final local/remote HEAD and commit list are recorded after push in
  `runs/as2/bench/focus_b1_b16_v1/delivery.json` (a report cannot contain its own commit hash).
- Branch `tilemega-ascend`; author Junhui Peng <pengjh0111@hnu.edu.cn>.
- Two950PR_9579 devices,28 AIC/56 AIV each; physical0=D7:00.0,1=55:00.0.
  One visible card/logical0 per job. Selected G8 means8 launch blocks/24 workers;
  AIC(g)=g, AIV(g,l)=8+2g+l. Both AIV lanes run independent tasks; MIX usesAIV0.
  G14/28 and BF16 direct UB remain disabled, without claiming hardware absence.
- CANN9.1.0, driver25.7.rc1, BiSheng15.0.5/dav-3510;
  CATLASS `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`;
  reused LLVM/MLIR23 `23a60f15f2fcafcf67b95b0a035053579958b732`.
- TileMega/eager: Torch2.7.1+cpu, torch_npu2.7.1.post7.dev20260707,
  Transformers4.51.3. Private vLLM: Torch2.10/NPU2.10.0.post4, Transformers5.5.4;
  SGLang: Torch2.10/NPU2.10.0.post6, Transformers5.19.0.
  Framework versions differ and remain explicit comparison metadata.
- vLLM core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`;
  plugin `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`;
  SGLang `d7f832d30ac3d88cf111ba242e3217dbf6ee6212`.
- Environments, generated files/cache/weights/logs stay under the private
  project subtrees in `/home/d50064638/junhuip`; AS1 dependencies are reused.
  No system/account configuration changes, NPU locks/exclusive mode/reset,
  CUDA regression, official weights/tokenizer download or other-process action.
  Engine world-size1 distributed bookkeeping is disclosed; no multi-device
  communication is configured. CPU builds use nice10/parallel1, one target at a time.
- This queue records95 attempts (including failures and superseded collectors),
  56 on physical0/39 on1; wrapper wall sum7829.420s, not device-busy time.
  94 complete recorded intervals have no overlap; one interrupted supervisor
  receipt remains incomplete/excluded. Two timeouts, five failed attempts and
  three interruptions/exclusions are preserved. No active own NPU jobs remain.

## Raw records, storage and remaining AS-2 work

All raw paths below are under ignored `runs/as2`; no tensors/binaries enter Git.
- `bench/focus_b1_b16_v1/comparison.{json,csv,md}`: all50 TPOT/TTFT/E2E/
  throughput values, ratios, devices and interference. `state.json`/`jobs/focus_*`
  retain every attempt, command, PID/start/PGID, exit, before/after snapshots.
- `selected_tilemega_plans_v1.json` associates each CTG/price/Plan/source/
  cached binary/launch/numerical/timing identity. `resource_summary.json` audits
  submissions; `correctness/focus_timed_token_identity_v1.json` audits40 outputs.
- `frozen_baselines_v1.json` retains all40 baseline samples/identities, SHA256
  `4f1fa3729e3f74ce5c48e72257009447bee39e453dec93735d3da851a55b37ab`.
  Final independent SHA audit confirms frozen sources/samples are unchanged.
- Local Llama8B solve logs imply approximately309.9s prefill/79.1s decode
  (creation→last-write resolution±1s); these are offline costs, excluded from
  TPOT. Build and initialization records remain in private logs/result identities.
- Safe manifest cleanup unlinked21 original files totaling6,169,619,868B after
  lossless gzip/decompressed-SHA verification; compressed evidence is retained.
  Net savings are smaller. Latest generated usage106,534,785,024B versus
  107,374,182,400B budget; historical peak108,181,118,976B triggered controlled
  capacity handling. Private environments use7,352,451,072B, separately budgeted.
  Final disk samples and zero-candidate cleanup are indexed in `delivery.json`.
- Remaining: teacher512/final configuration C2/C3 coverage for new queues;
  measured small-M/GEMV/Attention/fusion and broader Body/geometry/G search;
  BF16 direct evaluation, performance ablations and original paired statistical
  acceptance. No further NPU tests are queued; next changes await user review.
  MoE and B2/4/8 are explicitly deferred.

## Focused code commits before the report

- `de840dd17 experiments: Queue idle single-card baseline comparisons`
- `1b10877a6 runtime: Keep full request timing without tensor diagnostics`
- `9436f4107 backend: Register tested Llama3B B16 decode split partials`
- `2cbe1789b runtime: Recheck device health before owned child launch`
- `74882b577 experiments: Retain lossless profiles and schedule archives`
- `06de65b4e codegen: Bound schedule reports without dropping native tasks`
- `33fd27bbe experiments: Configure native graph and long request baselines`
- `7a135120d experiments: Gate row timing on real baseline and plan results`
- `103b397f2 backend: Add an experimental B16 separate projection candidate`
- `cb406c918 experiments: Diagnose the observed Llama8B B16 Q tail`
- `4257a33bd backend: Add an opt-in MIX Q tail reduction candidate`
- `bf12c6864 codegen: Compose registered request Body capabilities`
- `f54543915 runtime: Accept registered local Q geometry in diagnostics`
- `cca2180e8 experiments: Verify observed O and down projection tails`
- `67aeb72b9 backend: Preserve ordered residual projection tail rounding`
- `34bfb964d test: Isolate CPU timeout checks from live NPU submissions`
- `432e1cfd2 experiments: Audit timed TileMega output identities`
