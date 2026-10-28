# AS-2 TPOT follow-up — 2026-10-28

Status: **in_progress**. Five dense models/B1/B16 remain the main scope;
MoE and B2/4/8 remain deferred. Baselines are frozen and were not rerun.
Start HEAD `237439ff59eeb0c1802b43e85d358feac70d9b1d`, branch `tilemega-ascend`.
AS-2 Prompt SHA256:
`673b6deb619cb02f1c3f2d0875af2bdafb835bca09e3875d7dbfb840b1be00d6`.
Latest user instruction SHA256:
`e18a645d39e9cdb9ed76f89f9d577aada9b4b8145fb9e9ccc3af02e203838631`.

## Pipeline and optimization boundary

`verified-code/run`: local random config/weights → serving BuildModelPlan →
ModelDescription/CTG/ISL → actual implementation/decomposition → measured
positive native prices → structural/EFT placement/order comparison →
dependency/FIFO/MIX audit/frontier lowering → ABI4/materialization → generated
Ascend C/CANN → one persistent launch per full-model forward → greedy/KV.
The ten first-snapshot cells already have full64/1024 numerical passes.
Bounded two-Body outer search now executes native CTG/ISL solves. Full
multi-family Body/geometry/decomposition/G search, five-model final C2/C3
coverage and original AS-2 performance Gates remain incomplete. The semantic
forward chain is real; it is the optimization/acceptance chain that is partial.

Existing Qwen0.6 B16 profile: per-AIC Body span≈42ms in≈55ms queue windows
at past1086; cooperative Attention owner service is prominent. These are
software spans, not hardware utilization; nested AIV intervals are not added.
Evidence: `inventory/optimization_review_20261028_v1.json` under `runs/as2`.

## FIFO wait elimination: no reliable end-to-end gain

Same binary, input, Body, placement, queues and G8; only already-observed
producer-frontier waits change. Original prior numerical/C3 evidence remains.
ABBA pilot: two speedups1.01711/1.01797; all16 request token payloads pass.
This pilot was excluded from the independent final sample.

Before final sampling, freeze five fresh process pairs, alternating order,
one warmup/three full timed requests per process. All40 token payloads pass;
the same retained image also passes a fresh full1024 logits/KV numerical run.

| Qwen0.6 B1/G8 | Original | FIFO |
|---|---:|---:|
| Median process TPOT, ms | 6.712559 | 6.707102 |
| Median process TTFT, ms | 51.859726 | 49.944604 |
| Median process E2E, ms | 6919.406139 | 6911.287864 |

Paired geometric speedup **0.99530**, approximate paired-log-t95% interval
**[0.96135,1.03045]**. No slow samples were removed. Process CV2.84%/4.07%;
resident/shared load is retained. **No reliable gain; original stays default.**
This compares two TileMega plans, not eager, and does not pass original G-PERF.
Evidence: `bench/opt_fifo_qwen_b1_final_v1/{protocol,ablation}.json`.
`as2_ablation.py` rechecks process identity, nonoverlap, protocol freeze,
binary/input/queue/ownership identity, raw request samples and every token hash.

## Valid-key QK: actual Body and priced Plan candidate

`verified-code`: `LlmCubeBody.h::AttentionGemm` variant8 skips entirely masked
QK64 tiles. It retains physical stride1088 and the original final physical
tile, L0K32, FP32 arithmetic, scale, norm, fold softmax and full-capacity PV.
`ServingSoftmax` DMA reads valid scores only and creates its masked tail in UB.
No uninitialized omitted scores enter computation. GM visibility and single
owner/AIV0 ready/done/reverse-ack are unchanged; no Host operation is inserted.
Code-derived QK calls per task/request:17391→9711, **44.16% fewer calls**;
this count is not a measured latency improvement.

`verified-run`: all229 packed input buffers match the canonical hashes.
Qwen0.6 B16 four-output, full1024 diagnostic and **repriced full1024** runs
pass bitwise16384 tokens,2430976 final logits and56 checked KV tensors.
Fresh G8 control calibration:5/5 independent processes, physical1/55:00.0,
logical0; source-cost records remain context-local/contended, not hardware peaks.

Actual CTG/ISL/Solver calls produce245955 prefill/8403 decode tasks;
arena4565844480B. Geometry16x128x128, G8,24 workers. Same-role/static ownership
and original public task/dependency meanings are preserved.

| Phase | Structural prediction, ms | EFT prediction, ms | Selected |
|---|---:|---:|---|
| Prefill | 756.621523 | 759.583630 | Structural |
| Decode at calibrated upper context | 52.172573 | 52.017612 | EFT |

These are predictions, not TPOT. Solver wall199.75s/66.23s, with actual
`time -v` RSS logs; compilation/packing/initialization are outside TPOT.
Current selected queues/waits/ownership are byte-identical to the old B16
control; only decode Attention variant6→8 changes. Both use the same binary.
Generated request source matches the profiled image source exactly.

Earlier same-image full-request pilot50.066→46.711ms (6.70% reduction)
and all8 token payloads pass. It is excluded from final independent pairs.
The frozen eager/compile/vLLM/SGLang B16 medians remain
6.195/5.401/3.669/9.528ms; TileMega is still slower than all four snapshots.

`verified-run`: five predeclared alternating fresh process pairs completed;
warmup1/timed3 full requests per process, same physical1/image. All40 token
payloads and229 canonical inputs pass identity checks. Final process medians:

| Qwen0.6 B16/G8 | Original | Valid QK64 |
|---|---:|---:|
| TPOT, ms | 50.682721 | 46.248476 |
| TTFT, ms | 817.610325 | 819.859949 |
| E2E, ms | 53089.518118 | 48132.489691 |
| Output throughput, tokens/s | 308.610825 | 340.393778 |

Paired geometric speedup **1.100785**, log-t95% interval
**[1.071267,1.131116]**, process CV2.45%/2.49%; no samples removed.
The median TPOT reduction is8.75%; speedup is not a latency percentage.
**Pass versus TileMega control only**, with resident/shared load retained;
this does not pass original eager G-PERF. Protocol/ablation/token evidence:
`bench/opt_validqk_b16_final_v1/{protocol,ablation}.json`,
`correctness/opt_validqk_final_paired_inputs_v1.json`.
The exact measured image also passes teacher512, full1024 fast/diagnostic
C2 (92 common payloads) and C3 **5/5 fresh processes/10 reset requests**,
67 outputs crossing past127/128/129. Evidence:
`correctness/opt_validqk_b16_{teacher512_numeric,c2_full1024,c3}_v1.json`.
The repriced numerical run took560.41s under simultaneous load, with runtime
queue progress and no abort. Its wall time is not a performance comparison.
Numerics: `correctness/opt_validqk_b16_priced1024_numeric_v3.json`.
Prices/selected plans: `search/opt_validqk_*v3.json`,
`models/qwen3_0_6b/opt_validqk_b16_{prefill,decode,request}_priced_v3/`.

## Subphase attribution and bounded outer search

`verified-run CPU/CANN`: opt-in `AS2_ATTENTION_PHASE_PROFILE=ON` records
KV RoPE/append, valid-KV cast, tail zero, query RoPE/cast, QK, softmax, PV and
output cast. A fixed mask separates the AIC owner and AIV participant. The
parser rejects missing/wrong-role phases, out-of-Body timestamps and phase
reordering. The original service/pricing fields are unchanged; re-reading the
old archived decode profile produces an identical summary. Eleven price/profile
CPU tests pass; the native instrumented image builds with parallel1/nice10.
Production builds default to no subphase timers. These diagnostics exclude
interphase acquire/coordination and cannot be added as hardware utilization or
used in place of full-request TPOT.

The optional `AS2_DECODE_PROFILE_ONLY=ON` omits the unused prefill profile
download, explicitly recording `body_profile_capture=decode_only`. It executes
the complete prefill and decode; it preserves output/guard/numerical checks.
`verified-run`: early past66 and late past1086 profiles pass independent
numeric checks, all28 layers/B16. At past66, mean KV-cast/tail-zero service
is4.078/15.381us; at past1086,49.702/0.360us. Their share of the **sum of
measured phase service**, not request critical-path time, is51.64%/59.24%.
Each profile contains7168 owner/participant records; AIC and AIV intervals
are not treated as a globally ordered timer. This identifies repeated KV
conversion/zeroing as a concrete next target, rather than an assumed share
of TPOT. Evidence: `search/opt_attention_phases_*_profile_v1.json`.

`verified-code/CPU`: `as2_search.py` admits at most four declared measured
Body/geometry/decomposition/G candidates. Each invokes the native compiler's
actual Import/CTG/ISL/priced placement/order/materialization path, serially at
nice10, with disk checks, bounded CPU process groups and retained RSS/logs.
Selection returns the solved Plan with the smallest positive predicted forward
makespan; it never promotes a runtime default or claims TPOT admission.
Same-G alternatives must share a control calibration and all alternatives
must share physical card and ABI/past context. Duplicate candidates and altered
calibration hashes fail before solving. Two CPU tests cover these guards and
cost changes selecting a different returned Plan; fixture costs stay in tests.
`verified-run CPU`: two real QK Bodies are imported, solved and materialized;
the outer search selects valid-QK, predicted51.585843→51.544590ms at past1086.
Actual solve wall55.58/54.84s, peak RSS182348/182208KiB. The selected eight
ABI tables and generated source are byte-identical to the device-tested Plan.
This upper-context price underpredicts the whole-request improvement because
both Bodies execute17 QK tiles there; past-bucket pricing remains open.
Evidence: `search/opt_attention_outer_search_v1/search.json`,
`search/opt_attention_selected_abi_equivalence_v1.json`.
Full multi-family hierarchical optimization remains open.
Evidence/commands: `search/opt_attention_profile_followup_v1.json`,
`builds/opt_attention_phases_v2/identity.json`; the follow-up has a finite budget,
performs no NPU operation and changes neither defaults nor Git.

## Request-lived derived FP32 KV: new experimental Body9

`verified-code`: `cube-fp32-cached-kv`, initially Qwen0.6 B16 only, preserves
the authoritative BF16 K/V and all original rounding, QK/PV geometry/order,
softmax and visibility/owner/ready/done/reverse-ack contracts. Each layer gets
two derived FP32 planes. Prefill writes disjoint owned rows; decode appends
only the new row and reuses earlier conversion. All112 public/derived planes
reset before each request. No old request data or precomputed answer is reused.
The additional logical storage is3992977408B; guarded allocation3993006080B,
arena4565844480→8558850560B. Both phases and runtime enforce this lifecycle;
standalone invocation, wrong shape and mixed cached/uncached phases fail.

`verified-run`: unpriced bootstrap and actual repriced CTG/Solver request
both pass full64/1024 bitwise tokens/full final logits/56 KV outputs against
the cached independent reference. Source and image identity are preserved;
each forward is one persistent launch. Priced phases have245955/8403 tasks,
same logical graph/FIFO as Body8. Native solve wall195.67/53.67s.
Teacher512 and full1024 fast/diagnostic C2 pass bitwise; C3 **5/5 fresh
processes/10 reset requests**,92 payloads/request, with no timeout/OOM.
Evidence: `correctness/opt_cachedkv_b16_priced1024_numeric_v3.json`,
`correctness/opt_cachedkv_b16_{teacher512_numeric,c2_full1024,c3}_v1.json`.

Same-image positive measured prices drive two further native outer candidates:
Body8 predicted52.379118ms, Body9 **33.484880ms**, selected Body9. Actual CPU
solve54.36/54.37s, peak RSS178364/182380KiB; selected execution tables/source
equal the independently device-validated priced Plan. These are predictions,
not TPOT. `search/opt_cachedkv_outer_search_v1/search.json` retains both plans.
Full-request warmup1/timed3 pilot, same image/card/inputs:

| Qwen0.6 B16/G8 | Body8 | Cached KV Body9 |
|---|---:|---:|
| TPOT process median, ms | 49.907263 | 36.174659 |
| TTFT process median, ms | 817.843685 | 871.600396 |
| E2E process median, ms | 52132.492145 | 37825.895031 |
| Output throughput, tokens/s | 314.276171 | 433.142428 |

All8 token payloads pass. Observed TPOT reduction27.52%, speedup1.37962;
**pilot only**. Body9 warmup TPOT210.36ms under shared load is retained;
timed37.020/34.968/36.175ms. No slow sample is removed. Five independent
alternating pairs are frozen in `bench/opt_cachedkv_b16_final_v1/protocol.json`,
SHA256`da8708b2b3e4d8670c61fc0d4b1376db87ececbcb658f25f67f84921f6f43fb7`.
One finite serial submitter waits for idle observations; a finite CPU-only
auditor verifies correctness/input hashes/timed outputs and paired statistics.
No Body9 default or final performance admission yet. Evidence:
`bench/opt_cachedkv_b16_pilot_v1/pilot.json` and final queue state/receipts.

Large diagnostic profiles now optionally stream D2H into exclusive-new gzip
files, with SHA256, decompression/extent verification and an archive receipt;
the raw profile never touches disk. Uses existing zlib/OpenSSL, no installs.
A CPU transport fixture independently parses the archive, and both actual
prefill/decode profiles pass the existing parser. Fast math/sync are unchanged;
post-request compression is outside the timing interval. Source:
`Target/Ascend/BodyProfileCapture.h`, `AS2_COMPRESS_BODY_PROFILES` (defaultOFF).

## Rejected/confounded attempts and safety

- Wide QK variant7 remains gated/off. Initial packing omitted the verified
  NPU RoPE table and produced CPU coefficients; both initial candidate short
  comparisons were confounded. These failures do not prove Body arithmetic
  failure. `validate_rope_reuse` now rejects substitution before any files are
  generated. Corrected229-buffer identity and bitwise tests are retained.
- Its long diagnostic timed out600s; stream heads269→523→762 show progress,
  not established deadlock. Only its own process group was ended.
- G14 payload probe timed out120s; no G14/28 capability is enabled. Subsequent
  known G8 same-binary payload sanity passed5.27s. No failed G14 retry was run.
- Two950PR,28AIC/56AIV per card; CANN9.1.0/BiSheng15.0.5, pinned AS1 CATLASS
  `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`. Both AIV lanes run independent
  tasks; current cooperative Attention still uses AIV0. Direct BF16 remains off.
- One own NPU job at a time, one visible physical card/logical0 per process;
  no device locks/exclusive mode/reset, no other-process action, no shared
  environment edits, no CUDA regression or official weight downloads.

All raw data are under ignored `runs/as2`; private builds/caches remain in
`tilemega-as2`. Lossless SHA-verified archiving preceded removal of one old
schedule243741849B and two profile payloads200259584B. Archives/failures,
selected images and at least two complete builds are retained; AS1 is protected.
The100GiB generated-data budget was not raised. A new diagnostic batch crossed
it by5.4MB; preflight stopped new work. Two older completed solver schedules
(243741769B/156337290B) were losslessly archived/SHA-verified before manifest
dry-run/apply removal; C3 sharing preserves288 paths and reclaims62300160B.
AS1, failures, selected images and at least two complete builds are protected.
Checkpoint usage/cleanup/HEAD/
push and exact job inventory are recorded in
`runs/as2/bench/optimization_20261028_checkpoint.json`.
