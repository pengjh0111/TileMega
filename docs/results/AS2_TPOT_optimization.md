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
Full hierarchical outer Body/geometry/decomposition/G search, final C2/C3
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

Fast full64/1024 pilot, same physical1/binary, one warmup/three timed requests:

| Qwen0.6 B16/G8 | Original | Valid QK64 |
|---|---:|---:|
| TPOT process median, ms | 50.066034 | 46.710510 |
| TTFT process median, ms | 813.847131 | 821.586469 |
| E2E process median, ms | 52031.527755 | 48606.438032 |
| Output throughput, tokens/s | 314.886007 | 337.074689 |

**Observed TPOT reduction6.70%, speedup1.07184.** All8 warmup/timed token
payloads match their full numerical proof. Raw TPOT samples: original
50.066/52.467/48.348ms; candidate49.242/46.711/45.766ms. Shared load is
recorded; this sequential pilot is not a stable-speedup/default/G-PERF claim.
The frozen eager/compile/vLLM/SGLang B16 medians remain
6.195/5.401/3.669/9.528ms; TileMega is still slower than all four snapshots.

Five independent alternating process pairs were frozen before new sampling
in `bench/opt_validqk_b16_final_v1/protocol.json`. A finite90-minute no-lock
submitter waits for idle observations, then serially runs at most one own
single-card job, each≤900s. A finite CPU follow-up audits all40 token payloads,
source/input/queue/ownership identities and paired statistics; it does not
change defaults or Git. Raw state/summary/followup and own PID/start/PGID
receipts stay in that directory. Final paired validation is pending.
The repriced numerical run took560.41s under simultaneous load, with runtime
queue progress and no abort. Its wall time is not a performance comparison.
Numerics: `correctness/opt_validqk_b16_priced1024_numeric_v3.json`.
Prices/selected plans: `search/opt_validqk_*v3.json`,
`models/qwen3_0_6b/opt_validqk_b16_{prefill,decode,request}_priced_v3/`.

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
The100GiB generated-data budget was not raised. Checkpoint generated usage
106911100928B, warning=false, zero remaining cleanup candidates; environments
and AS1 dependencies remain protected. Checkpoint usage/cleanup/HEAD/
push and exact job inventory are recorded in
`runs/as2/bench/optimization_20261028_checkpoint.json`.
