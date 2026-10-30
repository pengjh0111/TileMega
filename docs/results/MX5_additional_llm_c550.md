# MX5 additional Mistral and OLMoE workload campaign

Historical collection below: OLMoE was unsupported at that checkpoint. Native support and new fullgraph decode baselines are now measured in [the follow-up report](MX5_olmoe_native_c550.md); the older rows remain unchanged.

Evidence: verified preparation and completed rows; pending work is explicit.
This expansion and new external-baseline measurements were requested by the
user. The original MX4 CSVs remain unchanged. Prompt SHA256:
`75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Campaign start HEAD622c505e6c3d07315f5b284ac4f4a21881534ef9.

Both models retain their full official architecture and layer count. As in
MX5, checkpoints use random BF16 weights, seed20260925, FP32 normal samples
scaled by initializer_range then BF16 rounding, and unit norm vectors.
This is synthetic-weight performance, not a pretrained quality evaluation.
The same16 frozen public-domain passages are tokenized with each model's
pinned tokenizer and truncated to64 tokens. All systems share those IDs
and exactly the same checkpoint within each model.

| Workload | Official revision | Layers | BF16 tensor bytes |
|---|---|---:|---:|
| Mistral-7B-v0.3 | caa1feb0e54d415e2df31207e5f4e273e33509b1 |32|14496047104|
| OLMoE-1B-7B-0924 |6d84c48581ece794365f2b8e9cfb043c68ade9c5|16|13838323712|

Sources: [Mistral pinned config](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/caa1feb0e54d415e2df31207e5f4e273e33509b1/config.json),
[OLMoE pinned config](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/config.json).
Checkpoint identities retain every shard hash and the exact initialization.
Prompt hashes: Mistral40a1ff3b6478fdb9f07b46c2aa61e3b0abc711e84fc3a7c8a513c921e12b3529;
OLMoE99c6d2e67b4de8c4e2d5e54aca87caf8a834816805099c41c3ea83681301bcc3.

B=1/2/4/8/16, ISL64/OSL1024, BF16 greedy, fresh-process warmup1/timed2,
2rounds and median follow MX5. All60 baseline processes now complete:
2models x3systems x5batches x2rounds, each guarded with rc0. Physical5 is
Mistral;4 is OLMoE. OLMoE torch.compile explicitly uses eager experts and
the separately labelled compile-boundary repair. A baseline failure is retained, never replaced with
another system's result. New data are append-only in
[MX5_additional_llm_c550.csv](MX5_additional_llm_c550.csv).
Single-round rows are provisional; only median rows combine both fresh rounds.

Completed two-round TPOT medians (ms), verified:

| Model | B | TileMega | vLLM | SGLang | torch.compile | Fastest baseline / TileMega |
|---|---:|---:|---:|---:|---:|---:|
|mistral_7b_v03|1|10.080156|11.148076|10.606020|18.437016|1.0522x|
|mistral_7b_v03|2|10.473901|12.525795|12.029260|25.129721|1.1485x|
|mistral_7b_v03|4|10.926902|12.676094|12.184568|35.886293|1.1151x|
|mistral_7b_v03|8|11.376112|12.986500|12.631397|57.645759|1.1103x|
|mistral_7b_v03|16|12.664796|13.668050|13.536814|100.390065|1.0689x|
|olmoe_1b_7b|1|unsupported|3.653538|3.156181|45.753506|—|
|olmoe_1b_7b|2|unsupported|4.762878|4.341845|78.633358|—|
|olmoe_1b_7b|4|unsupported|6.215642|5.817549|128.368666|—|
|olmoe_1b_7b|8|unsupported|8.259111|7.907548|197.038825|—|
|olmoe_1b_7b|16|unsupported|10.395362|10.015969|262.257277|—|

Mistral native selection now completes all five batches. Each batch measures
the original predicted top4 with two fresh timing processes per candidate;
all20 candidates pass C1/C2 at512 steps. All40 timing processes complete.
The prescribed1% rule selects the final pin; no target default is promoted.
The earlier B1 manual development median10.130998ms remains in the CSV and
is superseded for this table by the solver-selected result.

Each accepted normal/watchdog pair has512 threads, MTreg<=255, frame0,
static smem0, checked MIR and actual resident1. Its own576 history reaches
the prefetch floor and is retained with the D7 disposition. Diagnostic
cap227 queries with resident0/small nonzero stack are never model launches.
One controller wrongly required resident1 for such a diagnostic and stopped;
recovery preserves its real values and the failure. Execution checks remain
unchanged. Fresh normal/watchdog pairs share HEAD7c9e3522e and source hashes.

Fresh prefill pairs retain same-shape MX4 geometry, watchdog0 for timing and
watchdog1 for checks, with fragment KV writes matching native decode.
Earlier failures exposed mismatched KV layout, missing Mistral rotary-table
dispatch, immutable-source drift, root filesystem free-block accounting,
and noexec RAM cache. Those failed attempts remain under runs/.
The committed Mistral rotary helper and root-space accounting fixes were
validated before this collection; no implementation source changed during
the final receipt-consuming run.

N32768 head costs cover100 legal keys in three fresh processes. Each batch
has203/203 legal cost keys. Non-head rows are reused only where component
shape and archived body/header bytes match. Canonical per-plan histories,
actual resource queries and E2E selection are all consumed by the existing
strict results verifier. Closed solver TSVs are archived losslessly with
full decompression SHA256 checks and readable original paths; the disk
floor was not lowered and failed binaries/logs were not deleted.

At that checkpoint OLMoE TileMega was unsupported and had no native latency. An actual
CPU export attempt rejects OlmoeForCausalLM at serving/export.py:22 before
any GPU launch. Its64 experts and QPerKV1/hkv16 geometry do not match
MxMoeModel.cpp:89,98. Its Q/K RMS spans the projected2048-vector before
head reshape, and norm_topk_prob=false retains full-softmax routing mass.
Changing only the geometry assertions would produce the wrong arithmetic.
All five native CSV statuses are capability_status_not_a_measurement with
blank latency fields. External baselines are complete; this native gap is
not an OOM and is not replaced with a legacy or dense-model result.

The prior CPU MoE reference mismatch was a routing-weight dtype contract:
installed HF5.12 uses BF16 weights while the old export kept FP32.
The exporter now uses the matching BF16 boundary; strict reference and
Mistral tests pass7/7 without changed tolerances. Native Qwen MoE still
has28/472 non-tie routing differences at B1/B16; C1/C2 pass independently.
Earlier eight auditor fixture cases still reject an older header closure at
their immutable hash check; they are not relabelled as passing. The15
targeted negative/dtype checks passed separately. Failed receipts and
fixture evidence remain retained. No full-model routing acceptance or
original MX5 completion is claimed.

Raw campaign: runs/mx5/extra_models_20261030_v1/, including checkpoint
identities, frozen baseline sources, process/guard receipts, failed builds,
mistral_complete_top4/, mistral_d7_top4_v2/ and prefill pairs.
Detailed metrics, route explanation and approved Qwen timing exceptions:
[MX5_collection_c550.md](MX5_collection_c550.md).
Original dense25/MoE5/default gates and DNN work remain open.
