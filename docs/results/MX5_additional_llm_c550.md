# MX5 additional Mistral and OLMoE workload campaign

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

Completed vLLM two-round TPOT medians(ms):

| B | Mistral-7B-v0.3 | OLMoE-1B-7B-0924 |
|---:|---:|---:|
|1|11.148076|3.653538|
|2|12.525795|4.762878|
|4|12.676094|6.215642|
|8|12.986500|8.259111|
|16|13.668050|10.395362|

SGLang also completes both rounds for all five batches of both models.
Two-round baseline medians(ms):

| Model | B | vLLM | SGLang | torch.compile |
|---|---:|---:|---:|---:|
|mistral_7b_v03|1|11.148076|10.606020|18.437016|
|mistral_7b_v03|2|12.525795|12.029260|25.129721|
|mistral_7b_v03|4|12.676094|12.184568|35.886293|
|mistral_7b_v03|8|12.986500|12.631397|57.645759|
|mistral_7b_v03|16|13.668050|13.536814|100.390065|
|olmoe_1b_7b|1|3.653538|3.156181|45.753506|
|olmoe_1b_7b|2|4.762878|4.341845|78.633358|
|olmoe_1b_7b|4|6.215642|5.817549|128.368666|
|olmoe_1b_7b|8|8.259111|7.907548|197.038825|
|olmoe_1b_7b|16|10.395362|10.015969|262.257277|
 Mistral and OLMoE workload campaign

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

Completed vLLM two-round TPOT medians(ms):

| B | Mistral-7B-v0.3 | OLMoE-1B-7B-0924 |
|---:|---:|---:|
|1|11.148076|3.653538|
|2|12.525795|4.762878|
|4|12.676094|6.215642|
|8|12.986500|8.259111|
|16|13.668050|10.395362|

SGLang also completes both rounds for all five batches of both models.
Two-round baseline medians(ms); a dash means collection is still pending:

| Model | B | vLLM | SGLang | torch.compile |
|---|---:|---:|---:|---:|
|mistral_7b_v03|1|11.148076|10.606020|18.437016|
|mistral_7b_v03|2|12.525795|12.029260|25.129721|
|mistral_7b_v03|4|12.676094|12.184568|35.886293|
|mistral_7b_v03|8|12.986500|12.631397|57.645759|
|mistral_7b_v03|16|13.668050|13.536814|100.390065|
|olmoe_1b_7b|1|3.653538|3.156181|45.753506|
|olmoe_1b_7b|2|4.762878|4.341845|78.633358|
|olmoe_1b_7b|4|6.215642|5.817549|128.368666|
|olmoe_1b_7b|8|8.259111|7.907548|—|
|olmoe_1b_7b|16|10.395362|10.015969|—|

Mistral small BF16 decoder/HF, runtime rotary-table, architecture rejection,
and registry/config/prompt tests pass. Full32-layer bridges are exported.
Its B1 native manual development control passes C1/C2 at512 steps; normal
and watchdog decode use512 threads, MT229/231, frame/static0, MIR0 and
actual resident1/local0. Own576 history reaches the prefetch floor but
retains296B stack. Two new standard processes measure TPOT10.130749 and
10.131247ms; median10.130998ms, TTFT42.515704ms. All six full generations
agree. Speedup is1.100393x against vLLM and1.046887x against the faster
SGLang baseline. This is not a final exacttop4 solver choice.

The first engine attempt rejects linear prefill versus block_fragments
decode; the fresh pin enables mx_kv_fragments_v2 with unchanged geometry.
The second attempt exposes missing Mistral rotary_tables dispatch; the
committed helper now uses HF MistralRotaryEmbedding. The third attempt
passes checks and timing. All failures and binaries remain retained.
Prefill normal/watchdog construction now completes for all five batches,
using same-shape MX4 Llama8B geometries as manual controls. Later B8/B16
attempts hit the actual8GiB disk floor after solver TSV generation. Exactly
252 closed solver TSVs are archived losslessly, saving7.611699GiB; gzip
backups are checked by full decompression SHA256 and original readable
paths are retained through RAM views. Failed jobs are rebuilt successfully
without lowering the disk floor. Raw retry PROCESS receipts record rc0.

New N=32768 head costs complete100 legal keys across3 fresh processes.
All100 own-shape P0 and primary resource queries also complete. Every
included archived body/header equals current source bytes; old non-head
measurements are reused only for identical component shapes and source.
Five scoped schema3 tables each cover203/203 legal keys and generate
exact predicted top4 previews. These are diagnostic previews: per-plan
canonical576 histories, final resource qualification, changed-width costs,
model C1/C2 and four-by-two E2E selection remain required. Actual576 P0
head queries and the20 candidate primary builds are now queued on card6.
No target default or final pin is changed.

OLMoE native remains unimplemented. Its64-expert dispatch and QPerKV1/hkv16
geometry fail the current Qwen-native assumptions at
`lib/Codegen/MxMoeModel.cpp:89,98`. Its HF Q/K RMS spans the whole projected
2048-vector before head reshape, and norm_topk_prob=false retains full-softmax
routing mass; current native Qwen bodies implement different numerical
contracts. Both export() and direct Decoder construction now reject OlmoeForCausalLM
explicitly rather than emitting or evaluating a dense graph. No legacy native fallback or fabricated TileMega
latency is used. These are concrete pending adapter/body changes, not OOM.

Three Mistral prefill attempts failed before successful normal construction:
unprivileged disk allowance was0 while the root process had about14GiB;
a RAM-cache attempt compiled but its host queries hit the noexec mount;
then the C++ probe independently used the same unprivileged space check.
Python and C++ now count root-accessible free blocks for effective UID0;
non-root accounting and the1GiB/8GiB minimum checks remain unchanged.
CPU accounting tests2/2 pass. A separate compiler recompiled compile.cpp and
linked against the existing objects; its build succeeds and the original
producer path/hash is preserved for prior receipts. The successful prefill
uses this new producer; no old result is relabelled as a rebuild.

Raw campaign: `runs/mx5/extra_models_20261030_v1/`, including hub identities,
random checkpoint receipts, frozen baseline sources, every process/guard,
failed builds, `mistral_decode_development/`, `tilemega_v4/`, `tilemega_v5/` and head preparation.
The full MX5 LLM/routing/default gates remain open. These added workloads do
not mark the original dense25/MoE5 campaign or DNN phase complete.

An existing CPU MoE reference test fails strict BF16 equality at1 token,
max_abs0.0009765625. Isolated old622c505e6 sources reproduce it exactly;
the new architecture guard does not introduce it. Fixed-input diagnosis
shows installed HF5.12 returns BF16 routing weights while the old export
keeps FP32. Matching only that weight boundary yields exact grouped_mm
output for1/3/17/64 tokens; the unchanged original export differs at each
size. This establishes a CPU contract mismatch, not the complete cause of
native full-model route failures. No tolerance, seed, HF reference or native
arithmetic is changed to make a gate pass. Raw: old-source control and
`moe_recovery_20261030_v1/hf_dtype_contract_cpu/RESULT.json`.

The verified routing-dtype mismatch is now repaired in the exporter by the
same BF16 cast used by installed HF. The original strict test and Mistral
suite pass7/7 without changing seeds, expected values or tolerances.
The plan-source auditor now binds dispatch/combine explicitly to the chosen
routing dtype;15 targeted negative/dtype checks pass. Eight older fixture
cases still reject the pre-router-optimization header closure at its hash
check; that evidence is preserved, not updated to accept current headers.
Own-BF16 MoE plans are being rebuilt and tested separately from all old rows.

The first20 Mistral primary resource builds complete, but subsequent native
receipt consumption rejects changed immutable Python inputs after the MoE
export fix. No such receipt is qualified or reused as current evidence.
A fresh20-candidate canonical campaign is started after the source changes
settle, reusing only the unchanged actual576 P0 head query observations;
all subsequent builds and actual zero-launch queries use the existing
prescribed history auditor. Old failed controllers/receipts remain retained.

Current native boundaries: Mistral B1 has the verified two-round manual
result10.130998ms, while B2/B4/B8/B16 still lack accepted native timings.
All20 fresh candidate primary builds complete. Canonical history and own
original512 builds/queries are progressing; the first B1/B2 artifacts fit
(MT229/frame0 for B1 candidate0), but the auditor correctly still requests
candidate-bound D7 review. This is resource progress, not exacttop4 E2E
selection. OLMoE has all baseline data and no native TileMega timing.
These missing TileMega cells remain open work, not zero-latency entries.
