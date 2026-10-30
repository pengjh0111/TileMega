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
2rounds and median follow MX5. vLLM/SGLang/torch.compile are queued on
physical5(Mistral) and4(OLMoE), with one guarded job/card and900s per job.
OLMoE torch.compile explicitly uses eager experts and the separately labelled
compile-boundary repair. A baseline failure is retained, never replaced with
another system's result. New data are append-only in
[MX5_additional_llm_c550.csv](MX5_additional_llm_c550.csv).
Single-round rows are provisional; only median rows combine both fresh rounds.

Mistral small BF16 decoder/HF comparison passes, as do unsupported-architecture
and sliding-window rejection tests. Registry/config/prompt validation passes.
Full32-layer decode and prefill bridges are exported. Its B1 decode control
uses the same non-head shapes as Llama8B and a valid native head geometry;
this is a manual development control, not exacttop4 solver selection.
The model's own576 history retains296B stack at the prefetch floor;512
normal/watchdog builds use MT229/231 and zero stack, static memory and MIR
violations. Actual occupancy, C1/C2 and two standard rounds are queued.
B1 prefill normal construction succeeds with the MX4 same-shape Llama8B
prefill pin; watchdog construction is pending at this checkpoint.
For final selection,100 legal new N=32768 head keys are prepared for3 fresh
processes. Source archives equal all live included body files byte for byte;
only the compiled runtime-shape body is reused, never old head timings.
The new-cost collection follows the B1 model checks/timing on physical6.
Other B native integration and exacttop4 remain pending.

OLMoE native remains unimplemented. Its64-expert dispatch and QPerKV1/hkv16
geometry fail the current Qwen-native assumptions at
`lib/Codegen/MxMoeModel.cpp:89,98`. Its HF Q/K RMS spans the whole projected
2048-vector before head reshape, and norm_topk_prob=false retains full-softmax
routing mass; current native Qwen bodies implement different numerical
contracts. The exporter now rejects OlmoeForCausalLM explicitly rather than
emitting a dense graph. No legacy native fallback or fabricated TileMega
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
failed builds, `mistral_decode_development/`, `tilemega_v4/` and head preparation.
The full MX5 LLM/routing/default gates remain open. These added workloads do
not mark the original dense25/MoE5 campaign or DNN phase complete.
