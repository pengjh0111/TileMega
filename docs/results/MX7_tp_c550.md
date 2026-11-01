# MX-7: Qwen3-1.7B tensor parallelism on C550

Evidence labels: **verified** = executed here; **stated** = supplied specification;
**inferred** = reasoning, not a measurement. Raw evidence is under `runs/mx7/`.

## Specification and scope

- **verified** starting branch/HEAD: `tilemega-metax`,
  `6cab45b9604e306fef1916863fe16b09d895c32f`; clean tree; fetch/pull current.
- **verified** external specification: `/root/Prompt/TileMega_MX7_prompt.md`, UTF-8/LF,
  SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.
- **stated** scope: Qwen3-1.7B, TP2/TP4, B=1/2/4/8/16, BF16 greedy,
  ISL/OSL=64/1024; physical GPUs 4–7 only. TP1 baselines and MoE are frozen.
- **verified** background read: `MX7_plan.md`, `MX6_final_review.md`, `CLAUDE.md`.
  No H800 commits merged; those implementations are read-only references.

## CP0: environment and frozen references

- **verified** MACA `3.8.2.6`; cucc/mxcc `1.0.0 (890c70f3aa)`; driver `3.8.1`.
  `cmake --build build --target tilemega -j 8` succeeds at the starting source.
- **verified** Torch `2.10.0+metax3.8.2.2`, Transformers `5.12.0`,
  SGLang `0.5.14+detached.c432db2fb.maca3.8.2.122`;
  vLLM environment `/root/venv_vllm`: `0.25.0+empty`.
- **verified** `torch.distributed`: NCCL and Gloo available;
  MPI/UCC/XCCL unavailable. Availability alone does not qualify an engine.
- **verified** `mx-smi -i 4,5,6,7`: four available C550s, 64 GiB each, no owners.
  Read-only topology commands: `mx-smi topo -m`, `-p`, `-d`.
  The link matrix labels every pair MX; PCIe-only labels them NODE.
  Device P2P access and LL visibility remain to be tested by C0.
- **verified** the two frozen dense pins reproduce the exact `config_json`
  of the selected 2026-10-29 MX-5 source-bound comparison rows (B1/B16).
- **verified** all eight reference plans compiled from unchanged starting source:
  dense B1/B16, Qwen MoE B1/B16 decode/prompt, OLMoE B1 decode/prompt.
  `MX7_frozen_gate_c550.csv` records CU/binary/pin SHA256 and per-function
  instruction counts from `scripts/mx_mir_report.py`; no MoE GPU runs occurred.
- **verified** protected MoE source and final CSV hashes are unchanged.
  Onboarding environment, reference hashes and exact prospective TP invocation:
  `runs/mx7/sync/onboarding.md` (not committed).

## Autonomous handling

None requiring a specification decision so far.

## Remaining work

C0/CP1 communication qualification and group guard; baselines; CP2 TP model,
device code, runtime, engine and correctness; CP3 costs, solver and final results.
No TP performance or correctness result is available yet.
