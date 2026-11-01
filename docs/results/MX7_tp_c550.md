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
  Device P2P access and LL visibility are tested separately below.
- **verified** the two frozen dense pins reproduce the exact `config_json`
  of the selected 2026-10-29 MX-5 source-bound comparison rows (B1/B16).
- **verified** all eight reference plans compiled from unchanged starting source:
  dense B1/B16, Qwen MoE B1/B16 decode/prompt, OLMoE B1 decode/prompt.
  `MX7_frozen_gate_c550.csv` records CU/binary/pin SHA256 and per-function
  instruction counts from `scripts/mx_mir_report.py`; no MoE GPU runs occurred.
- **verified** protected MoE source and final CSV hashes are unchanged.
  Onboarding environment, reference hashes and exact prospective TP invocation:
  `runs/mx7/sync/onboarding.md` (not committed).

## CP1: work in progress

- **verified** all 12 ordered pairs within GPUs 4–7 support peer access and
  pass a device load/store reachability check. No topology-based speed claim.
- **verified** group guards use the existing per-card locks in ascending order;
  CPU mocks cover lock order, any-card occupancy and unchanged single-card use.
- **verified** default `cudaMalloc` / volatile LL8 pair cohort: 0/5 passed;
  torn=0, timeouts=1128 per process. These failures are retained in the probe CSV.
- **verified** fine-grained allocation / volatile LL8 passed 5/5 pair processes
  at P=1024 and 5/5 all-to-all processes at each P=1024 and P=262144;
  each passing process covers 2000 epochs with all 104 CTAs.
  Full combination qualification and latency/bandwidth comparison remain open.
- **verified** frontend TP changes pass the eight-plan frozen gate; TP codegen
  changes separately pass all eight again: generated CU bytes and all function
  instruction counts match. No frozen-path GPU performance run was necessary.
- **verified** 20 CPU tests pass for shard packing, group guards, optional ABI,
  TP communication cost ranking, k-best and the 1% rule.
- **verified** TP2 baselines below each use two fresh processes, warm1/timed2;
  TP4 remains in progress. TP1 columns in the CSV are frozen MX-5 values.

| B | vLLM TP2 TPOT ms | SGLang TP2 TPOT ms | torch.compile TP2 |
|---|---:|---:|---|
| 1 | 3.912391 | 3.083636 | N/A |
| 2 | 4.001944 | 3.438704 | N/A |
| 4 | 4.031721 | 3.487290 | N/A |
| 8 | 4.302299 | 3.776927 | N/A |
| 16 | 4.624672 | 4.215727 | N/A |

## Autonomous handling

- **verified, B2** torch.compile TP2 fails during HF `tp_plan="auto"` loading:
  `ValueError: Using a tp_plan requires accelerate.` All ten failed jobs and
  source identities are retained; all five cells are N/A. No eager substitution
  or dependency installation was made. NCCL process-group initialization works.
- **verified, B4** cost benches now use the dense V2 synchronization and a local
  grid variable for the compiler assumption, as the dense program does.
  CPU resource rejections and the earlier build snapshot remain under runs.
- **verified, B4** an initial frozen-gate report path error and a test invocation
  error were corrected; earlier logs remain. The completed frozen gate and
  corrected test invocation are reported above.

## Remaining work

C0/CP1 communication qualification and TP4 baselines; CP2 TP device code,
runtime and GPU correctness; CP3 GPU costs, TP constants, selection and finals.
No TP performance or correctness result is available yet.
