# TileMega A100 migration

## Baseline status

**Verified.** The local and remote `origin/tilemega` refs matched
`7425a5242580af16a7ea43be742e313a6eedb351`. Branch `a100` uses an independent
worktree at `/work/TileMega-a100`. The original checkout has pre-existing
untracked Barvinok build inputs.

**Verified.** The container exposes NVML index 0, NVIDIA A100 80GB PCIe
`GPU-5f9e7f53-b146-b1a6-7d48-f5b703cd2c74`, PCI `0000:3b:00.0`, NUMA node 0,
MIG disabled, compute capability 8.0, and 81,920 MiB of physical memory.
Driver 580.178.04 reports CUDA 13.0 support. Local NVCC is 12.8.93.
The guarded driver probe is queued to confirm the runtime-visible SM count,
shared-memory limits, register limits, and CUDA ordinal-to-UUID mapping.

**Verified.** The initial occupancy record contains 31 samples over 34.76 s.
Every sample showed 18 external or unresolved compute PIDs, 100% utilization,
43,126 MiB used, 38,027 MiB free, 59–61 °C, 102.90–126.44 W and an inactive
throttle mask. The container's `/proc` is in a child PID namespace, and the
scheduler could not resolve NVML's host PIDs in it. The occupancy record and
dispatch decision are in `runs/a100/initial-occupancy.jsonl` and
`runs/a100/baseline/occupancy.jsonl`.

**Verified.** All 160 baseline cells have three pending fresh-process jobs
with the host PID evidence above. All 40 TileMega cells await GPU validation.
TPOT, throughput and speedup remain pending. Blank fields in the CSVs denote
missing evidence.
The actual measured suite versions, checkpoint identities and artifact hashes
will come from per-process receipts.

| Installed environment | Framework | Torch | Transformers |
|---|---|---|---|
| vLLM | 0.10.2 | 2.8.0+cu128 | 4.57.1 |
| SGLang | 0.5.5 | 2.8.0+cu128 | 4.57.1 |
| TensorRT-LLM | 0.21.0, TensorRT 10.11.0.33 | 2.7.1+cu126 | 4.51.3 |
| TileMega and TorchCompile | torch.compile | 2.13.0+cu126 | 5.5.3 |

**Verified.** CPU metadata inspection used the existing name-seeded generator's
Transformers 4.57.1 checkpoint layout for every engine, including the
per-expert weight keys. The frozen 64-token prompts and all eight model config
hashes are in the results tables. Qwen3-30B-A3B has 56.871 GiB of BF16 weight
data. Its engine-specific memory requirement is pending measurement. The
TorchCompile environment reuses the installed Accelerate and psutil packages
through the isolated environment.

## Queue and measurement

Each job runs TP1, BF16 weights and KV, greedy 64/1024 generation, one fixed
batch, checkpoint seed zero, and auto loading. Each of N=1 and N=1024 has one
warmup and three formal request-wall samples. The per-process TPOT is
`(median(T1024) - median(T1)) / 1023`. The collector takes the per-cell
median across three independently accepted baseline processes. It checks
token counts, guard receipts and the shared checkpoint identity.

The scheduler is waiting on the same UUID. Jobs require readable host PID
mapping, the workspace's `/work/tilemega-gpu.lock`, 31 idle samples one
second apart, enough free memory, and verified ownership during execution.
The preflight uses utilization, process ownership, and memory, with power
recorded as telemetry. The CUDA probe is the next job, followed by the
baseline matrix. Queue, state, attempt logs and receipts live under
`runs/a100/`. The next action and admission blocker are recorded in
`runs/a100/next.json`.

For a restarted container, use its read-only host `/proc` mapping at
`/host/proc` or set `TILEMEGA_HOST_PROC` to the supplied path.

```sh
cd /work/TileMega-a100
export PYTHONPATH="$PWD/python" PYTHONDONTWRITEBYTECODE=1
runs/a100/envs/tilemega/bin/python -u scripts/a100_baselines.py run
runs/a100/envs/tilemega/bin/python scripts/a100_results.py
```

## Offline native decode verification

**Verified.** The SM80 CLI build uses a separate `build-a100` directory with
the installed MLIR, ISL and pinned CUTLASS. Fixed coordinates over the frozen
exports generated native CTG/tmexec and persistent CUDA for Llama-3.2-1B B1
and B16, OLMoE-1B-7B B1 and Qwen3-30B-A3B B1. Generated Llama programs use a
108-CTA test grid and zero tensor maps. Component cubins for GEMM, attention,
router and fused experts disassemble to SM80 HMMA, LDGSTS and MBAR.
Compilation and disassembly evidence lives under `runs/a100/build/`.

**Verified.** CPU tests cover resource accounting for 256 residual tasks on
108 CTAs, the strict host PID admission policy, the baseline request protocol
and the artifact contract. The seed-zero Llama-3.2-1B checkpoint passed the
CPU memfd, manifest, safetensors and BF16 checks. Its checkpoint SHA-256 is
`5a53ddca4daa1421acb73fc07e8d328fb7d43f4258d1b6b1bdc0d18be2748c9d`.

**Verified.** Llama-3.2-1B prefill and decode pairs passed the host ABI
check at B1 and B16. Both phases of each pair have SM80 artifact identities.
The persistent decode requests 128,000 bytes of dynamic shared memory at B1
and 129,536 bytes at B16. Artifact paths, SHA-256 values and host ABI receipts
are indexed in `runs/a100/build/verified_artifacts.json`.

**Verified.** OLMoE-1B-7B and Qwen3-30B-A3B also produced B1 SM80 persistent
decode libraries with verified artifact identities. OLMoE retains prompt
prefill. The frozen Qwen3-30B-A3B inputs contain a decode export, and its
separate prefill remains pending.

**Verified.** The SM80 compile entry accepts `--search pipeline` and requires
`sm80_native_constants` from the A100 target. The search stops at this gate
with the current uncalibrated `configs/targets/sm_80_a100_pcie.json`.
The fixed build coordinates are compilation fixtures. The runtime checks the
physical SM count, dynamic shared-memory opt-in and compiled occupancy before
creating a persistent launch.

GPU validation remains pending for all cells. The Llama pilot uses separate
prefill and decode libraries. Completing the original prefill configurations,
including the separate Qwen3-30B-A3B prefill, current-body calibration,
hardware-correctness repetitions, replay and measured pipeline selection
remain required before final TileMega timing. The Llama pilot still needs
teacher-forced validation, step-versus-loop checks and 1,024-token execution
over five fresh processes.
