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
with the host PID evidence above. The 40 TileMega cells are pending SM80
implementation and validation. No TPOT, throughput or speedup has been
measured for this campaign. Blank fields in the CSVs denote missing evidence.
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
`runs/a100/`.

For a restarted container, use its read-only host `/proc` mapping at
`/host/proc` or set `TILEMEGA_HOST_PROC` to the supplied path.

```sh
cd /work/TileMega-a100
export PYTHONPATH="$PWD/python" PYTHONDONTWRITEBYTECODE=1
runs/a100/envs/tilemega/bin/python -u scripts/a100_baselines.py run
runs/a100/envs/tilemega/bin/python scripts/a100_results.py
```
