# C550 bandwidth efficiency and AB4c

The confirmed scope is TP1 Qwen3-1.7B, Qwen3-30B-A3B-L24 and OLMoE-1B-7B
at B1/B4/B16, with graph and host-launch sequences. AB4b is **not measured**:
the reference is each cell's frozen main-experiment TileMega result and geometry.
Bandwidth efficiency reuses all 220 existing dense/MoE/TP timing rows across
TileMega, vLLM, SGLang and torch.compile. There are no external-baseline jobs.

Source designs were read from `origin/sm90r6` at `d47297f9b` and
`origin/sm90r6-ab4-hostlaunch` at `9d63adc5b`; immutable source manifests and
the confirmed scope are in `runs/mx8/design_sync/`.

## Queue

Run from the repository root with `PYTHONPATH=python`. First build the two
probes with `/opt/maca/tools/cu-bridge/bin/cucc -std=c++17 -O3`, adding
`-Xcompiler -resource-usage` for `hardware_probe.cu`; place both binaries in
`runs/mx8/hardware_build/` under their source names without `.cu`.

```sh
PYTHONPATH=python python experiments/mx8/queue.py --initialize-probes
```

Keep that process running. In separate shells, the following enqueue newly
built immutable plans and consolidate accepted evidence:

```sh
PYTHONPATH=python python experiments/mx8/prepare_cells.py
PYTHONPATH=python python experiments/mx8/launch_ablation.py
PYTHONPATH=python python experiments/mx8/launch_routes.py
python experiments/mx8/collect.py --watch
```

The two launchers may run concurrently. Every GPU execution goes through
`scripts/mx_gpu_job.sh`; at most four cards run concurrently and each card
has one job. All eight cards are authorized. Busy-card/guard rejections are
retained and requeued without a retry cap. Other failures remain failed and
block dependent timing jobs. `runs/mx8/queue/STATUS.json` is the live status;
`runs/mx8/tables/COLLECTION.json` describes completed data, not merely exits.
The queue keeps watching for new definitions until `INPUTS_CLOSED` exists.
Failed numerical checks are never retried as occupancy failures.

## Protocol

* Hardware: 8 GiB vector/LDS read probes and BF16 MMA, five fresh processes
  per item on each card. Normalize bandwidth with the largest **five-process
  configuration median** over the cards after all hardware jobs finish.
  C550 has 64 KiB LDS, so 32 KiB x4/x8 rings are explicitly unsupported.
* Each AB4c arm: five fresh correctness processes, full 1024-token single-step
  versus loop equality and cross-process/arm equality; the first also runs
  HF teacher forcing on 256 positions per row. Frozen main tokens are checked
  over the retained reference length, explicitly reported as 512 or 1024.
* Timing: three fresh processes, warmup1 + timed3 for both N1024 and N1;
  TPOT = `(median(E2E_N1024)-median(E2E_N1))/1023`. Also retain the event-TTFT
  formula used by the main MetaX experiment. A historical main comparison is
  not a fresh paired measurement.
* Production headers/pins/results are untouched. The builder overlays the
  runtime in `runs/`, retains every selected body/shape, and replaces grid
  stage barriers with same-stream kernel boundaries. Every stage initializes
  its own prefetch ring. MoE's per-CTA LDS route table is materialized between
  gate_up and down. Graph and host arms use identical stage kernels.
* Graph: one reusable decode-step graph and a device cursor advanced by a
  kernel; no host synchronization between decode steps. Host-launch: the
  identical stage sequence without a graph. Original prefill plans are reused.
* Routing: eight B>1 MoE captures, each with exactly 1024 positions: last prompt
  token (past63) and decode past64..1086. B1 has exactly eight distinct experts
  analytically. The same reference trajectory prices all four systems; this
  is logical-payload efficiency, not measured engine-specific DRAM traffic.

Only results passing all gates are assigned a final AB4c TPOT. Source/binary
hashes, compiler commands, failed attempts and raw arrays remain under `runs/`.
Compact tables are consolidated there first, then reviewed for publication.

## Offline acceptance

After the queue reaches zero pending/active jobs, create
`runs/mx8/queue/INPUTS_CLOSED`, stop the idle collection watcher, and run:

```sh
python experiments/mx8/collect.py
PYTHONPATH=python python experiments/mx8/audit.py
PYTHONPATH=python python -m unittest discover -s test/python -p test_mx8_experiments.py
```

The collector uses the five formal qualification job IDs registered for each
selected binary in `ABLATION_QUEUE.json`. Additional smoke artifacts are kept
as separate evidence and cannot substitute for a missing formal process.
The audit checks the original receipts, input and artifact hashes, stored
tokens, timing formulas, per-card concurrency, route arrays and resources.
It records process ranges; a completed collection does not imply low variance.
Graph/host comparisons with the historical main result are descriptive.

Vector probes may skip a final incomplete eight-load group. The published
rate uses their actual issued bytes, retaining nominal bytes and original raw
rates. This correction does not change the chosen LDS bandwidth peak.
