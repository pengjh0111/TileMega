# MX-7: Qwen3-1.7B tensor parallelism on C550

Evidence labels: **verified** = executed here; **stated** = supplied specification;
**inferred** = reasoning, not a measurement. Raw evidence is under `runs/mx7/`.

## Historical A5 stop: resumed with all-card authorization

- **verified** at 22:07:12 and 22:10:45 CST on 2026-11-01, `mx-smi -i 4,5,6,7`
  reports GPU6 **Not Available**, GPUs4/5/7 Available, and no device processes.
  The same state persists at 22:27:25 after our submitters have exited.
- **verified** a fresh original probe fails inside the initialization call at
  `experiments/mx7/ll_litmus.cu:147`, before peer binding or the LL kernel:
  `mxkwCreateQueueBlock` ioctl fails (`queues.c:850`), then
  `Device::acquireQueue` fails (`mx_device.cpp:1626`), followed by SIGSEGV.
  A signal backtrace reaches `mcMemset+0x2ae` / `wcudaMemset+0x4f`;
  host markers show a successful 4 MiB allocation immediately before memset.
  This is logical device0 in the guarded {6,7} group, hence physical GPU6.
- **verified** independent vLLM and SGLang TP4 initialization also reports this
  queue failure. This is not evidence of LL tearing or AR numerical failure.
  Why GPU6 became unavailable is **unresolved**; primitive-specific causation
  is not established by the chronology or these host stacks.
- **inferred, A5** further progress requires device recovery outside the
  specified implementation. All our GPU submitters have been stopped; no
  reset, driver, clock or power change was attempted. Raw logs, failures and
  interrupted cohorts remain, including `runs/mx7/A_STOP.json`.
- Communication primitives/allocation and the TP2 main group are **undecided**.
  No production TP device code or TP model launch was attempted. There is
  **no TileMega TPOT, speedup, scaling efficiency or final selected TP pin**.
- Code stop HEAD: `1f6dc640b` (23 commits after the starting HEAD);
  results/report are committed separately immediately afterwards.

## Resume: C0 qualified; implementation and baselines continuing

- **stated** user authorizes all physical cards and further torch.compile adaptation;
  instruction SHA256 `f67014253a3cca7234429ff3671e293c0104fb04b68fde1bb9dc52b518c0b284`.
- **verified** LL8 / fine-grained allocation / volatile AS1 store / `__ldcg` load:
  pair and four-card P=1024/262144 each pass 5/5 fresh processes, 2000 epochs,
  104 CTAs, zero tearing/timeouts. Current TP4 group {1,3,4,5}, TP2 main {1,3}.
- **verified** single-way latency: {1,3} 2373.504 ns; {4,5} 2629.120 ns (5 processes).
  Selected coalesced P2P bandwidth at 104 CTAs: one peer 8.121 GB/s;
  three peers 19.253 GB/s aggregate. These are diagnostics, not model speedups.
- **verified** GPUs0/6/7 became unavailable during previous optional probe runs;
  their cause remains unresolved. Failed records remain; no reset was attempted.
  No further unsafe system-atomic probe combinations are scheduled.
- **verified** torch.compile fullgraph TP2 B1 now completes warm1/timed2:
  first process TPOT 14.391347 ms. This is not yet a two-process cell result.
  Isolated accelerate 1.15.0 resolves the dependency; compiled autograd resolves
  HF's TP gradient-hook compiler rejection without removing hooks or using eager.
- **verified** guarded TP2 torch.compile and TP4 baseline campaigns are running.
  Core TileMega TP device implementation starts after this C0 decision.

## CP2 implementation progress

- **verified** peer-bound runtime, poisoned-plan checks, prompt mode, residual AR
  and LL head exchange compile. No production communication fence was added.
- **verified** the initial development builds were rejected at 256 MTreg and
  336–472 stack bytes. The approved per-item send/receive fallback is used.
  Dynamic private pointer-array indexing also materialized a 232-byte operand;
  constant-field pointer selection removes it without changing the protocol.
- **verified** development TP2 B1 / TP4 B1 / TP4 B16 now have respectively
  229 / 225 / 226 MTreg, zero stack/static memory. Actual residency is pending.
- **verified** the eight frozen CU/instruction comparisons pass after these
  changes. CPU tests: 29 passed. Device suites and full-model checks are queued.
- **verified** preliminary AR TP2 B1: 5/5 fresh processes, nine legal geometry
  cases per process, exact ordered BF16 reference and zero mismatches.

## Autonomous handling during CP2 / calibration

- **verified, B4** the combined TP4 development driver scheduled20 fresh
  processes under one900s guard. Fifteen exited normally; a sixteenth produced
  matching1024 tokens before batch teardown was interrupted. Individual runs
  took about52–65s, so this is aggregate batching exhaustion, not a kernel hang.
  Original guard124 and all process outputs remain. B16 litmus is isolated
  into a fresh five-process job; no protocol or acceptance limit is changed.
- **verified, B4** constant probe TP2 B8/TN128 failed in the non-AR comparison:
  its fixed32784-byte workspace was smaller than the GEMM's LDS layout.
  Workspace, generation offset and launch allocation now use the same actual
  layout bound. Old failure remains; affected/unmeasured scopes are rebuilt.
  Earlier B1/B2/B4 layouts fit and their valid three-process samples are retained.
- **verified, B4** three development resource-ledger LDS entries used an assumed
  floor. Corrected CU-derived values are appended, preserving the original rows;
  all four development plans subsequently pass actual all-rank resident=1 queries.

## Specification and scope

- **verified** starting branch/HEAD: `tilemega-metax`,
  `6cab45b9604e306fef1916863fe16b09d895c32f`; clean tree; fetch/pull current.
- **verified** external specification: `/root/Prompt/TileMega_MX7_prompt.md`, UTF-8/LF,
  SHA256 `5caed7edd4477b8fb63d33bed9254bbc4e33d92041eb5969113fefeb805d7cee`.
- **stated** scope: Qwen3-1.7B, TP2/TP4, B=1/2/4/8/16, BF16 greedy,
  ISL/OSL=64/1024. The user subsequently authorized all GPUs; TP1 and MoE remain frozen.
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

## CP1: partial, stopped at A5

- **verified** all 12 ordered pairs within GPUs 4–7 support peer access and
  pass a device load/store reachability check. No topology-based speed claim.
- **verified** group guards use the existing per-card locks in ascending order;
  CPU mocks cover lock order, any-card occupancy and unchanged single-card use.
- **verified** default `cudaMalloc` / volatile LL8 pair cohort: 0/5 passed;
  torn=0, timeouts=1128 per process. These failures are retained in the probe CSV.
- **verified** fine-grained allocation / volatile LL8 passed 5/5 pair processes
  at P=1024 and 5/5 all-to-all processes at each P=1024 and P=262144;
  each passing process covers 2000 epochs with all 104 CTAs.
  These are historical passing cohorts, not a current communication decision.
  Pair matrix: 219/252 completed cohorts, including all 126 small-packet
  combinations (8 passed); all-to-all matrix: 13/252 completed, none passed.
  Additional fine-grained all-to-all passing cohorts are recorded separately.
  Each cohort has five fresh processes; passing ones cover all 2000 epochs.
- **verified** fine-grained volatile ping-pong on {4,5}: median single-way
  latency 2557.952 ns over five processes; pointer device ownership and 64-byte
  alignment pass 5/5. This does not determine the main TP2 group: {6,7} and
  the required bandwidth matrix are incomplete. Large-packet LL timings use
  strided accesses and must not be presented as bandwidth measurements.
- **verified** frontend TP changes pass the eight-plan frozen gate; TP codegen
  changes separately pass all eight again: generated CU bytes and all function
  instruction counts match. No frozen-path GPU performance run was necessary.
- **verified** 23 CPU tests pass for shard packing, group guards, optional ABI,
  TP communication cost ranking, k-best, qualification hashes, the 1% rule,
  noise eligibility and trace attribution/compression.
- **verified** TP2 baselines below each use two fresh processes, warm1/timed2;
  TP4 has 14 incomplete single-process records: 4 measured, 4 dependency N/A,
  6 environment-invalid. They are not formal two-process baseline results.
  TP1 columns and baseline scaling efficiencies use frozen MX-5 values.

| B | vLLM TP2 TPOT ms | SGLang TP2 TPOT ms | torch.compile TP2 |
|---|---:|---:|---|
| 1 | 3.912391 | 3.083636 | N/A |
| 2 | 4.001944 | 3.438704 | N/A |
| 4 | 4.031721 | 3.487290 | N/A |
| 8 | 4.302299 | 3.776927 | N/A |
| 16 | 4.624672 | 4.215727 | N/A |

## Prepared implementation and verification

- **verified** rank-sharded meta export, frontend recipes/`serving.tp`, shard
  loading/state, optional Python ABI, TP cohort engine and measurement/check
  tools, dense TP cost-domain/build/measurement tools, k-best qualification
  orchestration and trace collection are committed. Their GPU TP paths are
  unverified; this preparatory work is not a usable native TP implementation.
- **verified** `MX7_frozen_gate_c550.csv`: 8 reference builds plus 8 frontend
  and 8 codegen rechecks, all pass CU-byte and function-instruction equality.
  `MX7_resources_c550.csv` contains the 24 CPU resource reports; resident
  occupancy is explicitly **not queried**. No GPU resource gate is claimed.
- **verified** `MX7_tp_correctness_c550.csv` records CPU tests only. AR/head
  exchange tests, whole-model litmus, C-1/C-2 and rank-token gates remain open.
- **verified** the uncompiled runtime draft is preserved as
  `runs/mx7/sync/runtime_tp_unfinished.patch`; the repository runtime was restored
  to the frozen version. TP1 and protected MoE code/data hashes remain intact.

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
- **verified, B4** small-packet screening was prioritized using the same binaries,
  polling bound, 104 CTAs and five-process cohorts. Only our job submitters were
  paused/released; active guarded probes were allowed to finish.
- **verified, B4** compile-time primitive specializations keep the LL protocol
  unchanged. Seven small-packet pair cohorts passed 5/5; one exited SIGSEGV.
  Later unrelated combinations also failed. They are retained as failures,
  not used to select a primitive while the device is unavailable.
- **verified, B4/B5** gdb is absent; signal stacks and host markers located the
  initialization failure. TP4 queue failures are environment-invalid records,
  not engine qualification or performance data. No retries conceal failures.
- **verified** provisional cost builds stay under runs; no GPU stage costs,
  communication constants or eligible cost table were published.

## GPU use

**inferred** child-job wall time from each guard's pgid-file to result-file
timestamps, including initialization/cleanup and excluding lock waiting:
GPU4 1.394 h, GPU5 1.394 h, GPU6 2.681 h, GPU7 2.681 h; total **8.150 card-h**.
This estimates reserved job time, not hardware busy time. Accounting details:
`runs/mx7/sync/gpu_usage.json`. No jobs ran on GPUs0–3.

## Remaining work

| Item | Existing location / missing target | Reason |
|---|---|---|
| C0 decision, bandwidth, main pair | `experiments/mx7/ll_litmus.cu:147` | GPU6 fails before LL launch |
| TP AR and head exchange | `DecodeGemmV2.h:332`, `DecodeHeadV2.h:32`; new `TpLinkV2.h` / `PeerAccess.h` not created | C0 selection prerequisite |
| Binding and native ABI | `DecodeProgramV2.cuh:36`, `DecodeRuntimeV2.cuh:143` | Native TP path not implemented |
| GPU checks and engine validation | `tp_engine.py:21`, `plan.py:114` | Optional native ABI is absent |
| GPU costs/constants/top4/finals/trace | `tilemega/tp/solve_tp.py`, `experiments/mx7/` | CP2 and C0 unfinished |

`MX7_tp_constants_c550.csv`, `MX7_tp_selection_c550.csv`,
`MX7_tp_final_c550.csv` and selected TP pins do not exist; no synthetic rows.

## Expectations and next steps

No TileMega TP measurement exists, so the TPOT/speedup expectations cannot be
evaluated. **Inferred** next steps require recovering GPU6 first, diagnosing
the availability loss, and repeating affected qualification samples while
retaining failures. A GPU-State check in the guard would help reject this
environment earlier; it has not been added after the A5 stop.
