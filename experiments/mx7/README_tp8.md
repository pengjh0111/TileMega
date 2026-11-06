# C550 TP8 collection

The user authorized all eight physical GPUs on 2026-11-06 (stated).
This campaign adds Qwen3-1.7B TP8, batches 1/2/4/8/16, for TileMega,
vLLM, SGLang and torch.compile. Existing TP2/TP4 baselines remain frozen.

From the repository root:

```bash
scripts/mx_tp8_sweep.sh plan
scripts/mx_tp8_sweep.sh prepare --out runs/mx7/tp8_queue
scripts/mx_tp8_sweep.sh run --out runs/mx7/tp8_queue
scripts/mx_tp8_sweep.sh status --out runs/mx7/tp8_queue
```

`plan` prints the scope without importing a GPU framework or creating a run.
`prepare` exports TP8 and builds P2P, LL8, AR and head tests on the CPU.
`run` reuses that preparation, collects the three baseline systems, qualifies
the eight-card link, measures TP8 costs, runs top-4 selection and completes the
selected-plan checks. `status` is read-only and can run beside the campaign.
The command defaults to `plan` to make an accidental invocation harmless.

For an unattended run:

```bash
mkdir -p runs/mx7/tp8_queue
nohup scripts/mx_tp8_sweep.sh run --out runs/mx7/tp8_queue \
  > runs/mx7/tp8_queue/launcher.log 2>&1 &
```

Each GPU job waits for all requested cards to satisfy the existing guard's
availability, ownership, hidden-memory, utilization, power and free-memory
conditions. It also checks the same per-card locks used by other TileMega
jobs. Readiness polling holds no GPU locks; the guard acquires the complete
group before launching the child, monitors it while running and checks it
after exit. No foreign process is stopped. Device monitoring has no deadline
by default (`--idle-wait-s -1`), and occupancy/guard interruptions have no retry
limit, as requested by the user on 2026-11-07 (stated). `--poll-s` defaults to
10 seconds. Each invalid attempt keeps a distinct directory and process record;
the next attempt waits for idle cards again. Actual GPU jobs retain their
900-second limit.

The baseline scope has 15 cells and 30 fresh processes. Systems are interleaved
within each batch and round. Each process uses BF16 greedy generation,
ISL/OSL=64/1024, warmup1 and timed2 for both N1024 and N1. Baseline TPOT is
`(E2E_N1024 - E2E_N1)/1023`; TileMega retains its existing prompt-mode CUDA
TTFT and wall E2E formula. The aggregate takes the median of two fresh
processes. TTFT, E2E and output throughput are retained alongside TPOT.

TileMega uses TP8's actual per-rank shapes: qkv N512, o K256, gate_up N1536,
down K768, head N18992 and hkv1. Its geometry comes from TP8 measured costs
and communication constants, with four resource-qualified candidates,
C-1/C-2 and the existing 1% E2E selection rule. TP2/TP4 pins are not copied.
An eligible measured TP8 table can be supplied using `--costs <table.json>`;
the matching `.qualification.json` and complete five-batch TP8 scope are
required. Otherwise the campaign builds and measures the full legal domain.
`--workers` limits CPU cost compilation, default 8.

Before accepting TileMega performance, the campaign requires all 56 ordered
P2P pairs, LL8 all-to-all at both packet counts (2000 epochs, five fresh
processes), and AR/head tests at B1/B16 (five fresh processes each). Selected
plans receive C-1 plus five fresh 1024-token C-2 starts. B1/B16 also receive
five injected-delay starts against their ordinary token reference. Timing
binaries have no watchdog, trace, DIAG or injected delay.

Executables default to the existing validated environments:

| System | Executable override |
|---|---|
| TileMega | `TILEMEGA_PYTHON` (default `python`) |
| vLLM | `--vllm-python /root/venv_vllm/bin/python` |
| SGLang | `--sglang-python <current Python>` |
| torch.compile | `--torchcompile-python /root/venv_mx7_tc/bin/python` |

The torch.compile environment includes the prior TP adaptation and pinned
accelerate dependency. There is no eager substitute for a failing compiled
baseline. An engine failure or malformed output is recorded as N/A with the
original log; a busy-card refusal remains pending. Running guard interruptions
return to the queue without exhausting a retry counter. This also applies to
solver correctness/timing cohorts, residency queries and nested qualification
jobs. Numerical or synchronization failures stop TileMega acceptance and
cannot be relabelled as occupancy failures.

Use the same `--out` and options to resume. Successful jobs and baseline
rounds are hash-checked and reused. A campaign lock prevents two launchers
from collecting the same run concurrently. All attempts, failures, commands,
process records, sources, binaries and token files remain under `runs/`.
Changing source code or model inputs requires a new campaign identity.

| Output in the run directory | Contents |
|---|---|
| `TP8_results.csv` | 20 cells, metrics, status and evidence links |
| `TP8_table.md` | Five-batch comparison and TPOT speedup |
| `STATUS.json` | Passing, N/A and pending counts |
| `jobs/*/attempt*/guard/readiness.jsonl` | Per-card readiness samples |
| `jobs/*/attempt*/guard/guard.jsonl` | Guarded GPU telemetry |
| `jobs/*/attempt*/stdout.log` | Exact child output and errors |
| `jobs/select_B*/attempt*/selection/` | Four candidates, selected pin and checks |
| `IDENTITY.json`, `source/` | Source hashes and archive; protected prior data |
| `ACCEPTED.json` | Written only after all required campaign checks finish |
| `LAST_STOP.json` | Latest stop reason; earlier attempts are retained |

Validation on 2026-11-07 (verified): 65 focused CPU tests pass; TP8 export and
a development-only B1 plan compile with 512 threads, MTreg229, stack0 and
static0. The CPU legality helper enumerates 700 TP8 coordinates with the
expected rank shapes. These are preparation checks, not GPU correctness or
performance results. No TP8 timing result is claimed until collected.

The earlier bounded-attempt campaign remains intact under `runs/mx7/tp8/`.
It produced no passing TP8 timing cells. The persistent queue uses
`runs/mx7/tp8_queue/`, with a new source identity; its launcher is recorded in
`runs/mx7/TP8_ACTIVE.json`. No old failure row or frozen TP2/TP4 data is changed.
