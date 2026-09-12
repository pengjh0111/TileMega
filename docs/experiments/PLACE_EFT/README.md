# PLACE_EFT — EX-S2: EFT placement and ordering against mode 5

The solver picks `(π, σ)` by earliest-finish-time list scheduling over the exact
runtime task DAG, the host materializes it through the EX-E1 contract, and this
directory measures it. R2 §6.3 raised the research gate from round one's "beat
L1" to **"beat mode 5"**, because mode 5 already passes the L1 gate (F-135); the
original wording is kept in `docs/TODO.md` §1.3 with the raise appended.

Reproduce: `bash docs/experiments/PLACE_EFT/run.sh` (sm_89, RTX 4090, 128 SMs,
`ctas_per_sm = 2`, `grid = 256`). `RUNS=25`, `CORRECTNESS_RUNS=50`,
`REALWIDTH=1`. `RAW_DIR` redirects the output tree, which is how
`run_sm120.sh` keeps a Blackwell run out of the committed sm_89 files.

## The candidates

Six arms per cell. Three are closed forms the host already had, selected by
`TILEMEGA_PLACEMENT` on the **committed round-one sources**, so H6's "modes 0 and
5 re-measured as control arms" re-measures the same binary recipe:

| arm | mode | what it is |
| --- | --- | --- |
| `legacy_grid_stride` | 0 | `π(s,t) = t mod grid`, queues stage-major (the L1-equivalent placement) |
| `balanced` | 4 | the affinity-first greedy (F-118), reference cells only |
| `rotate` | 5 | cross-stage continuous round robin (F-135), **the gate's denominator** |
| `eft` | 3 | this round's scheduler: EFT placement, σ = per-worker start-time order |
| `band` | 4→template | `w = (t / ceil(count/grid)) mod grid` (AFFINE_PROBE, F-93) |
| `wavefront` | 4→template | longest-path level, then rank inside the level (F-97) |

The last three are carried in the generated source as a materialized `(worker,
slot)` table and compiled with `-DTILEMEGA_PLACEMENT=0`, so the CTA-to-SM map is
the legacy one and **the Plan is the only difference**. The provenance diff runs
first and hard-exits: `tilemega-place-eft` emits its own `legacy_grid_stride`
source through its own import and codegen path, and that source is byte-identical
to the committed control on all four cells (`raw/provenance.tsv`), which is what
makes "only the Plan differs" a checked statement rather than an intention.

## How the schedule is computed

`lib/Solver/EftPlacement.cpp`, `ScheduleByEarliestFinish`:

* priority is the **upward rank in nanoseconds** — node weight
  `CostModel::TaskInstanceNs` at the cell's real coordinates, edge weight the
  §5.2 `hop_ns(N, R)` curve. Round one's `ListScheduler` ranked by edge-count
  height, which prices a 1.6 µs norm and a 18.3 µs projection the same;
* the worker is the one with **minimum earliest finish time**: a same-worker
  predecessor costs no synchronization, a cross-worker predecessor adds
  `hop_ns`, and two workers on one SM charge co-residency by resource vector;
* σ is the per-worker start-time order, so it **interleaves stages on purpose**
  — that is the freedom EX-E1 bought and the reason `BuildPlanQueues` orders a
  queue by σ alone;
* placement is resident-only (`ResidentScheduleLegal`), so the plan never names
  a worker the launch will not have.

Per R2 §6, **no artificial locality penalty was added**: this is pure EFT, and
what it does with locality is a measurement below, not a tuning knob.

⚠️ inferred, and worth naming because it is the one place the greedy and the
cost model disagree: the greedy prices a hop with `N` = the whole fan-out of the
producer and `R = 1`, the most pessimistic sharing; the simulator prices the same
edge with the exact cross-worker fan-out. On sm_89 that difference is worth
nothing measurable, because `c1` and `c2` are zero inside one standard error
(F-145) — but on a part where contention is real the two would diverge.

## What the run does

1. **provenance** — hard diff, above;
2. **compile** — six arms × four reference cells, `nvcc -arch=native -O2
   -DTILEMEGA_EVENT_KAPPA=1`;
3. **`raw/place_stats.txt`** — each binary reports its own `max_queue`,
   `same_worker_edges` and `cross_worker_edges` at launch;
4. **S2-a** — 50 fresh processes per arm-cell, `RESULT status=PASS` counted per
   process;
5. **S2-b** — 25 paired rounds, six arms rotating by `(round + slot) % 6`; the
   script hard-exits unless every arm contributes `runs` timed and `runs` correct
   processes;
6. **S2-d** — the four-arm synchronization decomposition (`neither`, `nowait`,
   `full`, `l1nosync`) on mode 5 and on the chosen plan, 8 combinations rotating
   by `(round + slot) % 8`. Only `full` is required to pass: the other three
   compile out a wait or a notify and are timing probes, not valid kernels;
7. **S2-c** — the real-width cell (`LAYERS=4 HIDDEN=4096 INTERMEDIATE=14336
   HEADS=32 KV_HEADS=8`), same rotation over five arms (`balanced` needs its own
   variant plan and is not a gate arm).

## Results

S2-a: **50/50 in all 34 arm-cells** (`raw/correctness.tsv`), SEQSCAN subset 12/12
at 50/50.

S2-b, the research gate, denominator mode 5 — **FAIL**, 0/4 cells:

| cell | eft | 95% CI | wavefront | band | balanced | mode 0 |
| --- | --- | --- | --- | --- | --- | --- |
| gqa2 s4 | **1.0096** | [1.0070, 1.0137] | 1.0172 | 1.5181 | 2.1469 | 1.5210 |
| mha4 s4 | **1.0141** | [1.0106, 1.0212] | 1.0265 | 1.5345 | 2.0867 | 1.5323 |
| gqa2 s128 | **1.0022** | [1.0017, 1.0026] | 1.0022 | 1.3624 | 2.2280 | 1.3445 |
| mha4 s128 | **1.0273** | [1.0263, 1.0275] | 1.0024 | 1.3533 | 4.2826 | 1.3377 |
| pooled, 100 pairs | **1.0111** | [1.0086, 1.0147] | 1.0105 | 1.4867 | 2.2080 | 1.4704 |

Every interval is entirely above 1 and every Wilcoxon p is ≤ 8.4e-05. Per H7 the
gate is recorded as failed and not moved; per R2 §6.3 it is **not** restated as
"faster than L1", a gate EFT does pass at 0.7276–0.8537.

S2-c, real width, same 25-round rotation:

| cell | eft | 95% CI | wavefront | band | mode 0 | eft / L1 |
| --- | --- | --- | --- | --- | --- | --- |
| real s4 | 1.0273 | [1.0269, 1.0281] | 1.0017 | 1.3017 | 1.3024 | 0.8419 |
| real s128 | 1.0487 | [1.0483, 1.0493] | 0.9987 | 1.1993 | 1.1959 | 0.9711 |

`wavefront` at real s128 is the one arm whose median falls below mode 5. Its
bootstrap interval excludes 1 but its Wilcoxon p is 0.063, so it is reported as a
tie, and it is not the plan the solver chose in any case.

S2-d, where the time goes — medians in ms, 25 rounds each:

| cell | arm | `neither` | `nowait` | `full` | `l1nosync` |
| --- | --- | --- | --- | --- | --- |
| gqa2 s4 | mode 5 | 0.0584 | 0.0612 | 0.2917 | 0.2918 |
| gqa2 s4 | eft | 0.2376 | 0.2764 | 0.2929 | 0.2929 |
| mha4 s4 | mode 5 | 0.1355 | 0.1433 | 0.5794 | 0.5795 |
| mha4 s4 | eft | 0.4700 | 0.5541 | 0.5880 | 0.5868 |
| gqa2 s128 | mode 5 | 0.1577 | 0.1874 | 0.4557 | 0.4557 |
| gqa2 s128 | eft | 0.3807 | 0.4321 | 0.4567 | 0.4568 |
| mha4 s128 | mode 5 | 0.2661 | 0.3697 | 0.9114 | 0.8558 |
| mha4 s128 | eft | 0.6975 | 0.9110 | 0.8787 | 0.8786 |

Read across: mode 5's placement is 2.4–4.1× better with synchronization removed,
EFT's synchronization is 3.6–4.2× cheaper, and the two cancel. F-148 works this
through.

S2-e, the simulator against this candidate set: per-cell Spearman +0.824 /
+0.794 / +0.812 / +0.928 (`raw/predicted.tsv` against the measured medians). It
ranks the families correctly and calls the eft/mode-5 near-tie the wrong way,
which is the honest limit of a model whose own start-time bias (F-146) is larger
than the gap being called.

## Files

| path | what |
| --- | --- |
| `place_eft.cpp` | the driver: imports each cell, runs the six candidates, writes `raw/plan/*.cu`, `raw/predicted.tsv`, `raw/eft_schedule.tsv` |
| `run.sh` | the measurement above; `RAW_DIR`, `RUNS`, `CORRECTNESS_RUNS`, `REALWIDTH`, `SKIP_GENERATE` |
| `run_sm120.sh` | the Blackwell runner (R2 §10). Written and CPU-self-checked here, **never run on a Blackwell part** |
| `summarize.py` | paired ratios, bootstrap CI (seed 20260912, 20000 draws), Wilcoxon |
| `collect_samples.py` | one row per process from the run logs into `raw/samples.tsv`, because `.gitignore:32` excludes `docs/experiments/**/*.log` |
| `verify.py` | R2 §11: re-checks every round-two gate from raw evidence only |
| `raw/provenance.tsv` | the four legacy-source byte diffs |
| `raw/place_stats.txt` | per-arm queue statistics as the host computed them |
| `raw/predicted.tsv` | the simulator's prediction for all six candidates × six cells |
| `raw/correctness.tsv`, `raw/samples.tsv`, `raw/summary.tsv` | S2-a counts, per-process timings, the paired tables above |

The binaries (`raw/bin`, 141 MiB), the runner logs (`raw/log`, `raw/final`, both
gitignored) and the fixtures they need (`SEQSCAN/raw/fixture`, 64 MiB for the
reference cells; `REALMODEL/raw/work/r2sim_s*`, 3.3 GiB per seq) are **not
committed**, which is why `raw/samples.tsv` exists and why `verify.py` falls back
to it.
