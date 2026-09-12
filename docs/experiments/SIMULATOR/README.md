# SIMULATOR — EX-S1 polling-contention calibration (sm_89)

`hop_ns(N, R)`, the edge-weight function the EX-S1 execution simulator prices a
cross-worker dependency with. `N` is the number of CTAs polling, `R` the number
of distinct event rows they poll, so `N/R` is how many CTAs share one 128-byte
line — the regime a placement decides.

Reproduce: `bash docs/experiments/SIMULATOR/run.sh` (sm_89, RTX 4090, 128 SMs,
`resident_ctas = 3072` for a 32-thread kernel). `ROUNDS=4096`, `GUARD=20000`.

## Why this benchmark exists

Round one's `headroom.py` priced every hop at the 1024 ns `%globaltimer` tick
and modelled no contention at all, and it underestimated mode 5 by 1.7–3.7×
(F-139). R2 §0 item four hypothesised that the missing factor is polling
contention: many workers hammering few event lines. This measures that directly,
with the primitives the runtime actually executes — `EventPoll` from
`include/tilemega/Codegen/tasks/EventSync.cuh` and `atomicExch` on a
`EventCounter`-shaped padded row, both lifted from `ModelHarness.cuh`.

## The answer: no, contention is not the missing factor

| arm | `c0` (ns) | `c1` per doubling of `N/R` | `c2` per doubling of `R` | sweep range / `c0` |
| --- | --- | --- | --- | --- |
| **RMW + `__nanosleep(64)`** (the generated wait, **this is the calibration**) | 1235.4 ± 17.2 | **−0.40 ± 2.41** | −2.67 ± 2.36 | 3.1 % |
| RMW, backoff removed | 325.0 ± 18.7 | 4.04 ± 2.62 | 1.81 ± 2.57 | 16.0 % |
| load poll + `__nanosleep(64)` | 1227.2 ± 17.5 | 0.75 ± 2.44 | −1.20 ± 2.39 | 3.1 % |
| load poll, backoff removed | 295.5 ± 16.4 | 8.43 ± 2.27 | 2.24 ± 2.17 | 23.2 % |

Fit form `hop_ns(N, R) = c0 + c1·log2(1 + N/R) + c2·log2(R)`, weighted least
squares over 24 cells (`N ∈ {1,2,4,8,16,32,64,128,256}` × `R ∈ {1,4,16,64}`,
`N ≥ R`), 4096 rounds each, errors scaled by the reduced chi-square. Full output
in `hop_fit.txt`; the emitted coefficients in `hop_ns.tsv`.

- ✅ verified — **in the arm the runtime actually runs, both contention
  coefficients are zero inside one standard error.** Across a 256-fold range of
  line sharing and a 64-fold range of live rows the hop costs 1211–1249 ns.
  Polling contention therefore does not by itself account for round one's
  1.7–3.7× underestimate of mode 5. R2 §0 item four is **not supported** by this
  measurement; it is recorded as a negative result and the gates are not moved.
- ✅ verified — with the backoff removed, contention becomes measurable and
  stays small: the single-consumer hop is 4.04 ± 2.62 ns per doubling of sharing,
  and the *fan-out* figure (slowest of the `N/R` consumers on one row) is
  12.2 ± 4.6 ns per doubling — about 100 ns over the whole 256× range.
- ✅ verified — **910 ns of the default 1235 ns hop is the `__nanosleep(64)`
  backoff granularity** (1235.4 − 325.0), not coherence traffic and not
  contention. That is a sync-protocol observation, and R2 §1 excludes protocol
  changes (EX-E3 is measure-only this round), so it is recorded here and carried
  to the next-round priority discussion, not acted on.
- ✅ verified — load polling is ~30 ns (9 %) cheaper than RMW once the backoff is
  out of the way, and indistinguishable from it with the backoff in
  (1227 vs 1235, overlapping errors). Recorded as comparison data only:
  `TILEMEGA_EVENT_LOAD_POLL` default stays 0.
- ✅ verified — `inversions = 0` in all 96 cells across both arms and both poll
  modes: no observe timestamp ever precedes its publish.

The simulator uses the first row. `c1` and `c2` are kept in the exported curve
even though they are zero within error, because the simulator's edge weight is
defined as a function of `(N, R)` and pinning them to exactly zero would bake
"contention is free" into the model rather than into the data.

## Measurement method, and two artifacts that had to be fixed

`%globaltimer` ticks every 1024 ns on sm_89 (`../TRACE_V2/resolution.md`). It is
one counter broadcast to every SM and read bit-identically, so a
publish→observe difference carries **no per-SM offset** — only two zero-mean
quantizations. No `clock64` mapping is needed or done here. Averaged over `K`
dithered rounds the standard error converges as `1024·sqrt(2/12)/sqrt(K)`, which
is 6.5 ns at `K = 4096` per consumer.

Two artifacts were found by measurement and are recorded rather than papered
over:

1. ✅ verified — with a fixed guard spin the sweep returned **exactly 1024.0 ns,
   `se = 0.00`, in all 24 cells**: the round loop was perfectly periodic, so the
   publish always landed at the same phase inside the tick and the difference
   was always exactly one tick. Fixed with a per-round phase dither of
   `(r · 2654435761) % 4096` cycles — longer than a tick at any plausible clock.
   The dither depends on the round alone, so all `R` publishers still release
   together: `R` rows contending at the same instant is the regime mode 5 puts
   the device in, and staggering the publishers would measure a gentler one.
2. ✅ verified — one process of the load-poll arm recorded a single ~2.3 ms hop
   in *every* cell while `p50`/`p90` stayed at one and two ticks: a device-level
   stall, not a hop, and 4096 rounds are too few for it to wash out of a 1.2 µs
   mean (untrimmed residual RMS 310 ns against 15 ns trimmed). The table now
   carries both `hop_mean_ns` and a 0.1 %-trimmed `hop_trim_mean_ns`; the fit
   uses the trimmed column and nothing is dropped from the table. The RMW arm
   showed the same thing once, milder (`hop_max_ns = 142336`).

`last_*` is the slowest consumer **of one row**, not the max across all rows:
taking the max across rows would mix independent publishes and measure the
dither instead of contention. `hop_*` is one consumer seeing one publish, which
is what a single dependency edge costs.

## Files

| file | what it is |
| --- | --- |
| `contention.cu` | the microbenchmark; `Row` is padded to 128 B exactly as `EventCounter` is, so `R` really is a count of lines |
| `run.sh` | builds both poll modes, runs both sweeps, fits, hashes |
| `contention.tsv` | default RMW poll, 48 rows (24 cells × backoff {64, 0}) |
| `contention_load.tsv` | `TILEMEGA_EVENT_LOAD_POLL=1`, comparison only |
| `hop_fit.py` | the weighted fit; also prints the untrimmed fit beside the trimmed one |
| `hop_fit.txt` | its output for the committed tables |
| `hop_ns.tsv` | the three coefficients the simulator reads |
| `sha256.txt` | hashes of the three data files |

Calibration/evaluation split (H8): this curve and the round-one gqa2 seq {4,128}
mode 0/5 dumps are the **calibration** set. mha4, real width, and every new
candidate are the **evaluation** set and are not looked at until the simulator is
frozen.

---

# SIMULATOR — EX-S1 execution simulator (the L2 cost model)

`lib/Solver/ExecutionSimulator.cpp` replays §5.7.2 executor semantics: each
worker walks `plan.queue[w]` in σ order, one task at a time (W = 1), and a task
starts at `max(worker free, every predecessor's end + hop)` with the hop zero
for a same-worker predecessor. That last clause is the entire reason this
exists — round one priced a schedule as `max(work_lb, queue_lb, critical_path)`
and underestimated mode 5 by 1.7–3.7× (F-139), because none of those three
bounds can express a worker idling at a not-yet-ready head while a task it
could have run waits behind it.

Reproduce: `bash docs/experiments/SIMULATOR/run_s1.sh` (sm_89). It is a separate
script from `run.sh` on purpose — `run.sh` produces the calibration set, this
one the evaluation set, so H8's disjointness is a property of the layout rather
than a claim in a paragraph.

## Where every input comes from

The point of the driver (`simulate.cpp`) is that nothing under simulation is a
re-derivation of what ran:

| input | source | how it is checked |
| --- | --- | --- |
| task DAG | `kDependencies0` parsed out of the generated `.cu` — literally the array `ModelHarness.cuh:1434` feeds the harness | node count against `E2E_SCHEDULE task_refs` |
| per-stage active counts, stage order | the `E2E_PLACE_BASE` lines a mode 5 run dumps | hard-fails if absent |
| (π, σ) | `MaterializePlanPlacement`, the one routine the host also calls | `materialize_check.tsv`: `same_worker_edges`, `cross_worker_edges` and `max_queue` against each run's own `E2E_PLACE_STATS` |
| worker → SM | `smid` in the trace's `slots.tsv` | a worker seen on two SMs is a hard error, since the launch is meant to be persistent |
| node durations | `CostModel::TaskInstanceNs` at the cell's real coordinates | `stage_price.tsv` records which stages fell back to `TaskStageNs` |

Because all three placement stats reproduce for modes 0, 4 and 5, the DAG and
the placement under simulation are provably the ones that ran, and a new
candidate is materialized on the same footing rather than on a parallel code
path.

Durations are priced at `active_ctas_per_sm = 1`, i.e. solo, and concurrency is
then applied by the co-residency model. Baking residency into the price would
charge a task that happens to run alone on its SM for a co-resident that is
idle.

## Two modelling choices that are arms, not assumptions

`proportional_sharing` stretches a co-resident set by its size; the default
stretches it by the aggregate §4.4.1 nine-lane demand of the set. **The real
models run the `proportional` arm**, and this is a limitation, not a preference:
only GEMM stages expose a `ResourceVector`, so a mixed zero/non-zero lane set
would make `LaneStretch` return 1 and systematically under-estimate. The lane
model is unit-tested and carried; it is not what the reported numbers use.

`flat_hop` prices every hop at `c0` alone. Since c1 and c2 are zero inside one
standard error on sm_89, the two arms should agree, and the pair is reported so
that "contention is free" stays a measurement rather than a modelling choice
baked into one number.

## Calibration/evaluation split (H8), restated for this half

The simulator was frozen — no coefficient, threshold or structural change — on
evidence from the **calibration** set alone: the `hop_ns` curve above and the
round-one gqa2 seq {4,128} mode 0/5 dumps. Two changes were made during that
period and both are on calibration data: an explicit per-worker `busy` flag,
after gqa2 seq 128 produced a `busiest_worker_ns` above its own makespan, and a
branchless edge-classification sweep for speed. mha4, seq 512, real width and
every new candidate are the **evaluation** set. No per-cell coefficient is
fitted anywhere; there is no per-cell coefficient to fit.

## Files

| file | what it is |
| --- | --- |
| `run_s1.sh` | builds, traces 2 models × seq {4,128,512} × modes {0,4,5}, times the untraced arm, simulates, reports |
| `simulate.cpp` | the driver; `tilemega-simulate REPO MANIFEST OUT_DIR` |
| `s1_report.py` | S1-a/S1-b/S1-c from the raw artifacts only |
| `s1_report.txt` | its output for the committed tables |
| `predicted.tsv` | every candidate × arm: makespan, queue bound, block time, critical path, eval time |
| `s1_start_error.tsv` | S1-a per-task start error per cell |
| `s1_ranking.tsv` | S1-b predicted against measured, per placement × config cell |

## Verdicts

| gate | verdict | number | evidence |
| --- | --- | --- | --- |
| S1-a agreement | reported, no gate | per-task start error over 18 cells; as a fraction of each cell's own span, \|p50\| 11.6–40.5 %, \|p90\| 19.7–65.2 %, \|max\| 23.0–81.5 %; the sign is negative in every cell | `s1_report.txt`, `s1_start_error.tsv` |
| S1-b ranking | **PASS** | pooled Spearman 0.880, within-config 0.973, argmin 6/6, predicted top-3 contains the measured top 3 % | `s1_ranking.tsv` |
| S1-c speed | **FAIL**, and trips §9 | median over the three candidates: 53 µs / 96 µs at seq 4, but 3.24 ms and 11.07 ms at seq 128 and 33.6 ms / 140.3 ms at seq 512, against < 1 ms; real width 0.70 ms at seq 4 and 91.2 ms at seq 128, against < 10 ms | `predicted.tsv` column `eval_us`, `s1_report.txt` |
| S1-d contention | PASS | `hop_ns.tsv`, N to 256 and R to 64 | `contention.tsv` |
| S1-e honesty | PASS | the split is stated above and restated per half; no per-cell coefficient exists to fit | this file |

S1-a's error is one-sided: the simulator predicts every task starting earlier
than it did, by a roughly fixed fraction of the cell's span. A bias of that
shape cancels in a ranking, which is what S1-b measures and why the hard gate
passes while the absolute error is large. It does not cancel in an absolute
makespan, so the simulator's output must be read as an ordering, never as a
predicted runtime.

Modes 0 and 5 are separated: `rotate` is predicted fastest in all six configs
and measured fastest in all six, and the predicted ratio between them
(0.654–0.813) has the same sign as the measured one (0.652–0.924). The regime
round one could not tell apart is the one this model orders correctly.

## S1-c fails by more than one order of magnitude, which is §9

`gqa2` seq 512 evaluates a single Plan in 33.6 ms and `mha4` seq 512 in
140.3 ms (median of the three materializable candidates; worst candidate
238.0 ms), against the gate's < 1 ms for reference models; real width seq 128
takes 91.2 ms against < 10 ms. That is over budget by more than one order of
magnitude, so it is the fourth §9 stop condition — "the algorithm was chosen
wrong, report first" — and it is reported as such rather than argued down. The
gate was fixed before implementation and is **not moved** (H7).

An earlier draft of this file recorded S1-c as failing "at 2.9×, not 29×, so it
is not the §9 stop condition". That was written when the evaluation set stopped
at seq 128 for the reference models; adding seq 512 and real width made it
false, and it is corrected here rather than deleted.

Two families of artifact are on disk but not in the commit, both by the
repository's own `.gitignore`, which H1 does not allow this round to touch:
`raw/bin/` (34 MB of compiled arms, `build-*/` rule, the same as
`PLACE_ROTATE/raw/bin`) and the nvcc transcripts `raw/log/*.log`
(`docs/experiments/**/*.log`). `raw/log/{build,simulate}.txt`, the per-cell
stdouts in `raw/run/`, the traced dumps and every TSV are committed, so nothing
`verify.py` re-reads is missing.

`eval_us` is wall time on the host, so unlike every other column it is not
reproducible to the digit: the numbers above are the ones in the committed
`predicted.tsv`, and an earlier draft of this table quoted a previous run's
(34.5 ms and 157.8 ms at seq 512, 97.8 ms at real width seq 128). Both runs are
over budget by the same order of magnitude and the verdict does not depend on
which is quoted; the older figures are recorded here rather than dropped.

The cause is measured, not guessed. Phase split on gqa2 seq 128: in-degree
initialisation ≈ 1.3 ms, the event loop ≈ 1.9 ms, the critical-path tail
≈ 0.53 ms. The tail is therefore not the lever. Both sweeps are per-edge, and
`edge_budget.py` shows the DAG carries 548,876 edges at that cell — exactly the
count the simulator reports — of which **530,432 (96.6 %) are the expansion of
`kAll` maps**, an all-to-all between two stages that means "every consumer waits
for the whole producer stage". Cost is linear in edges, and the edge count is
what grows: 548,876 at gqa2 seq 128 against 8,560,688 at gqa2 seq 512 and
34,109,552 at mha4 seq 512. The measured times track that ratio, so the failure
is the representation of the DAG and not the event loop.

Two degradations, in preference order:

1. **Barrier-node compression.** One pseudo-node per `kAll` group turns `n·m`
   edges into `n+m`: 548,876 → 23,580 at gqa2 seq 128, a 23× reduction on a
   cost that is linear in edges, and more at seq 512 where the groups are
   larger. Not done here, for two reasons worth stating rather than hiding:
   `MaterializeRuntimeTaskGraph` is shared with the host and its expanded edge
   count is what `E2E_PLACE_STATS` reports and E1-b pins byte for byte (H2), so
   the compression has to be simulator-local; and doing it now would be a
   structural change to a simulator already frozen for evaluation (H8), after
   the evaluation numbers had been read. It is the first item of next round.
2. **Coarse ranking then measurement** — simulate to order the candidates,
   measure the top-k. S1-b, not S1-c, is what the solver actually needs, and at
   the candidate counts in play (six) even 158 ms per Plan is usable offline;
   what it rules out is evaluating a Plan inside a host-side search.

A branchless edge-classification sweep was applied and is worth 26 % (3.93 ms →
2.89 ms on gqa2 seq 128) with byte-identical makespans and critical paths. That
is a constant factor on the wrong side of a structural one, which is why it is
reported as a partial mitigation and not as a fix.

## Two behaviour-preserving moves out of the frozen simulator

`HopCurve` moved to `include/tilemega/Solver/HopCurve.h` and `LaneStretch` to
`include/tilemega/Solver/CoResidency.h`, so that the simulator that prices a
plan and the EFT scheduler that chooses one cannot price a hop or a shared SM
differently. Both moves are textual. They happened after the evaluation numbers
were read, so the evidence that they changed nothing is byte identity rather
than an argument: re-running the simulate tail against the same dumps reproduces
all 16 numeric columns of the 55 reference-cell rows of `predicted.tsv`
exactly, differing only in `eval_us`, which is wall time.

## One correction to the real-width arm

The manifest's real-width rows first pointed at `${work}/export.json`, which is
the synthetic shape summary REALMODEL writes, not the
`tilemega.exported_program.v1` bridge; the driver rejected it with "unsupported
export bridge schema" and `set -euo pipefail` aborted the script before the
report step. The bridge artifact is `${work}/model.json`, which is also what
`REALMODEL/run.sh` compiles from. `SKIP_MEASURE=1` then re-ran the simulate and
report tail against the GPU artifacts already on disk; nothing was re-measured
and nothing measured is reported as fresh.
