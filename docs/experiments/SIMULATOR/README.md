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
