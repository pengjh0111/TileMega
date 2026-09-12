# `%globaltimer` resolution and cross-SM agreement (EX-D1 §3.5)

Measured before any per-slot timestamp is interpreted, because the tick size
decides what a reconstructed per-hop latency can mean. Device: NVIDIA GeForce
RTX 4090, compute capability 8.9, driver 610.43.02, CUDA 12.8. Reproduce with
`bash docs/experiments/TRACE_V2/run_resolution.sh`; raw output in
`docs/experiments/TRACE_V2/raw/`.

## Method

`resolution.cu` runs two probes. The first is one thread reading
`%globaltimer` 20000 times back to back, recording `clock64()` beside each
read; the statistics below are over the 19999 adjacent differences. The second
launches 128 CTAs, has every CTA arrive at a software barrier, releases them
together and then has each read once — so the spread reflects clock skew
rather than the order the CTAs happened to start.

`calibrate.cu` answers the follow-up question §3.5 requires once the tick
exceeds 100 ns. 128 CTAs repeat the barrier 256 times, each round recording the
pair `(clock64, %globaltimer)`, with a spin between rounds that spreads the
anchors over 2.186 ms. Because `%globaltimer` is one counter broadcast to every
SM, it is a shared ruler, and a least-squares fit of globaltimer against
`clock64` per SM recovers that SM's rate and offset.

## Result

| Metric | Value |
|---|---|
| `%globaltimer` min non-zero delta | **1024 ns** |
| `%globaltimer` p50 / p99 / max delta | 1024 / 1024 / 1024 ns |
| back-to-back reads returning an unchanged value | 98.2949% (19658 / 19999) |
| backward steps | 0 |
| cross-SM spread, 128 CTAs on 128 distinct SMs | **0 ns** |
| `clock64` min non-zero delta | 40 cycles (≈ 16 ns) |
| `clock64` zero / backward deltas | 0 / 0 |
| `clock64` offset spread across SMs | 3.54 × 10⁹ cycles (≈ 1.41 s) |
| calibrated rate | 0.396814–0.396823 ns/cycle (2519.916–2520.071 MHz) |
| fit residual, RMS (worst SM) / max | 303.8 ns / 555.4 ns |
| predicted quantization σ, 1024/√12 | 295.6 ns |
| resulting per-SM offset uncertainty (1σ) | **18.99 ns** |

The tick is exactly 1024 ns: minimum, median, 99th percentile and maximum
adjacent delta are all the same number, and 98.3% of consecutive reads do not
move at all. The counter never went backwards.

The two timers fail in opposite ways. `%globaltimer` has **no per-SM offset** —
128 CTAs spread over 128 distinct SMs read a bit-identical value, and the same
holds for every anchor round in `calibration.tsv` — but each reading is
quantized to 1024 ns. `clock64` has the resolution but is per-SM: at one
instant its readings differ by 3.54 × 10⁹ cycles, about 1.41 s of accumulated
skew, and its rate follows DVFS (the idle SM clock here is 210 MHz against a
3105 MHz boost ceiling), so raw `clock64` values are not comparable across
workers at all.

## Per-SM offset estimation and its error bound

Fit `t_ns(clk) = t0_ns + a·(clk − clk0)` per SM against the common globaltimer
ruler, stored anchored rather than as a bare intercept because a globaltimer
reading is ≈ 1.8 × 10¹⁸ ns and a `double` intercept would lose exactly the
nanosecond digits at issue. `calibration.tsv` carries `a`, `clk0` and `t0_ns`
per SM.

The error bound is measured, not assumed, and it comes out where theory says it
should: the worst-SM residual RMS is 303.8 ns against a predicted 295.6 ns for
uniform quantization over a 1024 ns tick. The residual **is** the globaltimer
quantization; no drift, DVFS excursion or barrier-skew term is detectable on
top of it across the 2.186 ms window, and the 128 per-SM rates agree to 0.006%.
Averaging 256 anchors therefore leaves an offset uncertainty of
303.8/√256 = **18.99 ns (1σ)**, with the rate contributing under 0.01% of any
interval length.

## Consequence for trace v2

`needs_clock64_columns` is 1, so §3.5's conditional branch applies: the trace
records `clock64()` as well, `slots.tsv` gains `run_begin_clk` and
`run_end_clk`, and `TaskTraceV2` gains the two matching fields. This is an
amendment §3.5 mandates, not a departure from §3.1 — the field order §3.1 fixes
is preserved as a prefix and the two columns are appended.

Three consequences for how the §3.6 numbers may be read:

1. A single globaltimer timestamp carries ±512 ns; a latency built from two of
   them carries up to ±1024 ns and is quantized to multiples of 1024 ns. Where
   a per-hop latency is of the same order, the globaltimer figure reports the
   tick, not the hop.
2. Quantization by a floor is monotone, so `publish_ns ≤ ready` is preserved
   whenever it held in real time. Gate D1-e (`hop(j) < 0` must be 0) cannot be
   broken by the tick alone; a negative hop would indicate a real ordering
   problem.
3. A task's own duration, `run_end − run_begin`, is a same-SM subtraction and
   needs no calibration at all: at ≈ 16 ns effective resolution the `clock64`
   pair measures it roughly 64× more finely than the globaltimer pair.

§3.6 fixes the per-hop and bound definitions on the globaltimer fields, so
those remain the reported figures; the calibrated `clock64` columns are carried
alongside as the finer cross-check, and both are reported where they disagree.
