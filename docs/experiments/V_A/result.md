# V-A — Cross-block event synchronization

Status: ✅ verified on an NVIDIA GeForce RTX 4090 (SM 8.9), CUDA 12.8.93.

## Method

Each cell below ran in 50 fresh processes. The kernel used 256 threads, four
iterations, 1,024 floats per producer tile, and a four-producer fan-in except
for A-5 (one exclusive producer). Runtime occupancy was six CTAs/SM on 128 SMs,
so the largest grid (512 CTAs) remained below the measured resident limit of
768 CTAs. The raw machine-readable output is in `matrix_raw.txt`.

| Variant | SM/4 (32) | SM/2 (64) | SM (128) | 2×SM (256) | 4×SM (512) |
|---|---:|---:|---:|---:|---:|
| A-1 elementwise, alternating fill | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| A-2 cross-warp reduction, constant fill | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| A-3 cross-warp reduction, alternating fill | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| A-4 multiple consumers share producers | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| A-5 exclusive producer per consumer | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |

No cell failed or hung (1,250/1,250 fresh-process runs).

## Negative controls and fill masking

At grid=128, each control was run 50 times per fill mode:

| Deliberate change | Constant fill | Alternating fill | Interpretation |
|---|---:|---:|---|
| no wait/event | 46 mismatches | 47 mismatches | checker detects the race; constant fill hid 4/50 versus 3/50 |
| omit post-wait CTA barrier | 50 mismatches | 50 mismatches | the barrier is load-bearing for cross-warp consumers |
| omit thread fences | 0 mismatches | 0 mismatches | failure not observed; this does **not** establish portability or correctness |
| all threads poll | 0 mismatches | 0 mismatches | failure not observed; this is inefficient contention, not a proven negative control |

Alternating fill slightly improved detection for the pure-race control in this
sample, but did not reveal a qualitatively hidden failure. The earlier 20-run
probe similarly caught 19/20 alternating versus 16/20 constant runs. Therefore
the verifier remains justified as protection against stale-allocation masking,
but this experiment does not claim a large measured effect size.

## SASS and resource evidence

`ptxas.log` reports 38 registers for the correct reduction specialization,
one barrier, no spills, and no static shared memory (the launch supplies 32
bytes of dynamic shared memory for eight warp partials). `sass_report.txt`
contains backward `BRA` instructions for loops, `BAR.SYNC`, `MEMBAR`, global
atomics, and eight emitted `NANOSLEEP` occurrences across the compiled
specializations. Branch matching accepts instruction suffixes and imposes no
back-edge distance limit.

## Compute Sanitizer

- ✅ `compute-sanitizer --tool synccheck`: 0 errors.
- ✅ `compute-sanitizer --tool racecheck`: 0 hazards, errors, or warnings.

Both used A-3 at grid=32; full logs are `synccheck.log` and `racecheck.log`.

## Conclusion

The global event protocol (per-thread release fences, CTA convergence, one
signaling thread, single-thread backed-off polling, CTA convergence, acquire
fence) passed the requested matrix including cross-warp reductions. This
supports continuing the megakernel work on SM 8.9. It does not remove the
co-residency requirement: grids above the measured resident capacity can
deadlock when a resident consumer waits for a producer CTA that cannot launch.
