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

## V-A supplement

All supplement cells used fresh processes on the same RTX 4090. Statistical
correctness cells used 50 runs each. Raw records are the files prefixed with
`supplement_` in this directory.

### 1. no_fence under hostile layout

The hostile modes reuse one `G × tile` data buffer across four iterations, omit
spin backoff, and use 8,192-float tiles. A consumption counter prevents the next
round from overwriting a tile before all consumers finish the current round.

| Mode | grid=64 | grid=128 | grid=256 |
|---|---:|---:|---:|
| `correct_hostile` | 0/50 mismatches | 0/50 | 0/50 |
| `no_fence_hostile` | 0/50 mismatches | 0/50 | 0/50 |

✅ The combined hostile no-fence construction did not trigger a failure. As
required, each hostile change was then isolated on top of `no_fence`:

| Isolated change | grid=64 | grid=128 | grid=256 |
|---|---:|---:|---:|
| reuse the data buffer only | **50/50 mismatches** | **50/50** | **50/50** |
| remove spin backoff only | 0/50 | 0/50 | 0/50 |
| increase tile to 8,192 only | 0/50 | 0/50 | 0/50 |
| correct fences + reuse only (control) | 0/50 | 0/50 | 0/50 |

The first mismatch at every grid read the exact previous-iteration reduction,
not a partial sum. For example, at grid=128 the current expected value was
525,674, the previous value was 523,695, and the observed value was 523,695.
The other samples are recorded in `supplement_mismatch_samples.txt`.

Thus address reuse was the critical single change. The combined three-change
variant passing shows a timing interaction: the larger cooperative write and
CTA barrier allowed stores to become observable before publication on this
GPU even without explicit fences. It does not weaken the need for §8.5.

### 2. barrier inside divergent spin loop

| Mode | grid=64 | grid=128 | grid=256 |
|---|---:|---:|---:|
| `barrier_in_spin` | 50/50 hangs | 50/50 hangs | 50/50 hangs |

✅ No silent mismatch or successful completion occurred. `hang_probe.sh`
sampled changing CUDA PC distributions and 6–27% SM utilization while the
kernel remained unable to complete. This is classified as a collective stall
or livelock, rather than a quiescent deadlock. The concise evidence is in
`supplement_hang_probe.txt`.

### 3. all-thread polling cost

CUDA events bracketed only the kernel. Each median is from 20 fresh processes
with the same reduce/circular workload and 1,024-float tile.

| grid | correct median | all-thread median | all-thread / correct |
|---:|---:|---:|---:|
| 64 | 0.035328 ms | 0.035200 ms | 0.996× |
| 128 | 0.035840 ms | 0.034816 ms | 0.971× |
| 256 | 0.037888 ms | 0.035840 ms | 0.946× |

⚠️ This workload showed no measurable all-thread slowdown; the small apparent
speedups are treated as timing noise, not a performance benefit. All CTAs
publish before waiting and the spin window is extremely short, so this test
does not create sustained atomic contention. The semantic conclusion remains:
all-thread polling outside a collective is benign here, while single-thread
polling is the conservative code-generation rule for avoiding unnecessary
atomic traffic in workloads with real wait time.

### 4. Co-residency boundary

`tools/tilemega-occupancy` loaded the correct-reduction function from the cubin
and reported:

```text
block_size=256 registers_per_thread=40 static_smem_bytes=0
dynamic_smem_bytes=32 ctas_per_sm=6 num_sms=128 resident_limit=768
```

| grid relative to `resident_limit` | Concrete grid | Result |
|---|---:|---:|
| `limit − num_sms` | 640 | 50/50 pass |
| `limit − 1` | 767 | 50/50 pass |
| `limit` | 768 | 50/50 pass |
| `limit + 1` | 769 | 50/50 pass |
| `limit + num_sms` | 896 | 50/50 pass |
| `2 × limit` | 1536 | 50/50 pass |

❌ The predicted hang boundary was not observed. The occupancy result agrees
with the runtime API (six CTA/SM), but the current `kCircular` relation only
waits on the next four block indices. Most resident CTAs can complete and free
slots, allowing later indices to launch; it does not require the whole grid to
be resident simultaneously. Consequently this scan validates the occupancy
calculation but cannot validate a full-grid co-residency boundary. No boundary
hang existed to feed to `hang_probe.sh`.

The portable capacity formula is:

```text
resident_limit(kernel, target) =
    TargetSpec::Probe().res.num_sms
  × active_ctas_per_sm(kernel metadata, block_size, dynamic_smem_bytes)
```

This is a resource upper bound, not by itself a necessary grid-size bound. A
schedule requires `grid ≤ resident_limit` only when its wait-for relation can
leave every resident CTA waiting exclusively on not-yet-launched CTAs.

### 5. Impact on skeleton §8

- §8.1: strongly supported for barrier placement—putting `__syncthreads()` in
  the divergent spin loop hung 150/150 runs. All-thread polling alone remained
  correct; its atomic-contention cost was not measurable in this short-wait
  workload.
- §8.3: removing backoff alone produced no correctness failure. Backoff remains
  a performance/contention rule, not a correctness fence.
- §8.5: supported by the reuse-only test, which read the exact prior iteration
  in 150/150 no-fence runs while the fenced control passed 150/150.
- §8.7: the metadata-based formula correctly computes resource capacity, but
  the existing circular test does not prove that full-grid residency is
  necessary. Co-residency must be checked against the generated wait-for graph,
  not inferred from `grid` alone.
