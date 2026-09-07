# Queue-era epoch control

Reproduce with `RUNS=50 bash run.sh`. Each process launches the queue-driven L2
kernel 32 times over persistent event storage. The `reset` build enables
`TILEMEGA_NEGATIVE_RESET_EVENTS=1`, clears counters between repeats, and
announces iteration zero after the first repeat.

Evidence status: ✅ RTX 4090 (`sm_89`), BF16, κ=1, 2026-09-07. The four arms
comprise 200 fresh processes. Raw counts and timing reductions are in
`raw_taskqueue/iterations.tsv` and `raw_taskqueue/timing.tsv`.

## Result

| model | arm | processes | pass | total mismatch | timeout |
|---|---|---:|---:|---:|---:|
| gqa2 | monotone epoch | 50 | 50 | 0 | 0 |
| gqa2 | reset / iteration zero | 50 | **50** | 0 | 0 |
| mha4 | monotone epoch | 50 | 50 | 0 | 0 |
| mha4 | reset / iteration zero | 50 | **50** | 0 | 0 |

❌ The required negative control did not fail. This is recorded as a failed
acceptance, not converted into evidence for the epoch mechanism.

The reason is structural. `iteration` is still a host launch parameter, every
launch uses the default stream, and `LaunchL2` synchronizes before returning.
No CTA from iteration `i` can coexist with one from `i+1`; clearing an event
after the former has completed cannot create ABA. Fifty or fifty thousand
processes cannot exercise an impossible interleaving.

This does not contradict the queue overlap measurement. Across 50 traces,
mean early starts are 68.72/200 gqa2 and 205.70/512 mha4 tasks. That proves the
stage barrier was removed. It says nothing about overlap between separately
synchronized launches.

## Timing under repeated launches

| model | arm | single L2/L1 | final repeated L2/L1 |
|---|---|---:|---:|
| gqa2 | monotone | 1.091393 | 1.083433 |
| gqa2 | reset | 1.095543 | 1.083825 |
| mha4 | monotone | 1.098425 | 1.094005 |
| mha4 | reset | 1.095564 | 1.088992 |

The repeated-launch ratios remain in the same 1.08–1.10 regime as a single
forward; there is no persistent-kernel cross-iteration reuse to change it.
These are medians from the correctness processes, not a separately interleaved
performance claim.

## Missing acceptance surface

⚠️ This harness is also not 32-step greedy autoregressive generation with a
growing `past`. It has one fixed `seq=1,past=0` fixture and compares each repeat
to its first output. The requested test needs all of the following together:

- a device-side iteration loop so workers can wrap independently;
- versioned/interleaved activation storage so iteration `i+1` cannot overwrite
  data still consumed by `i`;
- per-layer KV cache rotation into a growing maximum-size allocation;
- logits/token selection and a PyTorch reference for every step.

Adding only the device loop would manufacture buffer WAR/WAW races unrelated
to epochs; adding an iteration barrier would make the negative unreachable
again. Therefore no such pseudo-test is substituted. The monotone formula is
implemented, but its ABA necessity remains ⚠️ unverified until this serving
lifetime exists.
