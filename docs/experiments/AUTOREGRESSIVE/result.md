# Part 5 — multi-iteration coverage of §8.2's monotone counters

Reproduce:

```
TILEMEGA_ITERATIONS=32 ./model <fixture>          # the sweep
nvcc ... -DTILEMEGA_NEGATIVE_RESET_EVENTS=1       # the negative control
```

Evidence status: ✅ measured on an RTX 4090 (sm_89), BF16, structured
ownership, 2026-09-06.

## What was asked and what is delivered

The brief asks for N-step autoregressive generation with `seq = 1`, a `past`
that grows and a KV cache that accumulates, compared against PyTorch token by
token. **That is not what this experiment delivers, and the difference is
stated up front rather than buried.** What is delivered is the property the
epoch mechanism exists for — N launches over event memory that is never
cleared — plus the negative control the brief asks for, and the negative
control produced a result that changes what the whole item means.

`TILEMEGA_ITERATIONS=N` runs the persistent L2 kernel N times with
`ResetBuffersOnly` between launches, so the buffers are restored and the event
counters are carried over, each launch announcing its own `iteration`. Every
launch's output is compared against launch 0's.

✅ 32 iterations, 50 fresh processes per model (`raw/iterations.tsv`):

| model | iterations | processes | passes | total mismatch across all repeats | timeouts |
|---|---:|---:|---:|---:|---:|
| gqa2 | 32 | 50 | **50** | **0** | 0 |
| mha4 | 32 | 50 | **50** | **0** | 0 |

Each process compares all 32 launches against launch 0 element by element, so
that is 1600 launch-to-launch comparisons per model with no event memory ever
cleared.

## The negative control does not fail, and that is the finding

The control is the shape a naive implementation takes: clear the event
counters between iterations and always announce iteration 0
(`TILEMEGA_NEGATIVE_RESET_EVENTS=1`). §8.2's argument is that a CTA still
finishing iteration `i` would then satisfy iteration `i+1`'s wait from a
cleared counter — the ABA the monotone target prevents.

✅ Measured, same protocol, and it **passes**:

| model | arm | processes | passes | total mismatch | timeouts |
|---|---|---:|---:|---:|---:|
| gqa2 | monotone target (shipped) | 50 | 50 | 0 | 0 |
| gqa2 | **counters reset, iteration 0** | 50 | **50** | **0** | 0 |
| mha4 | monotone target (shipped) | 50 | 50 | 0 | 0 |
| mha4 | **counters reset, iteration 0** | 50 | **50** | **0** | 0 |

❌ **The hazard is unreachable in this harness, so the sweep provides no
coverage of it.** The reason is structural, not statistical: `iteration` is a
*launch* parameter, and every launch is issued on the default stream and
completes before the next one is issued. There is never a CTA still finishing
iteration `i` while another begins `i+1`, so no amount of repetition can
produce the interleaving the counters defend against.

For the hazard to be reachable one of two things has to be true, and neither is
today:

* the iteration loop is **inside** the kernel — a persistent kernel that
  generates token after token without returning to the host, which is what
  Phase 6's continuous batching is; or
* two iterations are **concurrent** — separate streams sharing the event
  memory, which nothing in the harness sets up.

So `needed = triggers × (iteration + 1)` is correct-by-construction defensive
code whose necessity begins at Phase 6. Calling it "tested" because a
32-iteration sweep passes would be exactly the mistake §7's standing condition
warns about: a green matrix that cannot observe the failure it is supposed to
observe. ⚠️ It is recorded here as **untested**, with the specific reason.

## Why growing-`past` autoregression is not here

It is not a matter of a loop: buffers are allocated at the fixture's dims
(`BufferDesc::Elements`), the KV cache's per-head stride *is* `dims.total`, and
`ResetBuffersOnly` re-lays the past cache into the full cache at that stride.
Growing `past` per step therefore needs allocation at the maximum length, a
per-step re-upload of `Params`, a device-to-device rotate of every layer's
`full_k`/`full_v` into the next step's `past_k`/`past_v`, and a fixture
carrying a PyTorch reference per step. That is a harness feature, not a test,
and it is the same feature Phase 6 needs — which is also where the device-side
iteration loop above belongs.

What the existing evidence already covers of that surface: SEQSCAN's matrix
runs `seq = 1` against `past ∈ {0, 3, 512}` for both models at 50 fresh
processes each, so the *shapes* an autoregressive step takes are exercised;
what is not exercised is their *sequence* inside one process.
