# Task-queue L2 attribution

Evidence status: ✅ measured on RTX 4090 (`sm_89`), BF16, 2026-09-07, after
logical-task events, kAll aggregation, and selective publication. No earlier
executor result is used.

`run.sh` rotates four arms over 25 fresh-process rounds. `full` is the shipped
queue, `nowait` keeps notify only, `neither` removes wait and notify, and
`l1nosync` removes L1's cooperative-grid barrier. Every full cell passes 25/25;
each unsafe cell reports a mismatch in 25/25.

For every paired round:

```
L2 = neither + notify + wait
L1 = l1nosync + barrier
L2 - L1 = loop + notify + wait - barrier
```

The maximum numerical closure error is **0.000000 µs**. This is an accounting
identity over unsafe probes, not a claim that those probes are valid kernels.

## Measured decomposition

All values are paired medians in milliseconds; `loop = neither-l1nosync`.

| model | seq | L1 | L2 | gap | wait | notify | barrier | loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gqa2 | 4 | 0.372736 | 0.412672 | 0.035648 | 0.029760 | 0.058112 | 0.039456 | −0.011104 |
| gqa2 | 128 | 0.524288 | 0.573440 | 0.049440 | 0.031744 | 0.067584 | 0.039936 | −0.009248 |
| mha4 | 4 | 0.739360 | 0.802816 | 0.066496 | 0.053248 | 0.102400 | 0.073952 | −0.015360 |
| mha4 | 128 | 1.140576 | 1.259520 | 0.119616 | 0.053312 | 0.171008 | 0.076960 | −0.026368 |

Selected paired 95% bootstrap intervals:

| model / seq | gap | wait | notify | barrier | loop |
|---|---|---|---|---|---|
| gqa2 / 4 | [0.033696,0.038784] | [0.026816,0.030720] | [0.055296,0.060096] | [0.037120,0.042016] | [−0.013056,−0.009504] |
| gqa2 / 128 | [0.048128,0.051200] | [0.029696,0.032800] | [0.064512,0.069696] | [0.036544,0.046080] | [−0.012288,−0.007168] |
| mha4 / 4 | [0.064512,0.072704] | [0.051200,0.055296] | [0.100352,0.110816] | [0.071584,0.075904] | [−0.018432,−0.012512] |
| mha4 / 128 | [0.117664,0.122720] | [0.049152,0.054272] | [0.167936,0.173056] | [0.073728,0.078176] | [−0.028672,−0.023488] |

All gap/wait/notify/barrier intervals exclude zero (`p ≤ 1.48e-5`). The bare
queue loop is 9–26 µs faster than barrier-free L1, but wait plus notify exceeds
the barrier it replaces. Notification is the largest positive term in every
cell; at mha4 seq=128 it is 171 µs versus 53 µs for waiting.

## Poll complexity and interpretation

✅ A task walks only its materialized unique event interval. kAll is one
aggregate event per CG edge, and narrowed edges use logical-task groups;
monotone lifting removes events already observed earlier by the worker. On the
reference fixtures this is 500 waits for 200 gqa2 tasks and 1,076 for 512 mha4
tasks. The final `sm_89` wait control path is 50 static instructions, including
a 23-instruction retry loop. Polling is CTA-parallel.

✅ `L2 = neither + notify + wait` still closes exactly. The previous
stage-loop attribution and both intermediate queue decompositions are invalid:
their publication points and event keys differ from this executor.

❌ Events are not intrinsically cheaper than the grid barrier here. The queue
must earn back 30–53 µs of wait and 58–171 µs of notify through useful overlap;
the measured models do not. Direct overlap is nevertheless real and reported
separately in `../OVERLAP/`.
