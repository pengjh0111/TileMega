# Queue-era Place comparison

Reproduce with `RUNS=25 bash run.sh`. The two arms use the same generated CUDA
and binary; `TILEMEGA_SCHEDULE_POLICY` selects either the emitted
critical-path-first stage order or the numeric topological baseline while the
host materializes worker queues. Arm order alternates inside each round.

Evidence status: ✅ RTX 4090 (`sm_89`), BF16, κ=1, `seq=128,past=3`, 25 fresh
paired processes per model, 2026-09-07. Raw data and deterministic reduction
are under `raw_taskqueue/`.

## The schedule is now real

Unlike the retired PLACE experiment, this knob changes the queue consumed by
the persistent L2 kernel. Every `TaskRef` is appended according to the selected
stage order, then executes with task-local waits and notifications. The old
placement numbers measured only a CTA-index permutation outside the generated
solver schedule and are ❌ not reused.

Both arms passed 25/25 for both models. Generation and host materialization
also reject cycles and backward dependencies before launch.

## Small-instance oracle

✅ `list_scheduler_test` exhaustively enumerates all 120 permutations of a
five-node diamond DAG. Three are dependency-feasible; the optimum maximum
dependency span is 2, and the critical-path scheduler also obtains 2:

```
PLACE_ORACLE nodes=5 feasible=3 optimum_span=2 cp_span=2
```

This proves optimality for the bounded oracle instance. It is not presented as
an optimality proof for either full model.

## End-to-end comparison

The paired statistic is `round_robin / critical_path`; values above one favor
critical-path order.

| model | n | critical path median | round-robin median | paired ratio | bootstrap 95% CI | Wilcoxon p | CP faster |
|---|---:|---:|---:|---:|---:|---:|---:|
| gqa2 | 25 | 0.615424 ms | 0.615296 ms | 1.000000 | [0.996476, 1.001664] | 0.2373 | 11/25 |
| mha4 | 25 | 1.255424 ms | 1.255232 ms | 0.998552 | [0.997113, 0.999804] | 0.06337 | 6/25 |

Separate medians are not divided post hoc; the within-round paired ratio is the
valid statistic. gqa2's interval crosses one. mha4's bootstrap interval lies
slightly below one, but its Wilcoxon test does not reject at 0.05 and the
direction favors round-robin by only 0.145%, far below the 2% decision
threshold.

## Decision

✅ Place is now an executed decision variable and its correctness path is
covered. ❌ On these models the critical-path order does not deliver a
statistically separated >=2% latency benefit over numeric topological order.
Therefore Place stays in the execution contract—removing it would regress to
the original architectural bug—but it is not claimed as a performance win.

The result also narrows the next objective: maximum DAG span is useful for
legality and the small oracle, yet it did not predict end-to-end latency here.
A richer objective would need to price task duration and ready-queue slack; no
unmeasured heuristic is substituted in this round.
