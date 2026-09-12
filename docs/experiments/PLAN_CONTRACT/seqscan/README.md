# E1-d: arbitrary σ is correct on the SEQSCAN subset

The host no longer builds queues stage-major; it walks `plan.queue[worker]` in σ
order (`e7b4c5cc`). E1-b shows that for modes 0, 4 and 5 the *dumps* are byte
identical to the baseline harness, but byte identity of a placement table is not
correctness of the executor that consumes it — the wait hoisting and local
elision rules read σ, and a σ path that is wrong only under some seq/past shape
would still reproduce the table. So this gate runs the numerics.

## Result (verified, 600 fresh processes)

| | seq 4 | seq 128 | seq 2048 |
| --- | --- | --- | --- |
| **gqa2** past 0 | 50/50 | 50/50 | 50/50 |
| **gqa2** past 512 | 50/50 | 50/50 | 50/50 |
| **mha4** past 0 | 50/50 | 50/50 | 50/50 |
| **mha4** past 512 | 50/50 | 50/50 | 50/50 |

`E1-d: PASS`. 12 cells × 50 processes, the pass rate reported per cell as
CLAUDE.md requires of a synchronization claim. Raw per-cell logs are in `raw/`
and the table in `raw/correctness.tsv`.

Placement is 0 here — the legacy plan travelling the new σ path, which is the
combination E1-b can only check as a table. Modes 4 and 5 are covered byte for
byte by `../run.sh`, and mode 5's own 50-process correctness is in
`../mode_identity/`.

## Reproduce

```
bash docs/experiments/PLAN_CONTRACT/run.sh      # generates /tmp/plan_contract/*.cu
bash docs/experiments/PLAN_CONTRACT/seqscan/run.sh
```

`run.sh` compiles `/tmp/plan_contract/${model}.cu` when it exists — the
plan-carrying source `../run.sh` generated — and falls back to the archived
SEQSCAN source otherwise. The fallback does not exercise σ, so the two scripts
are run in this order and not independently.

## A cell already at 50/50 is kept, not re-measured

12 cells × 50 processes at seq 2048 is about an hour of wall clock, and this run
was in fact interrupted twice by a dropped connection. Resume reads
`raw/correctness.tsv` and keeps only cells already at `50/50`; everything else
is measured again from scratch. Every cell in the table above is therefore its
own 50 fresh processes — just not all in one stretch, which is why `(kept)`
appears in the runner's output and is recorded here rather than being presented
as a single uninterrupted run.
