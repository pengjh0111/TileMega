# H3 / E1-b: modes 0, 4 and 5 replayed through the Plan contract

Gate E1-b (R2 §4.3): each of `TILEMEGA_PLACEMENT` 0, 4 and 5 must be expressible
as a Plan, and the queues, waits and event tables the host materializes from
that Plan must be byte-identical to what the pre-plan host produced.

Reproduce with `../run.sh`, which prints the E1-a and E1-b verdicts.

## Method

Two binaries per cell, differing only in how the queues are built.

* **baseline** — `git show 6c359e2b:include/tilemega/Codegen/tasks/ModelHarness.cuh`
  materialized into a copy of `include/`, i.e. the R2 baseline host with its
  original `for worker → for stage : stage_order → for task` materialization.
* **plan** — the working tree, which resolves a `PlacementMode` and calls
  `solver::MaterializePlanPlacement` / `solver::CheckPlanLegality`, then walks
  `plan.queue[worker]` in σ order.

The `TILEMEGA_PLAN_DUMP` block that writes the three TSVs is *the same source
text* in both: it was lifted verbatim from the working-tree harness into the
baseline copy. Only the materialization logic differs, so a byte-identical dump
is evidence about the placement, not about the dumper.

Dumps are written before any device work and are deterministic:

| file | columns |
|---|---|
| `schedule.tsv` | worker, slot, stage, logical_task, dependency_begin, dependency_count, wait_begin, wait_count |
| `waits.tsv` | index, producer, group |
| `events.tsv` | stage, event_offset, event_flags |

The `.cu` sources are generated from the same commit for both binaries; only the
harness header differs.

## Cells

2 models (`gqa2`, `mha4`) × 3 placements (0, 4, 5) × 2 seq (4, 128) = 12 pairs,
24 runs. Placement 0 is `legacy_grid_stride`, 4 is `balanced`, 5 is `rotate`
(base = prefix sum of the active task counts along `stage_order`, mod grid).
Placement 4 needs the variant to request the balanced L-sched writeback, so its
two binaries are built from the `_balanced` source; 0 and 5 are selected by the
`TILEMEGA_PLACEMENT` macro alone.

## Result — PASS (verified)

* 36/36 dump files byte-identical: 12 pairs × 3 TSVs. `sha256.txt` holds the 72
  hashes (24 runs × 3 files); the baseline and plan hash of each cell match.
* 24/24 runs report `RESULT status=PASS`.
* `E2E_BALANCED` unchanged.
* `E2E_PLACE_STATS` is unchanged except for one appended field, `plan=`:

  ```
  mode 5, mha4 s128:  max_queue=18 same_worker_edges=3624 cross_worker_edges=545252 base_rotation=1 plan=rotate
  mode 4, mha4 s128:  max_queue=34 same_worker_edges=5274 cross_worker_edges=543602 base_rotation=0 plan=balanced
  ```

The `plan=` field is new output, not a changed value; every pre-existing field
is byte-identical to the baseline.

## Files

* `sha256.txt` — 72 lines, `<run> <sha256> <file>`.
* `dumps.tar.gz` — all 24 dump directories.
* `{base,plan}_{gqa2,mha4}_p{0,4,5}_s{4,128}.out` — the 24 run logs.
