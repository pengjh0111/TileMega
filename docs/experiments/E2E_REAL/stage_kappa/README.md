# §5.3 B2: per-producer-stage kappa

Event coarsening `κ` was one value per plan.  It is now a runtime field per
producer stage: the solver projects each stage's events at its own `κ`
(`stage_kappa` on `PlacementSolveOptions`), the solved plan carries the table
(`tilemega.solved_stage_kappa`), codegen emits it as
`TILEMEGA_EVENT_KAPPA_TABLE` behind `TILEMEGA_EVENT_KAPPA_PER_STAGE`, and the
runtime reads the coarsening for a wait from that table instead of the global
macro.  With the flag off nothing is written, so the default build's SASS is the
baseline's (H2, `../sass_identity/`).

The search dimension is a coordinate descent over projected stages, started
from the uniform `κ` the outer search picked: every trial is a full placement
solve under the trial table, a stage with one task is skipped (one group at any
`κ`), and the descent stops after two passes or the first pass that moves
nothing.  A table pinned from the command line (`--stage-kappa`) is re-solved on
the winner rather than relabelled, because a coarser event groups later producer
tasks into the wait and a slot order solved for uniform `κ` could put a consumer
ahead of one of them on its own worker.

`stage_kappa.py` runs two arms per model at seq 4, past 3, capacity 12:

    searched   --per-stage-kappa 1   the descent; whatever table it delivers
    forced     --stage-kappa CSV     a 1,2,4,1,2,4,... table pinned over the
                                     winner's projected stages, so the per-stage
                                     runtime path executes even when the search
                                     keeps uniform κ

`solve.json` records HEAD `f6b00ac11`; the `tilemega-compile` that ran was built
from that tree plus the `stages=` field of `SOLVE_STAGE_KAPPA` and the solver's
`stage_count`, which went in with the segmentation commit `1d7f2884b`.
`--report` rewrites `results.tsv` from the recorded runs (used once, after the
macro reader was fixed to take a comma list; the runs themselves are unchanged).

## Correctness (verified, 200 fresh processes)

| model | arm | projected stages | table | rounds | passing | binary sha256 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| gqa2 | searched | 44 | uniform (1) | 50 | 50 | `116507709f45…` |
| gqa2 | forced | 44 | 1,2,4,… over 44 stages | 50 | 50 | `2466b103a68f…` |
| mha4 | searched | 88 | uniform (1) | 50 | 50 | `2690d963602d…` |
| mha4 | forced | 88 | 1,2,4,… over 88 stages | 50 | 50 | `2a4232973e8c…` |

The two arms of a model are different binaries (the sha256 differ), which is
also the evidence that the pinned table reaches codegen and the runtime.
`results.tsv` carries the full tables and hashes; `<model>_s4/<arm>/correctness`
the process logs.

## Per-stage against global κ (verified: no difference)

| model | descent trials | moves | uniform_ns | per_stage_ns |
| --- | ---: | ---: | ---: | ---: |
| gqa2 | 88 | 0 | 169097 | 169097 |
| mha4 | 176 | 0 | 352991 | 352991 |

Every single-stage trial priced exactly the incumbent (`auto.cu.search.tsv`,
rows suffixed `-s<stage>k<kappa>`: one distinct `(floor, predicted)` pair per
model), so the descent never left uniform `κ` and §5.3's "if no difference, say
the global κ sufficed" is the result.  The invariance is a property of the
winning family, not of the pricing.  Both winners are `wavefront` at residency
4, and for the winner geometry the search's own global-`κ` rows show:

| family | gqa2 κ=1 / 2 / 4 | mha4 κ=1 / 2 / 4 |
| --- | --- | --- |
| wavefront (winner) | 169097 / 169097 / 169097 | 352991 / 352991 / 352991 |
| rotate | 179361 / 179361 / 179361 | 386120 / 386120 / 386120 |
| eft | 166272 / 174419 / 174419 | 348652 / 352810 / 352810 |
| chain | 180963 / 185037 / 185037 | 384971 / 388076 / 388076 |
| legacy_grid_stride | 217845 / 225671 / 225671 | 439662 / 455313 / 455313 |

The families that place task by task (`eft`, `chain`, `legacy_grid_stride`) do
move with `κ`; the two that are stage-major in slot order (`wavefront` by proved
level, `rotate` by stage order) do not.  Inferred from the template construction
in `include/tilemega/Solver/ParametricTemplates.h`: in a stage-major order every
producer of a consumer sits at a lower slot on every worker, so the wait is
resolved before the consumer's slot at any grouping, and per-stage `κ` has no
schedule to move.  The winner is chosen on `(floor, predicted)` and `wavefront`
has the lower floor, so `eft`'s smaller predicted makespan at `κ=1` does not
make it the winner; that ordering is R6's and is not changed here.

Solve cost (verified, this machine, not concurrent-controlled): gqa2 searched
647 s and forced 491 s; mha4 searched 1291 s and forced 681 s.  The difference
is the descent, 88 and 176 extra full solves.

## Not done

No cell where the descent moves exists in this round's evidence, so the
per-stage runtime path is exercised only by the forced arm.  A model whose
winner is a task-by-task family would be the first place to look for a
non-trivial table.
