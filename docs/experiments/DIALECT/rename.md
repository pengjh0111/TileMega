# BE-8 — the rename, term by term

CG-stage objects are tiles. `task` belongs to the executor, so the names that
describe a tile now say so and the names that describe execution do not move.

## IR

| before | after |
|---|---|
| `tilemega.task_space` | `tmcg.tile_space` |
| `tilemega.event_tensor` | `tmcg.event_tensor` |
| `tilemega.coupling` | `tmcg.coupling` |
| `tilemega.fused_task_space` | `tmcg.fused_task_space` |
| `tilemega.placement` | `tmexec.placement` |
| `tilemega.implementation` | `tmexec.implementation` |
| — | `tmcg.graph` (new) |
| — | `tmexec.plan` (new) |
| `#tilemega.<attr>` | `#tmcg.<attr>` |
| `tilemega.solved_*` (module attributes) | `tmexec.solved_*` |

The module-level `solved_*` attributes moved with the ops: they are the
solver's decisions, which is what `tmexec` means. `tmexec.placement_table` and
`tmexec.placement_bindings` moved for the same reason.

## C++

| before | after |
|---|---|
| `TaskSpaceOp` | `TileSpaceOp` |
| `CG_TaskSpaceOp` (tablegen) | `CG_TileSpaceOp` |
| `CGDialect` | `CGDialect` (now `tmcg`) plus `ExecDialect` (`tmexec`) |

## What deliberately did not move

`TaskBody`, `TaskKind`, `TaskRef`, `TaskTraits`, `TaskOwnership`, `TaskSmem`
and the harness's `task` variables all keep their names. They are executor-side
names for executor-side things: a tile space is a CG object, a task is what one
worker runs. The `CG_TaskKindAttr` attribute keeps its name too, because it
names the kind of *body* a tile space will be executed by.

Historical artifacts under `docs/experiments/` keep the old spelling: they are
evidence of runs that happened, and rewriting them would break the trace back
to the logs they came from. `docs/FINDINGS.md` keeps its historical entries and
gains a terminology note at the top instead.
