# D-e: what the solver decided, and what a human still decides

One command produced the anchored Llama-3.2-1B megakernel (§7.1). This is the
item-by-item split of that command, read off the invocation and the generated
source rather than described from memory.

The command, verbatim, is in `llama/solve.json`; the generated macros quoted
below are from `llama/auto.cu`, and the candidate table from
`llama/auto.cu.search.tsv`.

## Decided by the solver

| Decision | Where it shows up | Value on the anchored run |
|---|---|---|
| GEMM geometry (tile M/N/K, stages) | `TILEMEGA_GEMM_TILE_*`, `TILEMEGA_GEMM_STAGES` | 32x16x64, 2 stages |
| split-K | winner key in `auto.cu.search.tsv` | 4 |
| event kappa | `TILEMEGA_EVENT_KAPPA` | 1 |
| residency (CTAs per SM) | `TILEMEGA_RESIDENCY_CAP` | 4 |
| grid | `TILEMEGA_SOLVED_GRID` | 512 |
| placement family, out of six | `E2E_PLACE_STATS plan=` | `eft` |
| slot order inside each worker | the emitted plan table | 388 tasks over 512 workers |
| which candidates to price, in what order | `auto.cu.search.tsv`, `search_deferred` | 12 evaluated, 69 deferred |
| per-stage kappa refinement, when enabled | `--per-stage-kappa` descent | not enabled on this run |
| the interval cut and per-segment geometry, when enabled | `--segments` (§5.4 B3) | not enabled on this run |

Nothing in that column was passed in. The winner is the `(floor, predicted)`
minimum over everything the capacity budget admitted.

## Still decided by a human

| Decision | How it enters | Value on the anchored run |
|---|---|---|
| which program to compile | the exported `.pt2` | Llama-3.2-1B, public config, seeded weights |
| the theta point | `--seq`, `--past` | seq 4, past 3 |
| the evaluation budget | `--search-capacity` | 12 |
| the admitted geometry domain | `--search-domain` | five measured backend geometries |
| numerically rejected geometries | `--numerical-rejections` | none passed on this run |
| the machine description | `--solve target.json` | sm_89 |
| the hop curve | `--hop-curve` | `docs/experiments/SIMULATOR/hop_ns.tsv` |
| segment count and segment candidate count | `--segments`, `--segment-candidates` | off |
| prefetch page size | `--prefetch-page-bytes` | default |
| protocol macros of the harness build | `-DTILEMEGA_*` in `measure.build` | BARRIER_V2, WAIT_POLICY, SLOT_WINDOW, MIDPOINT_REFINE |
| target architecture and compiler flags | `nvcc -arch=` | sm_89, -O2 |
| the comparison tolerance | `Compare()` in `ModelHarness.cuh` | 1.6e-2 absolute plus 1.6e-2 relative |

Two entries in that column are budgets rather than answers: the capacity and
the geometry domain bound what the solver may look at, and both are disclosed
in the generated module (`tilemega.search_deferred`,
`tilemega.search_restricted_geometry`) so a reader can see the search was
finite and reduced. The rest are properties of the question, not of the plan.
