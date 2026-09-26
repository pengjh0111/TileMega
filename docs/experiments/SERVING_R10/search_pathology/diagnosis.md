# R10 search pathology and repair

The first pruned Llama decode B=1 equivalence run terminated with SIGSEGV
after its second attention-coordinate sweep. The backtrace in
`decode_search_segmentation.gdb.log` ends at
`SolveSkeletonImported -> ModelDescription::MetricBindings`: switching back to
an already scored attention structure cleared `SearchContext::base` and
`floor`, and memo hits did not rebuild them. Commit `379c0b6ab` restores the
selected structure before writing floor evidence. The two-pass search-only
reproduction then produced `pruning_repaired.floor.tsv` without crashing.

The first Llama prefill B=1 search stopped advancing after candidate 1386.
Attaching to the live process put it in `OracleProgram::Build`, inside ISL's
affine-hull solver, for a one-piece floor-division relation; the relation is
recorded in `oracle_ast_diagnostic.stderr`. Removing whole-domain AST
construction alone moved the stall to `isl_pw_multi_aff_from_map`, then to
`isl_pw_aff_eval`. Commit `4fb8954f1` retains the Unique classification and
uses an exact, locally bound Presburger fiber for many-to-one maps. The same
problematic candidate has identical Level 1 score
`12563533.813808465 ns` in `oracle_piece.search.tsv` and
`oracle_bound.search.tsv`; the bound-fiber run completed in 8.493 s.
`symbolic_oracle` checks exact set equality and out-of-domain behavior for
the new path. The full CTest suite passed 80/80 after this change.

The interrupted search logs are preserved here. They are diagnostic data,
not completed plan results or evidence for G-6/G-7.

Qwen3 decode B=1 subsequently spent minutes in `InflightDramServer::Advance`
for configurations with many identical streams. The original server advanced
each active task on every event. Commit `c19b027f0` groups streams with the
same rate cap and in-flight-byte class behind one service clock; the per-task
remaining bytes still determine each task's completion event. A 400-case
staggered-stream reference test compares the grouped server with the dense
implementation. For the stalled candidate, the old search row 611 and the
replayed grouped run both report `760352559.42175853 ns`; the grouped two-case
replay completed in 28.4 s, including flow preparation. The interrupted Qwen3
matrix log and replay inputs/outputs are retained here. This repairs a CPU
evaluation stall; it is not a completed Qwen3 plan or GPU performance result.

A second stall survived cohort grouping. The Qwen3 B=16 single-case diagnostic
hit 100,000 in-flight `Next()` calls with one rate class and 16 active tasks:
`clock=567680787.363556`, `due=567680787.363556`. At that absolute timestamp,
the residual duration rounded below one double clock ULP. `Next()` returned
zero; `Advance(0)` could not meet the old fixed 1e-6-byte completion tolerance,
so the same event repeated indefinitely. The repaired server uses an absolute
due clock, lazy class settlement, and a completion tolerance of at most two
clock ULPs converted through that class's current rate. The identical B=16
candidate now completes with score `2271847038.5698881 ns` in about 59 s.
`stage_flow` retains 400 staggered-group comparisons with the dense reference;
the numeric stall evidence is in `qwen3_b16_zero_progress.stderr` and the
post-repair score is in `qwen3_b16_ulp.search.tsv`.
