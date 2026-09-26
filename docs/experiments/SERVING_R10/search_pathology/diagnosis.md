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
