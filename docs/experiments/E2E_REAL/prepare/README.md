# C1-b: the bound read off the relation instead of the edges

`prepare_bounds.tsv` is the before/after.  `prepare_bounds` prepares each cell
twice -- once materializing the dense edge set, once asking
`PreparePlacementProblem` for bounds only -- and times the bound stage in
isolation inside each arm, because the whole preparation is dominated by
quasipolynomial task pricing and would hide the change (`dense_total_us` and
`interval_total_us` are that whole, kept so the share is readable: at seq 4 the
bound stage is 0.9% of preparing `gqa2`, 0.11% of preparing `llama`).

Both arms answer the same two numbers, so every row is also an equality check
on a real relation rather than on the unit test's synthetic ones:
`identical` is 1 on all nine rows, `work_ns` and `critical_path_ns` agreeing to
under 1e-9 in both the isolated and the in-pass comparison.  The tool throws
`interval bounds disagree with the dense edge set` if they ever do not.

    build_tools.py --out <dir>
    <dir>/prepare_bounds COSTMODEL/event_fit/target.json prepare/ \
        JOINT2/bounded_search/gqa2_s4/auto.mlir:4 \
        JOINT2/bounded_search/gqa2_s128/auto.mlir:128 \
        JOINT2/bounded_search/gqa2_s128/auto.mlir:512 \
        JOINT2/bounded_search/mha4_s4/auto.mlir:4 \
        JOINT2/bounded_search/mha4_s128/auto.mlir:128 \
        JOINT2/bounded_search/mha4_s128/auto.mlir:512 \
        JOINT2/bounded_search/real_s4/auto.mlir:4 \
        JOINT2/bounded_search/real_s128/auto.mlir:128 \
        PIPELINE/raw/llama/auto.mlir:4

`build/` is the flags this machine compiled it with.  The two rows at seq 512
are the `_s128` coupling graph re-projected at 512, not a solved cell of their
own; they are here because the edge count is quadratic in seq and the point of
the table is how the two arms separate as it grows.

# Why the search capacity does not move

`solve_profile.tsv` is why C1-b cannot buy search space on the real model.  It
samples one `tilemega-compile` of the llama export at seq 4, past 3, capacity 1
(`sample.sh PID N SLEEP LOG`, one `gdb -batch -ex "bt 40"` every 15 s): columns
are the epoch second, how many imports the process had reached by then, the
innermost TileMega or ISL frame, and the three innermost TileMega frames.

Of 41 samples, 32 are in `PatternMatcher::DependsOn` / `OperandConstraint` /
`FxNodeRecord` -- the pattern match `BuildModelPlan` runs, re-run from scratch
by every import -- 9 are in ISL quasipolynomial pricing, and **none** is in
`isl_set_foreach_point`, the enumeration C1-b removes.  The import counter
moves 2 -> 3 -> 4 across the 628 s window, so one import is about 3.5 minutes
and the outer loop pays it once per (geometry, split) pair.

Sampling stopped at 41 rows.  The rows after it are discarded rather than
present-and-empty: `tools/tilemega-compile` was relinked while gdb was attached
to the still-running old image, after which every backtrace came back with no
top frames.  The measured process kept its old image and is unaffected; only
the sampler's view of it broke.
