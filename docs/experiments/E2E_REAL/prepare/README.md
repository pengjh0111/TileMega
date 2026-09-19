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

# What the hoist bought

`BuildModelPlan` is the pattern match the profile above spends 32 of 41 samples
in, and it does not read `ImportOptions`, so the search builds it once and hands
the same plan to every import.  Two whole searches of the llama export at seq 4,
past 3, capacity 12, same target and search domain, both to completion:

    control (02e1a2c81, plan rebuilt per import)  12982.1 s
    hoisted (f6b00ac11, plan built once)           5167.2 s   2.51x

The two `auto.cu.search.tsv` are byte-identical (361 rows, md5
`d42b6bb8c2df1fc55d25950f535093b9`), so the hoist moved the cost and nothing
else -- verified.  The two runs are not concurrent, so that ratio carries this
machine's load as well as the change.

`import_rate.tsv` is the contention-controlled half, written by `imports.sh`:
one row every 30 s counting `IMPORT_DEGRADED` lines in the hoisted run and in a
still-running pre-hoist process (the capacity-1 llama solve), while both shared
the machine.  Over 4631 s the hoisted run completed 36 imports (128.6 s each)
against the control's 18 (257.3 s each), a factor of 2.00 -- verified.  The
control is at capacity 1 and the hoisted run at capacity 12; what makes the two
comparable is that the outer enumeration imports one coarse module per
(geometry, split) pair regardless of capacity, and both processes were inside
that enumeration for the whole window.

Neither number is a capacity verdict.  §6 C1-b is scored on the searchable
space, and one import still costs minutes, so see `solve_profile.tsv` above and
the C1-b row of `verify.py`.
