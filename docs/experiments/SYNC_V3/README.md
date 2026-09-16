# R4 reproducibility and evidence layout

Run `python3 docs/experiments/SYNC_V3/verify.py` from the checkout. It uses
Python's standard library and reads raw process logs, sidecar metadata,
materialized trace tables, generated DAG source and SASS. It does not trust
summary verdicts. Any failed hard gate produces a nonzero exit. Research and
report results remain visible even when a hardware-dependent item is pending.
The original premature C1 commit remains an explicit historical deviation;
the resumed A/target/B/C prerequisite order is checked in git ancestry.

The source prompt is outside the repository at
`/root/Prompt/TileMega_R4_prompt.md`; its digest is in `summary.md`.

## Protocol build and measurement

`../FENCE/run.py` builds the R3 B protocol plus explicit flags and supplies
five-arm, correctness, SEQSCAN and SASS phases. Each new runtime mechanism is
default-off. C2 requires C1 and BARRIER_V2. Window variants additionally use
`--window 2` or `--window 4`; the runner stamps an unmaterialized source and
lets the host construct its Plan for the actual resident grid.

`matrix_manifest.json` identifies the seven independently built configurations.
`measure_matrix.py` rotates all configurations, both placements and all five
probe arms together, in 25 rounds per reference cell. It refuses to overwrite
any process log. `realwidth.py` builds/measures the required five configurations
at real-width seq=4, placement 0. `target_positions.py` measures the three W=1
protocol variants on each additional frozen candidate; balanced uses placement
4, matching its frozen A trace. An earlier unused placement-0 balanced build
was corrected before target-position measurement and is not a timing sample.
`continue_measurements.py` serializes the suites after dependent correctness
checks; no two performance processes are launched concurrently.

`metrics.py` and `report.py` render review tables. Attribution is
`wait=full-nowait`, `notify=nowait-neither`, `fence=full-nofence`,
`barrier=L1(full)-L1(l1nosync)`. The fence marginal overlaps the other terms;
it is not added to wait+notify again. Signed finite paired differences are
kept. Bootstrap intervals use 10,000 median resamples with seed 167. Neither
mechanism selection nor report generation edits the frozen targets.

## Correctness and scope

`litmus_v3/` is the sensitive two-suite recheck; `litmus_recheck/` preserves
the earlier insensitive rerun. `c1/` and `c2/` each contain four 50-process
reference cells and six 50-process SEQSCAN subset cells. `c2_dependency/`
adds a kappa=2 configuration and trace witnesses where next-slot waits retain
an event dependency on the current publication. `local2/local4` and their
`window2/window4` controls contain four 50-process cells each. `sharded_red/`
checks RED/shard composition on sm_89. Numerical results from unsafe timing
arms are never accepted as correctness evidence.

`cluster_compile/` and `cluster_model_sm120_composed/` contain compile-only
sm_120 evidence. `cluster_degenerate/` is the C2-based sm_89 byte-equivalence
check; final `sass_identity/` regenerates a second fallback check at the sealed
head. `run_sm120.sh` includes target-local litmus, the grouped dependency case,
all protocol/window arms, and a same-session cluster-off/on comparison. It
hard-checks compute capability and disk space and rejects inherited
TILEMEGA_* variables. All three R4 runners pass `SELF_CHECK=1` on sm_89;
none has been executed on sm_120.

On the target, rebuild `build-portable` and provide the portable SEQSCAN
fixtures/exports plus REALMODEL `r2sim_s4` and `r2sim_s128` sources, exports
and fixtures. These inputs may be regenerated or transferred; materialized
worker/slot tables must be solved there. The Chain2 runner independently
remeasures the target hop curve into its R4 output directory before solving
the three chain variants. Run GPU experiment scripts one at a time.

## Final identity stamp

All implementation and documentation changes are committed before
`sass_identity.py` compiles baseline/current/default-with-cluster-switch
binaries and compares complete SASS. The final artifact-only commit stores
this evidence. Its parent must equal `manifest.json::tested_head`; a commit
cannot embed its own content hash. The verifier checks this relation, the
changed path set and all recorded input hashes. A new code/documentation
commit after the stamp invalidates freshness and requires a new stamp.

Binaries, object files, large exploratory edge inventories and unrelated R3
working-tree files are not delivery artifacts. Exact build commands, source
and executable digests, compact raw diagnostics and process logs are kept.
