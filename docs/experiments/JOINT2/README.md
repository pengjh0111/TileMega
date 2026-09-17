# Round six evidence

`summary.md` records the complete gate table, verifier output, limitations and
follow-up work. Implementation completion and acceptance are separate: a
passing research comparison does not erase a failed hard gate.

## Recheck the archived run

From the repository root, run:

```sh
python3 docs/experiments/JOINT2/verify.py
```

The verifier runs every gate before returning. Any failed hard gate makes the
exit status nonzero. It reads raw process logs/metadata, task and Plan dumps,
cost-evaluator rows, proof logs and the SASS manifest; it does not consume the
report's conclusions. No GPU launch is needed for this archived-data check.
The final SASS gate also checks the Git parent/child relationship: its
evidence-only commit must immediately follow the final source/document commit.

Evidence is divided by responsibility:

| Directory | Inputs and results |
| --- | --- |
| `../COSTMODEL/` | Derived-access prices, K-loop traces, calibration and replay rows |
| `cells.json` and its six indexed directories | Frozen choices, separate pilots, 25-round comparisons, 50-process correctness and trace dumps |
| `fuse_upper/` | Existing legal fusion-pair model bounds and aggregation |
| `../WRITEBACK/` | Compiler commands, CG/host round trips, legacy tables, CTest and SEQSCAN |
| `../REBASE/raw/` | Ten configurations × five probe arms × 25 processes in each of six cells |
| `../REBASE/probe_audit/` | Signed barrier pairs and unchanged-kernel control checks; no sample filtering |
| `../SYMBOLIC/complete/` and `../SYMBOLIC/cross_grid/` | Finite interval certificates and complete materialized-table comparisons |
| `../MODELS/` | Public config bytes/URLs, operator coverage and the explicitly cut MLP graph |
| `sass_identity/` | Final baseline/current disassembly, empty diffs and source-parent manifest |

The phase and Plan tables use nanoseconds where marked `_ns`; benchmark
`l2_ms` and intervention columns marked `_ms` use milliseconds. A protocol
probe difference is not automatically an isolated nonnegative hardware cost.
The real-s128 control-band limitation is documented in
`../REBASE/probe_audit/README.md`.

Large executables, object archives, random model fixtures and `.pt2` inputs
are regenerated locally rather than checked in. The model-subset recipe is
`../MODELS/subset.md`. Per-build command logs record flags and resource checks.
Performance collectors refuse to overwrite existing raw samples; use a fresh
output directory for a new campaign.

## sm_120 execution

These runners were written and self-checked on sm_89. They have **not** been
executed on sm_120. The complete parameter and artifact schema is in
`sm120_common.sh` and applies to all four entry points:

```sh
SELF_CHECK=1 OUT_DIR=/tmp/r6-cost-selfcheck bash docs/experiments/COSTMODEL/run_sm120.sh
SELF_CHECK=1 OUT_DIR=/tmp/r6-joint-selfcheck bash docs/experiments/JOINT2/run_sm120.sh
SELF_CHECK=1 OUT_DIR=/tmp/r6-rebase-selfcheck bash docs/experiments/REBASE/run_sm120.sh
SELF_CHECK=1 OUT_DIR=/tmp/r6-models-selfcheck bash docs/experiments/MODELS/run_sm120.sh
```

For actual execution on the target machine, omit `SELF_CHECK`, use fresh
`OUT_DIR` paths and a target-configured `build-portable`. The runners require
compute capability 12.0, reject inherited `TILEMEGA_*` flags, check
`NEED_MIB=131072` before compilation and write `status.txt` and `pipeline.log`.
`SEARCH_CAPACITY` defaults to 18 and remains an explicitly reduced search
budget. `LOCAL_R5_ROOT` may reuse only a runner-produced local calibration
with the same GPU UUID; otherwise the pipeline regenerates and recalibrates
the controls. Materialized sm_89 Plans must not be copied to the target GPU.
Symbolic portability comparisons also use that target's freshly solved CG.
