# sm_120 artifacts that landed in the sm_89 paths

`run_sm120.sh` delegates to `run.sh`, and `run.sh` writes into this
experiment's ordinary `raw/` tree and its top-level tsv files.  On the
Blackwell run of 2026-09-12 that overwrote the sm_89 evidence the round-one
report cites, so every file the run replaced is kept here under the path it
was written to, and the sm_89 files were restored from `886f0dc9`.

Read this tree as the sm_120 evidence and the sibling `raw/` tree as the
sm_89 evidence.  Absolute latency is not comparable between them: they are
different devices and different sessions, and only paired ratios measured
inside one session mean anything.

`raw/realwidth/status_s4.tsv` is byte-identical on both machines; it is
copied here for completeness rather than because the run changed it.

`run_sm120.sh` now runs a `harvest` step before it writes `PASS`: everything the
run changed under the experiment is copied here, tracked files are restored from
the index, and untracked ones are removed.  Compiled binaries, exported model
weights and scripts are left alone, so a later Blackwell run leaves both the
sm_89 tree and any local edit intact.
