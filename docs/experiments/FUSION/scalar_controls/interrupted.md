# Interrupted scalar controls

The tool connection restarted before summary.json was emitted. Individual
completed tables remain, but this directory is not a completed 14-table gate.
No failure or success count is inferred from the stale RUNNING state of the
other concurrent runner. A separate directory retains the subsequent rerun.
