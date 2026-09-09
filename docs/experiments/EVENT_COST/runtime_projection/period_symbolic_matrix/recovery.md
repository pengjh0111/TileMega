# Interrupted CPU batch

The process was no longer alive on reconnection. `status.txt` is the
unchanged stale RUNNING marker, **not** evidence of an active experiment.
Eight completed queries are in `manifest.json` and `counts.tsv`; all exited
zero. The last completed query was mha4/split4. No result is claimed for the
unrecorded ninth query from that interrupted process.

The two missing queries were run separately with the same frozen binary in
`../period_symbolic_continuation/`. `merge_projection.py` checks exact query
coverage and binary identity before joining complete cells; it does not
overwrite either raw batch. The combined 150 cells then drive an independent
comparison against every process in the archived 7500-process GPU matrix.
