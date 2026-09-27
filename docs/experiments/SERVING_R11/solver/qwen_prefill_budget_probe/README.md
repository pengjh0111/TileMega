# Qwen3 B=16 prefill plan-time diagnostic

The current-source `python -m tilemega build` started the Qwen3 B=16 prefill
solver at 2026-09-27 02:17:11 UTC. After 22 min 14 s (search TSV last write
02:39:25 UTC), it had logged 1,657 Level 1 evaluations and had not completed
even prefill search. The diagnostic was interrupted at that point to avoid
spending further CPU time on an already failed 600 s per-plan budget; no
selected plan or GPU measurement was produced. Its stdout/stderr contain no
solver exception. `search.tsv.xz` preserves the partial search trace. The
binary at launch was `61d19ec59`; the Oracle cache change `11b1fc347` was
committed while this already-loaded process ran and therefore was not used by
its search. An isolated build and flow-audit show the Oracle change preserves
the score but saves only about 14 ms in a 218 ms structure phase, so it is
not expected to close this budget alone.

The diagnostic overlapped an isolated CPU build and host tests. The elapsed
time is therefore a resource-contended measurement, although it exceeds the
budget by more than 2.2× before materialization or top-3 compilation.
