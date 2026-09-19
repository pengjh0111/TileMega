# What each folder under `raw/` is

`correctness_head/` is B1-a's evidence: every cell, both arms, 50 fresh
processes each, all built at one driver vintage (`build/*.json` carries it).
The gate reads only these.

`correctness/` is the first round, built before the real cells' page was
raised.  It is kept because it is the round that established the page, but it
predates the `E2E_PREFETCH` counters and so cannot show why: `page_probe/`
does.  Three fresh processes of `real_s4` built at the 1024-byte default report
`E2E_PREFETCH slots=12208 declared=32 issued=0 queue_heads=0 page_bytes=1024`
and still `RESULT status=PASS` -- the mechanism compiled and idle, which is
what `run.py`'s `PAGE` map lifts the real cells to 8192 to avoid.  Three, not
fifty: it is a counter reading, not a correctness or race claim.

`correctness_p8192/` is the real-width re-run at the raised page, taken before
the rebuild that put every binary at one vintage.  Its `inline` arm was
interrupted at 22 rounds and is named for that; nothing here is a gate input.

`build_prev/` holds the build metadata each cell carried before that rebuild,
so the earlier folders' binaries stay identifiable.

`llama/` is the maximal connected graph replayed at HEAD, `llama_sigma0/` the
same replay with the pipelining dimension priced at zero (`regeneration.tsv`
compares both with R6's admitted geometry).

`seqscan/` is R5's twelve-case subset: `<cell>/<case>/` holds the plan
re-projected at HEAD beside JOINT's own recorded command, and
`<cell>/seqscan/<case>/<arm>/` the 50 processes per arm.  `regeneration.tsv`
is the byte comparison against the plan JOINT ran, with the guarded
`no_producer` field stripped and nothing else.

`phase_head/` is B1-c's evidence: `TRACE_KLOOP` builds of all three arms, 50
rounds each, one dump per round.  `overlap.py` reads every round and writes
`overlap_head.tsv`; only round 0's dump of each arm is committed, because the
full 900 dumps are 3.7 GiB of TSV.  The command that regenerates them is in
`build/*_phase.json`, and the derived table is the evidence the gate reads.
A solver `task_dag.tsv` is left out of the commit for the same reason -- the
four `s2048_p0` plans alone are 23 GiB -- and is reproduced by the recorded
projection command in `<cell>/<case>/process.json`.
