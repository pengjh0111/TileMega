# A2 — element ownership and runtime variants

✅ BF16 7500/7500 fresh processes: two models, split 1/2/4/8/16,
seq 1/4/128/512/2048, past 0/3/512, 50 rounds per cell. Every process has
identical L0.5/L1/L2 output hashes and passes the existing reference check.
No numerical criterion was changed. Execution order rotates the complete
model/split/seq/past case list each round.

All four ownership flags are element-chunk in this additional matrix.
Each compiled model/split has two runtime variants: seq 1–128 uses
128x128x16s3, seq 129–2048 uses 64x128x16s3. This is additional evidence,
not a relabeling of the earlier tile-ownership matrix.

✅ The frozen symbolic projection (20 model/split/tile queries with S/past
unbound) agrees with all 15000 recorded `task_refs`/`waits` values exactly.
`verification.json` records 7500 processes and 15000 comparisons. The verifier
checks unique round/execution positions, complete state rotation, per-log
hashes, resource fields, cross-level hashes and immutable query receipts.
`prefix_verification.json` preserves the earlier 4000-process prefix audit;
it is not the final pass count.

`manifest.json` records frozen compiler/projection/binary/source/plan/export
hashes and compile commands; `source.diff` records the build-time source
state. Generated sources, ptxas output and all 7500 original logs are retained.
Executable artifacts are excluded from git. The projection's initial missing
audit-marker failure is preserved under `projection/initial_missing_audit.txt`:
the runner had omitted `TILEMEGA_ISL_AUDIT=1`, not discovered a reference leak.
Audited retries have distinct `_audit1` receipts; no failure log was overwritten.

Implementation/verification: `run_element_variants.py:90` runs fresh processes,
`:127` checks the frozen projection, and `:185` compares exact fields.
All rows in `correctness.tsv` can be traced to their hashed `logs/*.txt`.
The absent `max_worker_tasks` runtime field is not claimed verified; this
gate checks the requested task-reference and wait counts.

This is correctness evidence only. No latency or confidence-interval claim
is drawn from these processes. New attention-chunk or placement rewrites
will require new projection and GPU gates; they are not covered here.
