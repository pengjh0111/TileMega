# R4 single-writer release recheck

✅ Verified on RTX 4090 / sm_89: 36 arm/cell combinations, each with 50 fresh
processes. `audit_raw.litmus()` reads all 1,800 raw RESULT lines and their
command/exit-code sidecars, not `scan_status.json`. The two positive arms pass
in every cell; each suite's negative mismatches in every process.

`cache` tests the release fence with address reuse and no consumer fence;
`skew` tests the producer barrier with the consumer fence and barrier intact.
The latter delays nonpublisher warps in odd CTAs by exactly 2,000,000 cycles
in every arm. The pilot established sensitivity before the formal 50-process
scan; the delay and expected values were not tuned during the scan.

The two orthogonal negative tests replace the single R3 arm that removed
both barriers and sometimes passed. This is a detector correction under the
user's resume clarification, not a relaxation of the expected mismatches.
The raw insensitive rerun is preserved in `../litmus_recheck/`.

`build.json` holds the exact compile command and source/binary digests.
`litmus.sass` and `census.tsv` show six static BAR.SYNC instructions for
reference/candidate, five for the producer-barrier negative, and one fewer
MEMBAR for the fence negative. Reference and candidate both contain two
static MEMBAR instructions; their placement and participating threads differ.
The litmus block has 256 threads, whereas the model executor has 128.

The original kernel fills, nonce selection, expected sums, address reuse and
monotone event/consumed counters remain unchanged. Its occupancy check refuses
nonresident grids before launching the circular dependency ring.
