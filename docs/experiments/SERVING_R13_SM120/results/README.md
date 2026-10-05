# R13 sm120 result index

verified: E4 collects all 195 eligible records in three paired rounds; 102 of the
297 registered records are explicitly unavailable. This is not a full-protocol
PASS. PR page-trace correction r7 is pending; performance records stay frozen.

| Table | Contents |
|---|---|
| S1.tsv | Environment, caps, codegen, baseline, PDL CPU replay, solver rejection |
| S2.tsv / S2_ceiling.tsv / S2_processes.tsv | Native target vs sm89, ceiling, five fresh processes |
| S3.tsv / S3_points.tsv | MB-1 summaries and points, unsupported/quarantined counts |
| S4.tsv | B0/vLLM anchor, TTFT, paired E2E and TPOT speedups, floors, sm89 |
| S5.tsv | Same-group/same-round mechanism effects and cross-architecture signs |
| S6.tsv / S6_kinds.tsv / S6_sm89_reference.tsv | Per-stage and category ledgers, sm89 T4 |
| S7.tsv / S7_tasks.tsv / S7_pages.tsv / S7_sm89_reference.tsv | Steps, TRACE_V2, pages, sm89 T6 |
| S8.tsv / S8_sm89_reference.tsv | Failed joint selection and frozen partial candidates, sm89 T9 |
| S9.tsv | Smoke, C-1/C-2, scoped fresh-process results and CPU replays |
| S10.tsv | All preregistered prediction observations, main range and saving-us checks |
| measurements.tsv / medians.tsv | Canonical paired samples, medians, sample lists and ranges |
| acceptance_coverage.tsv | Every registered arm's eligibility and unavailable reason |

The 384 MiB S7_tasks.tsv, 23 MiB copied sm89 S6 reference, and detailed 14 MiB
S6.tsv are retained inside the compressed evidence archive rather than duplicated
as huge Git text objects. They also remain at their generated local paths.
All original result files (including other smaller tables) are archived before r7.
The small tables and semantic ledger are additionally checked in for convenience.
An empty `not_collected` placeholder in analysis_errors.tsv means zero analysis
error rows; it does not mean the collector failed. S7_pages.tsv originally has
no measured rows; it must not be interpreted as zero page wait time.

Archive: `../raw/acceptance_04/original_e4_e5_e6.tar.xz`.
SHA256: `19058e454dece28b55f987fc2f1120fdaa1a0b20d67e3ebac052199cb7433f13`.
File index: `../raw/acceptance_04/original_files.json` (SHA256 per file).
Member prefix: `root/tilemega-r13-sm120/docs/experiments/SERVING_R13_SM120/`.

To inspect archived large tables without overwriting current/generated results:

```bash
evidence_dir=$(mktemp -d)
tar -xJf ../raw/acceptance_04/original_e4_e5_e6.tar.xz -C "$evidence_dir"
```

Run from this results directory. The original worktree and scheduler snapshots
are reconstructed under the temporary directory, not over the live experiment.
Do not run the old archived queue commands or replace the current summary.
After r7, clean-guarded PR trace replacements feed regenerated S6/S7; originals
remain in this archive, and the final acceptance must archive the replacements.
