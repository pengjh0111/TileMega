# R13 sm120 result index

verified: E4 collects all 195 eligible records in three paired rounds; 102 of the
297 registered records are explicitly unavailable. This is not a full-protocol
PASS. PR page-trace correction r7 has passed terminal/guard/source-identity
review. Performance and correctness records stay frozen; no GPU work remains.

| Table | Contents |
|---|---|
| S1.tsv | Environment, caps, codegen, baseline, PDL CPU replay, solver rejection |
| S1_vendor.tsv | Verified CUTLASS/CuTe header version and SHA; no invented vendor commit |
| S2.tsv / S2_ceiling.tsv / S2_processes.tsv | Native target vs sm89, ceiling, five fresh processes |
| S3.tsv / S3_points.tsv | MB-1 summaries and points, unsupported/quarantined counts |
| S4.tsv | B0/vLLM anchor, TTFT, paired E2E and TPOT speedups, floors, sm89 |
| S5.tsv | Same-group/same-round mechanism effects and cross-architecture signs |
| S6.tsv / S6_kinds.tsv / S6_sm89_reference.tsv | Per-stage and category ledgers, sm89 T4 |
| S6_cross_arch.tsv | Semantic stage excess by past, with each architecture's TL-2 ceiling |
| S7.tsv / S7_tasks.tsv / S7_pages.tsv / S7_sm89_reference.tsv | Steps, TRACE_V2, pages, sm89 T6 |
| S7_cross_arch.tsv | Matched TR-3 arm/cell diagnostic medians, sm89 T6 |
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
error rows; it does not mean the collector failed. The archived original
S7_pages.tsv has no measured rows; it must not be interpreted as zero page wait.
The final S7_pages.tsv has 34 measured observations: two TR-1 last-past records
and two models' sixteen TR-3 steps. TR-1 page counters accumulate sixteen
launches while timestamps retain only the last; mean waits are divided by
sixteen, raw values retained, and incompatible loader fractions omitted.
TR-1 stage/step traces cover all three pasts, but the page exporter retained
only past1000, not past64/575. This limitation is not repaired by interpolation.
Floor interpolation uses actual past and the B0 request-floor proxy, not
measured hardware DRAM traffic. All trace ledgers are diagnostic: matched
uninstrumented overhead <=2% has not been verified. Missing lag stamps are not
proof of zero wait. B16 paged traces and paged loop arms remain unavailable.

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
The clean-guarded r7 raw traces, rebuild commands and unreviewed page accounting
are preserved in `../raw/acceptance_05/unreviewed_r7_evidence.tar.xz`.
SHA256: `328d32d4949c5ee302be4832ebd647934a09f6e7632f0ed293ff5923f863061f`.
Corrected final tables, report and review proofs are sealed in
`../raw/acceptance_05/final_analysis.tar.xz`; its SHA and per-file index are
`final_analysis.tar.json` and `final_analysis.tar.files.json` in that directory.
Extract either archive into a fresh temporary directory as above.

Final closure: `../raw/acceptance_05/completion.json`; CPU checks: 38/38 sm120,
40/40 inherited R13, nine matching calibration stamps. These supersede interim
human-review/CPU fields without altering the original GPU correctness records.
The idle scheduler was stopped normally after every registered node was terminal.
29 frozen/trace SOs, sidecars and generated sources are also persisted at
`/root/shared-nvme/junhuipeng/TileMega_R13_SM120/artifacts/r13_sm120_frozen_binaries.tar.xz`.
SHA256: `ef0c8df08eebc3bd31d8b287d800aac026c7fbd9219dd1153be9b9d1fee6dbbf`.
The bundle's metadata/per-file index are in acceptance_05; the large bundle is
not duplicated into Git. Historical failures are retained. No push was made.
