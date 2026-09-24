# R9 background handoff — 2026-09-24

Verified: all 32 real-model skeleton arms have exactly one live runner at
handoff (five admitted, 27 queued). All eight legacy controls are complete.
`queue_snapshot.json` records the processes observed, not a durable guarantee
that a future process cannot fail. The user requested that the interactive
agent stop after independent validation and queue verification; final review
will resume on the user's next instruction.

## Independent checks

- `independent_ctest.log`: current CPU tests for coupling cache, operator
  classes, variant resources, symbolic Oracle, Plan Skeleton, ready scheduling,
  isolated evaluation and serial/parallel search equivalence. The accompanying
  command JSON records the exit status and source revision when finished.
  At handoff, seven tests passed and the last serial/parallel equivalence test
  remained running independently (exec session 99139); its pending snapshot
  records the PID. Read the final command metadata before claiming 8/8.
- `gpu_admission.log`: four admission/resource-failure guard tests.
- `independent_structure.log`: 14 source/build-command checks pass. The six
  remaining C checks need completed anchored searches or generated results:
  C-2, C-5, C-12, C-13, C-15 and C-17. They are not waived.
- `verify_snapshot.log`: all 20 C checks and all 13 G gates were evaluated.
  Missing real-model results still produce FAIL. This snapshot is not final
  acceptance and does not imply that pending performance comparisons lost.

## Dependency jobs already running or queued

| Work | Owner | Trigger and output |
| --- | --- | --- |
| Full coordinate search for all 32 arms | `run_matrix.py`, existing runners | Admission slot; full domain, P=3 and all legal residency values |
| Actual top-5 residency, re-solve on disagreement, five simulations | Compiler invoked by each runner | Coordinate search completion; resource TSV, final plan dumps and top-3 manifest |
| Top-3 builds and ten fresh GPU processes each | Each runner's `measure.py --top3` | Search completion; GPU lock/admission; raw process logs and measured winner |
| Per-arm Oracle audit | Each runner | Measurement completion; audit log and exit metadata |
| Reuse the dedicated fifth CPU lane | `promote_queued_arm.py --drain-queue` | Lane available; only an unadmitted runner may be promoted |
| Winner-specific exact-set audit, evidence archives, report tables, all gates | `finish_matrix.py` | All arms terminal; calls `audit_winners.py`, archive scripts, `report_tables.py`, `verify.py` |
| Sixteen-section report generation | `render_summary.py --wait-for-matrix` | Completed post-matrix verifier output; `summary.md` and `verify_report.log` |

The final verifier checks generated per-GEMM variant mappings, one-time import,
cache activity, resource-before-skeleton order, top-5 query counts, real-model
hash equality, queue interleaving, attention critical-path attribution and all
performance comparisons. These checks are downstream of completed searches;
there is no reason to hold an independent CPU test until then.

Scripts do not commit, push, repair unknown failures or declare missing gates
passed. Existing background report generation is provisional: final code and
evidence review, any necessary fixes, final documentation and push remain work
for the agent after the user resumes the task. Implementation exists for
SV-1 through SV-7, but this handoff does not certify complete specification
compliance before the dynamic gates run. Declared deviations remain in
`../deviations.md`.

## Manual observation and resume

Read `../matrix_progress.json` for pending arm names. An empty `pending` list
means every arm is terminal, which can include failures; it does not mean
PASS. Then inspect `../verify_after_matrix.log` and `../summary.md`.
If a runner exits unexpectedly without its terminal metadata, inspect its
`runner.log` and `solve.log`; do not start a second copy blindly.

On resume, run the current report renderer once after reviewing terminal
failures (the existing waiting process may have loaded an earlier renderer):

```sh
cd /root/TileMega
python3 docs/experiments/SOLVER_V2/render_summary.py
```

No further interactive polling is scheduled by the agent at this handoff.
The existing queue and completion scripts remain alive independently of the
conversation and record their own output.
