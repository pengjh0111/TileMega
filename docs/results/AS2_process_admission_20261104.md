# Process-based NPU admission override — 2026-11-04

`stated`, latest user instruction: ignore the Health field; use a device when
its device process list is empty. This supersedes both the former HealthOK
gate and the former permission to run functional jobs alongside external
processes. The scope remains five dense models/B1/B16; MoE is deferred and
frozen baselines are unchanged.

Override SHA256:
`7f5dbc3ba11bb8dd56eff71f5c89d562082eacb4d5d6f8849915c3b9985daf9e`.

`verified-code/run CPU`, commit `7375401a1`:

- `as1_run.py` (shared by AS-2) and `as2_queue.py` parse the actual installed
  `npu-smi info` process table per physical device. Health and utilization are
  retained in evidence but neither gates admission. A process occupying0MiB
  or reporting0% utilization still counts as a process.
- A complete empty list admits a job, including an Alarm/100%-utilization
  fixture. Any listed process defers our job. Missing, truncated or malformed
  process information is unknown and is never treated as an empty list.
  Installed output with both `-` and `=` table endings is covered.
- The wrapper rechecks process occupancy immediately before creating its
  child, after storage/capability checks. A newly arrived process produces
  `blocked-busy` with no child. The queue retains that observation and retries
  later; actual execution failures/timeouts are not silently retried.
- Automatic selection prefers an observed process-idle device, then free
  HBM. Existing finite timeouts, private disk/output budgets, implementation
  capability checks, sufficient HBM and single own job/device limits remain.
- The TaskBody watch uses the same criterion for functionality and timing.
  A Health value alone no longer stops it after a completed job. Actual ACL
  initialization errors, aborts and timeouts retain their failure handling.

This only controls when our own job starts. It creates no device lock,
exclusive mode or reservation and performs no action on other users' jobs.
Other jobs may arrive after an observation; their presence remains evidence
when interpreting timing. Historical Alarm/error/slow-request records and
`blocked-health` receipts are retained without changing their original status.

`verified-run CPU`:25 checks pass (10 queue,4 finite-watch,11 runner). They
cover process-idle Alarm admission, occupied OK rejection, prelaunch arrivals,
unknown query results, malformed tables, retained/requeued admission attempts,
and termination of only test-owned CPU children. Mocked launch receipts are
explicitly labelled CPU fixtures; these are not device numerical/C3 evidence.
No NPU job, baseline or compiler build is launched by this policy change.

The last successful live observation shows processes on both cards, including
processes on card0 with HealthOK/util0. A later bounded query times out; the
separate direct query is interrupted and exits130. Its own isolated session
is tracked; no external query/process is signalled. Unknown query output is
not evidence that the cards became idle. No further query is blindly retried.

Raw user override/hash, snapshots, CPU test logs, query timeout and own-session
record: `runs/as2/inventory/process_admission_20261104_v1/`. The override is
also retained outside Git under
`tilemega-as2/Prompt/Process_idle_override_20261104.txt`.

The earlier dense checkpoint's Alarm hold is superseded. The pending bounded
QK diagnostic and residual probe now wait for a confirmed empty process list;
their source/image identities, finite limits, numerical requirements and
disabled defaults are unchanged. No new TPOT or device correctness claim is
made. Current resume: `runs/as2/resume.json`.
