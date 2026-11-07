# SM90X1 parallel execution acceptance

[verified] Snapshot: 2026-11-07 15:46 UTC. Parallel dispatch and accepted
execution on different GPUs were observed. The campaign remains incomplete;
abnormal process exits remain an open reliability issue.

[stated] The user authorized concurrent CPU preparation, concurrent tests on
different idle GPUs and removal of same-card/whole-machine restrictions.
`configs/bench/sm90x1_parallel_policy.json` records this authorization; every
X1 cell now uses effective class `trial`, including MoE. The original guard,
GPU allowlist, frozen cell list and candidate selection remain in force.

## Executed checks

[verified] Artifact validation checked actual binary hashes and identities for
165/165 native sweep builds and 165/165 stage-level-2 trace builds. Native
bindings passed the host ABI check; trace builds are performance-ineligible.
Measurement HEAD remains `303e8e94da804bc6cdb28605964bfb2d25b4f9e1`.
The kernel/runtime/compiler source directories and H800 target have no diff
against that HEAD. Tooling commits wait until measurement workers finish.

[verified] The CPU suite passed 24/24 tests. Separate fixtures each used five
fresh processes: trace receipt reuse 5/5, postprocessing flow reuse 5/5,
preflight handoff 5/5 and active-child retention 5/5. These are CPU fixtures;
actual GPU execution is evidenced separately below.

[verified] Two seven-sample windows recorded queue state, physical GPU locks,
guard state and NVML processes at approximately 30-second intervals:
15:35:25–15:38:31 and 15:42:49–15:45:51 UTC. X1 held trial claims on GPUs
0, 1 and 3 while GPU2 was used by another project. Each window gained one
accepted trace coordinate, with two fresh processes per coordinate.

[verified] Accepted guard logs show overlapping sampled CUDA-context intervals
on GPU3 (`qwen3_1_7b-B16`, W correctness) and GPU0 (`llama3_2_1b-B16`, L
correctness) from 15:42:48.484 to 15:43:00.415 UTC: 11.931 seconds. Both
complete correctness units passed their correctness and hardware audits.
The cumulative scan found 35 pairs of overlapping intervals on different
GPUs among 214 accepted process intervals; these are interval pairs, not
35 distinct experiments. This establishes concurrent CUDA contexts rather
than continuous simultaneous kernel activity. Host PID ownership remains
provisional in the container namespace.

[verified] Accepted required-scope progress:

| Quantity | 15:27:59 UTC | 15:46:01 UTC | Required |
|---|---:|---:|---:|
| Trace coordinates | 79 | 86 | 165 |
| Fresh trace processes | 158 | 172 | 330 |
| Cells with reportable non-trace arm timing | 3 | 3 | 14 |

## Repairs and remaining limits

[verified] The supplementary postprocessing controller previously submitted a
cell once and did not resubmit its failed flow. It now persists failure
records and permits at most three failed flows per cell, with a 30-second
retry wait, for recorded abnormal process exits only. Explicit correctness
or build failures stop automatic retries. Existing complete units are reused;
failed attempts remain excluded and retained raw. Qwen1.7B B16 W correctness
completed after this controller replacement. Its later L attempt failed and
was resubmitted within the same bound.

[verified] Queue and scheduler polling use five seconds. Polling is distinct
from launch latency: physical locks, guard preflight/cooldown and CPU weight
preparation can delay a CUDA process. Checkpoint generation still occurs
inside the guarded unit, so an X1 claim can temporarily have no NVML process.
The original campaign parent is deliberately stopped while the independent
pipeline, ready queue and postprocessing drivers execute; this is not the
only work-driving process. The CPU build producers have finished; the trace
producer's stale running state was reconciled after artifact verification.

[verified] Exit code 247 has affected correctness and timing attempts in
multiple cells. Its cause remains unresolved. Inspected cgroup memory events
reported zero OOM kills; that observation does not establish the cause of an
exit. Failed batches cannot contribute reportable timing. Additional owner
count checks supplement the original guard but cannot prove host PID ownership.

[verified] Raw evidence remains under ignored `runs/sm90x1/`:
`parallel-acceptance-1535.json`, `parallel-acceptance.json`,
`parallel-acceptance-samples.jsonl`, `full-build-validation.json`,
`cpu-tests-ready-queue.log`, dated `progress-*.json` and individual unit receipts.
The ready and postprocessing queues remain running; 79 trace coordinates and
11 cells' complete arm timings were still outstanding at this snapshot.

## Follow-up at 16:30 UTC

[verified] Five samples over 81 seconds (16:28:52–16:30:13 UTC) found X1
trial claims on all four visible GPUs. The ready and postprocessing state
files were 1.5–4.1 seconds old at sampling. Concurrent CUDA contexts were
observed on different GPUs while other guarded units prepared CPU weights
or performed preflight checks. This does not imply four simultaneous kernels.
The sampled ready queue gained one accepted trace coordinate.

[verified] Progress at 16:30:10 UTC was 104/165 trace coordinates (208 fresh
processes) and 7/14 cells with accepted non-trace arm timings, compared with
88 coordinates and four timed cells at 16:00:23 UTC. The newly completed
timing cells were Qwen1.7B B1/B16 and Llama1B B1. There were 61 trace
coordinates and seven cells' complete arm timings outstanding.

[verified] Accepted final timing units also overlapped on different GPUs.
Qwen1.7B B1 on GPU3 and Qwen1.7B B16 on GPU0 had overlapping sampled CUDA
contexts from 16:14:37.469 to 16:14:40.430 UTC (2.962 seconds). Both complete
timing units passed their hardware audits. The scan found four overlapping
interval pairs across the accepted final timing units; these are interval
pairs rather than four distinct experiments. This is evidence of actual
concurrent final timing processes, beyond merely admitting them to a queue.

[verified] Llama3B B1 advanced from CPU analysis through arm construction and
compilation to P-CORR during this follow-up. GPU0 later encountered foreign
occupancy between trace processes; its guard returned 75 and the scheduler
retained a pending retry. Other units continued, and the postprocessing
controller reported no blocked cells. Earlier code-247 attempts remain
excluded; their underlying cause has not been established.

[verified] Follow-up raw evidence: `health-20261107T162852Z.json`,
`health-latest.json`, `check-health.log`, dated progress snapshots and the
referenced accepted timing unit receipts, all under `runs/sm90x1/`.

## Follow-up at 16:57 UTC

[verified] Five samples over 81 seconds (16:55:58–16:57:19 UTC) again found
X1 trial claims on all four visible GPUs. Ready and postprocessing state ages
were 0.14–3.50 seconds. GPUs 0 and 2 served Llama3B B1/B16 correctness or
timing units while GPUs 1 and 3 served Llama8B B1 traces. Sampled CUDA
contexts overlapped across these tasks. No cell was listed as blocked by the
postprocessing controller. The ready queue gained one accepted coordinate.

[verified] At 16:57:29 UTC, accepted progress was 118/165 trace coordinates
(236 fresh processes), up from 107 coordinates at 16:31:54 UTC. Accepted final
timing remained 7/14 cells. During the sample window Llama3B B16 completed
P-CORR and entered its final timing unit; Llama3B B1 retained accepted W
correctness and continued L correctness. Remaining scope was 47 trace
coordinates and seven complete cell timing units.

[verified] Raw evidence is retained in `health-20261107T165558Z.json`,
`check-health.log`, dated progress snapshots and the individual unit records
under `runs/sm90x1/`. The frozen runtime/compiler inputs remain unchanged;
this check required no scheduling code change. Earlier abnormal-exit causes
remain unresolved and failed attempts remain excluded.

## Follow-up at 17:54 UTC

[verified] At 17:53:59 UTC, accepted progress was 143/165 trace coordinates
(286 fresh processes) and 9/14 complete non-trace cell timing units, up from
119 coordinates and seven timed cells at 16:58:33 UTC. The newly timed cells
were Llama3B B1/B16. Remaining scope was 22 trace coordinates and five complete
cell timing units; 19 remaining coordinates belong to the two MoE cells.

[verified] Five samples over 81 seconds (17:52:28–17:53:49 UTC) found X1
trial claims on all four visible GPUs. Queue state ages were 0.29–3.59 seconds,
and the postprocessing controller listed no blocked cells. GPUs 0/1 served
Mistral B1 and Llama8B B1 correctness, while GPUs 2/3 served Mistral B16 traces.
Sampled CUDA contexts overlapped across these tasks. Complete trace-unit
counts stayed constant during this short window while fresh processes advanced.

[verified] Llama8B B1 retained accepted W and L correctness and was checking
Wstar. Mistral B1 was checking W. The runtime/compiler inputs and frozen cell
list still had no diff against the measurement HEAD. Earlier abnormal-exit
causes remain unresolved; no failed attempt is promoted to reportable timing.
Raw follow-up evidence is `health-20261107T175228Z.json`, dated progress
snapshots and individual unit receipts under `runs/sm90x1/`.

## Completion check at 19:38 UTC

[verified] All 12 required dense cells now have accepted correctness and
non-trace final timing units. Required-scope totals are 12/14 timed cells and
153/165 trace coordinates (306 fresh trace processes). The campaign is not
complete: Qwen3-30B-A3B B1 has 7/10 accepted trace coordinates and B16 has
0/9, with its first coordinate in progress. These cells still require completed
trace analysis, arm selection, correctness and final timing.

[verified] Ready and postprocessing queues remain live and list no blocked
postprocessing cells. One MoE attention-v8 trace attempt ended in a preflight
handoff and has no accepted unit receipt; it remains outside the accepted
counts. Final report sealing and deferred commits have not run. The dated
`progress-20261107T193826Z.json` snapshot and per-unit records under
`runs/sm90x1/` retain this check's evidence.

## Idle-card investigation at 20:00 UTC

[verified] Required-scope progress is 156/165 accepted trace coordinates
(312 fresh processes) and 12/14 reportable cells. All 12 dense cells are
complete. MoE B1 has all ten trace coordinates and has entered correctness;
B16 has four variant flows in flight and five further coordinates ready.
The ready queue polls every five seconds; these are concurrent flows.

[verified] The old MoE launcher claimed a GPU before regenerating the same
61,064,245,248-byte synthetic CPU checkpoint for every fresh child. This
caused claimed cards to appear idle during CPU preparation. A CPU-only
service now retains read-only descriptors for the same checkpoint, after
actual SHA256 validation of all 114 shards against accepted B1 W trace
weights. New matching X1 jobs borrow per-process checkpoint paths and still
launch fresh guarded GPU children. Existing children were left running.

[verified] Five fresh CPU reader processes passed 5/5, including reads after
the original source closed and checks that service closure waits for active
readers. The full X1 CPU suite passed 31/31. Tests also preserve the trace
launch wrapper, prefill/decode arguments, correctness fresh index, guard
requirement and X1/model-only scope. Raw evidence includes
`checkpoint-cache/validated.json`, `cpu-tests-checkpoint-full.log` and the
accepted checkpoint reference recorded in the validation proof, under
`runs/sm90x1/`.

[verified] The first cached GPU trace child on B16 qkv-v2 completed with
exit code zero at 20:01:14 UTC and advanced to its second fresh process. Its
checkpoint manifest records reuse and the accepted checkpoint hash. The
complete two-process cached trace unit remains pending at this snapshot. Three
sampled GPUs retained about 70,545 MiB each at 0% utilization while their
CPU children continued running. A zero-utilization card with that allocation
is unavailable to another test under the user's empty-card requirement.

[inferred] Growing CPU memory and sustained CPU execution after weight
upload indicate further preparation work, consistent with loading and host
plan construction. The checkpoint change removes repeated random generation,
but does not establish an end-to-end speedup or eliminate all preparation
inside the guard. Runtime bodies, compiler, frozen candidates and all
measurement binaries remain unchanged.

[verified] Five hardware/queue samples at 20:00:52–20:01:33 UTC show
continuing CPU execution and queue updates less than five seconds old.
Accepted traces increased to 157/165 (314 processes); timed cells remain
12/14. B1 correctness has started and its per-process checkpoint manifest
confirms cache reuse. GPU utilization and allocations changed as children
finished and new guarded children started. Raw evidence is
`idle-audit-20261107T200133Z.json` and `checkpoint-cache-acceptance.json`
under `runs/sm90x1/`. Host GPU PID ownership remains unresolved; these
samples do not establish continuous overlapping kernel execution.

## Completion check at 20:40 UTC

[verified] Required testing is incomplete: 163/165 trace coordinates
(326 fresh trace processes) and 12/14 reportable final timing units are
accepted. All 12 dense cells remain complete. Since the previous 157/165
snapshot, six additional trace coordinates have been accepted.

[verified] MoE B1 has all ten trace coordinates; W passed C1/C2 across
five fresh processes, while four L correctness children have completed and
the fifth is running. MoE B16 has seven of nine accepted trace coordinates;
`experts-v7` and `experts-v8` remain, followed by analysis, arms, correctness
and final timing. The postprocessing queue lists no blocked cells. GPUs 0
and 3 were sampled at 100% utilization during this check; this does not
establish continuous kernel overlap or resolve host GPU PID ownership.

[verified] Cached B16 qkv-v1 and qkv-v2 trace flows now each have a complete
two-process accepted unit and a passing hardware audit. Deferred tooling
hashes still match, and frozen runtime/compiler inputs, target and cell list
retain no diff from the measurement HEAD. Final report sealing and commits
remain deferred. The completion audit and dated progress snapshot under
`runs/sm90x1/` retain the exact counts and remaining coordinate names.

## Completion monitoring and publication

[stated] The user authorized persistent completion monitoring, Prompt result
organization and publication of the `tilemega` branch to `origin`.

[verified] The watcher is CPU-only and retains the existing campaign and
finalizer. It polls every 30 seconds and publishes only after all 165 trace
coordinates and 14 reportable cell units are accepted and measurement
workers, the checkpoint service and existing finalizer have exited.
Validation rechecks non-trace arm hashes and HEAD, prefills, C1/C2 receipts,
classes/GPUs, hardware audits, interleaved order, complete CSV coverage,
X/Xprime and Y/Yprime aggregates, and P-PAIR decisions. All failed attempts
remain under ignored `runs/sm90x1/`. Optional Phase 3 is not launched.

[verified] The publication path uses a normal fast-forward push and confirms
the remote branch tip afterward. A local bare-repository fixture verified
publication and rejection of dirty workspaces and divergent remotes. Five
fresh CPU processes attempted the already-held monitor lock and were
rejected 5/5; after release a fresh process acquired it. The startup
preflight revalidated all 12 already-complete dense cells (146 trace
coordinates and 22 unique correctness units), retaining a subset label.

[verified] The GitHub push dry-run passed before watcher startup; no actual
GitHub push was performed during this setup check. Raw watcher state,
validation receipts, test logs and later push confirmation live under
`runs/sm90x1/`. An incomplete terminal state or validation/publication
failure is recorded as `needs_review`, preserving its reason.

[verified] Startup checks passed all 44 X1 CPU tests and the native-sweep
C++ unit test. Current figure normalization/scatter values match their
measured arm/candidate tables. All 165 required trace coordinates are now
accepted; both MoE cells still require final correctness/timing completion.
The watcher is launched with CUDA_VISIBLE_DEVICES empty, separate from GPU
measurement processes. Its source and tests are included in the deferred
focused-commit manifest; measurement HEAD remains unchanged until quiescence.

## User-authorized remaining correctness waiver

[stated] The user explicitly requested: “跳过这个正确性验收，直接进行性能计时测试”.
This exception applies only to the remaining Qwen3-30B-A3B B16 cell.
The earlier publication criteria above are amended for this one cell.

[verified] At this handoff, all 165 diagnostic trace coordinates (330 fresh
processes) and 13 of 14 formal timing units were accepted. B16 had no
complete P-CORR receipt. Its remaining correctness drivers and guarded
process tree were stopped after recording their PID/start identities;
partial attempts remain in ignored runs/sm90x1/. The final native sel
unit has started through the existing scheduler and GPU guard, with the
original 15-entry interleaved order and five fresh processes per arm.
No accepted timing or correctness result was manufactured.

[verified] The authorization is recorded in the separate execution policy
and copied into the final timing spec with performance_eligible=false.
SM90X1_arms.csv retains all 14 cells but leaves uncertified B16 TPOT and
C1/C2 fields blank; strict B16/overall paper aggregates remain pending.
SM90X1_uncertified_arms.csv records its actual timing separately when
complete, and SM90X1_summary_user_waived.csv supplies distinct full-scope
aggregates with the waiver and eligibility explicitly recorded. The
monitor checks both forms before the authorized ordinary push.

[verified] All 48 X1 CPU tests passed after the waiver implementation.
The added fixtures verify exact-cell authorization, rejection without
authorization, no fabricated correctness receipt, the original timing
order, all-cell coverage, strict exclusion and supplemental aggregates.
No runtime/compiler body, model constants, search rules, frozen cell list
or measurement HEAD changed. This snapshot verifies launch, not completion
of the B16 timing unit; later completion/push receipts remain in runs/.

## Selection processes within the last cell

[verified] A subsequent queue audit found that earlier parallel execution
was limited to different cells. Each sel unit still fixed one GPU and
chained all process jobs with after dependencies. That remaining restriction
was not removed by the earlier idle-card policy.

[stated] The user requested parallel testing of these remaining jobs.
The X1 execution policy now enables process-level selection dispatch with
up to four workers. Logical W/L/Wstar round labels and five fresh processes
per arm are retained; actual launch and completion order may overlap. This
amends the original one-card, serial interleaved selection protocol.
The previously authorized B16 P-CORR exception remains explicitly labeled.

[verified] The first two accepted processes and the already-running third
process were preserved. The original scheduler was restricted to that
prefix; its waiting parent was transferred without stopping its guard or
measurement child. Twelve unstarted processes moved to independent jobs,
each claiming an idle allowed GPU through the unchanged claim/guard path.
The parallel coordinator now records individual device assignments and
merges only complete, audited process receipts. The arm CSV records the
actual set of GPUs per arm and its parallel execution mode.

[verified] All 53 X1 CPU tests passed. Five fresh coordinator processes
passed 5/5, each retaining fifteen rounds with four worker slots. Other
fixtures verify adoption, dependency isolation, device/round/artifact
provenance rejection and qualified CSV rendering. At the live snapshot,
four guarded selection jobs occupied GPUs 0, 1, 2 and 3 and had sampled
running CUDA owners. Raw handoff and acceptance snapshots remain under
runs/sm90x1/. This verifies launch/overlap, not final completion or a
performance improvement. Host PID ownership remains provisional.

[verified] The completion watcher now checks the authorized execution mode,
all per-process device identities and guards, complete logical-round
coverage, actual TPOT values and per-arm GPU sets before publication.
The frozen source cohort, candidate coordinates and binary hashes remain
unchanged; all tooling commits still wait for measurement quiescence.
