# R9 specification differences and implementation choices

This ledger is provisional while the anchored matrix runs. It does not waive
missing measurements or turn an incomplete gate into PASS. The final report
must incorporate this ledger and any additional differences found by verification.

## Declared specification extension: executor ordering

- **Specification (§4.4/§4.6):** predecessor/successor queries come from the
  exact producer-to-consumer CG coupling; `ready(t,w)` considers those producers.
- **Actual:** `PlanSkeleton.cpp::ExecutionOrdering` additionally composes the
  unchanged executor's `requested_events` with event-group membership. The
  search Oracle uses the union of exact CG dependencies and these executor
  ordering prerequisites, with both relations stored separately.
- **Reason:** the existing wait-window projection can require more ordering
  than exact CG data dependence. The first gqa2 seq=128 top-K materialization
  failed L-c because such a producer appeared later in the consumer's queue.
  Altering the executor, wait semantics, or legality checks is prohibited by R9.
- **Evidence:** `native_union_test.log` independently expands requested events
  at κ=1/2/4 (13,760/13,952/14,336 pairs) and checks set equality and order.
  `reference/` contains the subsequent 120/120 internally equal fresh runs.
  This extension adds ordering; it does not claim those pairs are data edges.

## Disclosed search and measurement forms

| Specification | Actual form | Reason and effect |
|---|---|---|
| §4.7 searches operator classes; §7.3 explicitly permits GEMM-only search | Discrete coordinate descent searches GEMM SemSig classes. Scalar ownership remains the current TaskBody ABI. | No TaskBody changes are allowed. This is the permitted GEMM-only form, not a claim of searching all scalar tile shapes. |
| §4.8 reference-model regression | Reference G-3 uses one disclosed tile shape, all split values, and P=1. Each of three shortlisted candidates has ten fresh internally equal runs in all four cells. | Establishes the reference correctness gate; does not establish full-domain search quality. Real-model arms retain the full CandidateGenerator domain and P=3. |
| §4.0 legacy control is the current solver | Legacy measurements use the deployed R8 five-shape search domain, capacity 12, and the unchanged six heuristics. | The uncontrolled unrestricted audit did not finish and is excluded. This domain must be stated when comparing solve latency; skeleton's larger domain is not a matched search-work comparison. |
| §4.3 each semantic variant has cached resources | Logical cache keys include SemSig/tile/stages/split. The physical compiler cache shares identical template specializations across classes and splits. | Split-K is a runtime field. Recompiling identical source would not add resource information; actual compile counts, command logs, and exact source are preserved. |

## Details not fixed by the specification

- Optional `--search-jobs` evaluates a fixed coordinate's candidates in isolated
  processes. Defaults to 1; current primary runs use 3. Acceptance retains the
  original candidate order and ties. The import occurs once in the parent;
  children inherit immutable semantic state and isolated copies of caches.
  `search_native_cache_test.log` and `search_relation_memo_test.log` compare
  all 26 candidate records and five final schedules with serial evaluation.
- Four whole-solve admission slots bound CPU contention. Waiting before
  compiler launch is recorded separately in `solver_admission.json`; it is not
  solver execution time. Phase durations sum child work, so their sum can
  exceed wall-clock total. The primary physical variant cache is warm across
  diagnostics and arms, and compile counts report actual new compilations.
- Exact expression and relation memoization stores complete input keys and
  immutable values within a bounded search scope. It does not cache successful
  verification or skip L-a/L-b/L-c/L-e checks. The short relation-memo profile
  is diagnostic only; it is not an anchored performance observation.
- General Oracle queries may locally enumerate bounded candidate coordinates
  only after exact membership testing of every emitted point. A bounding box
  is never substituted for the dependency set. Large sparse fibers retain ISL
  enumeration. This is the exact local enumeration permitted by §4.4, not a
  claim that a General edge has been proven Rectangular.
- Three pre-existing frontend defects were repaired: F32 RoPE constant storage,
  translated head-local reductions, and alternative valid Qwen FX emission
  order. Static-export default dimension-role symbols were also preserved in
  the split importer. No TaskBody, event, barrier, old heuristic, simulator
  semantics, or CUTLASS implementation was changed.

## Excluded attempts

`before_execution_order/`, `before_isolation/`, `before_bulk_union/`, and the
named `*_profile/` directories are diagnostic or interrupted attempts. They
must not supply completed real-model latency or performance claims. Admissions
that failed before code generation remain as debugging evidence. Invalid new
shortlist candidates remain recorded and cannot become an arm's selected
result; an entirely invalid shortlist fails that arm. A regression in an
existing reference configuration remains the prompt's global-stop condition.


## GPU resource admission and a rejected control attempt

Verified: the first Qwen seq16 control attempt had four pre-output CUDA
allocation failures at `ModelHarness.cuh:2585`, followed by six internally
bit-exact runs. The ten-process gate failed; its six timings are not a control
result. The entire attempt is retained under that cell's `failed_attempts/`.
Recovery repeats all ten fresh processes, not just the failed four.

The measurement runner now waits for three consecutive device observations
with utilization at most 5% and available memory at least fixture bytes plus
2048 MiB. This headroom is an admission estimate, not a measured peak-memory
claim or a changed acceptance gate. Observations are saved in
`gpu_admission.jsonl`. External GPU users can still race admission; any new
failure remains in the raw logs. Numeric/hash failures and unexplained exits
are not automatically retried. `gpu_admission_test.log` exercises these guards.


## Additional admission lane after concurrency equivalence

Verified CPU fixture: serial and `--search-jobs=36` evaluation produced identical
116 candidate records, top-five keys, and final worker/slot/start/end tables;
each imported once (`search_isolation_36_test.log`). On 2026-09-24 the still
queued Llama seq1/k8 arm was moved to a fifth admission slot with 36 workers.
The four admitted seq4 arms were not interrupted and retain three workers.
The compiler snapshot, full 1218-candidate class domains, P=3, residency
sweeps and top-K procedure are unchanged. Original queued launch metadata is
retained in that arm's `queued_launch_history/`. Compare wall times with the
recorded per-arm concurrency and warm-cache state, not as equal-CPU budgets.
No anchored speedup is asserted from the CPU fixture.

The two control cells Llama seq64 and Qwen seq4 had busy-GPU pre-measurement
snapshots (93%/100% utilization, 5157/43962 MiB used). These snapshots alone
do not prove interference throughout the full sample. Their original valid
internal-equality samples are retained under `excluded_attempts/`; complete
fresh ten-process timings are remeasured with resource/idle admission. This
is not selective removal of slow rounds, and numerical failures cannot be
retried by this route.


The fifth slot is now a dedicated continuation lane: after one promoted arm
releases it, `promote_queued_arm.py --drain-queue` can promote another still
unadmitted arm with the same 36-worker setting. A singleton supervisor lock
and an explicit `--solver-slot=4` keep at most one such high-parallelism search
admitted. Active searches are checked while their queue parent is frozen and
are resumed rather than cancelled if admission raced the check. Original
queued launch metadata is archived; search-domain and compiler bytes are
unchanged. `dedicated_slot_test.log` checks exclusivity, release and range
validation. This is CPU admission policy, not a new placement/search policy.
