# Round 5 A2: symbolic counts and coordinate repair under validation

This is a partial A2 implementation report. A9 concrete event pricing is now
verified in `../round5_structured.md`; that does not complete A2's remaining
ownership/variant coverage or A6. No fusion/placement implementation
has started. The historical event-price report remains below the current
summary in `../result.md`.

## Recovery after the scoped stop

✅ The user clarified that stops apply only to the failing item and its
dependents, and permitted a concrete repair followed by renewed validation.
Commit `77c942e` aligns runtime task decoding with CG's `(m,n,chunk)` order
(`RuntimeOwnership.h:12`, `GemmStageTaskBody.h:485`). The host's internal
partial→combine windows use contiguous chunk ids in the same order
(`ModelHarness.cuh:960`); `RuntimeProjection.cpp` applies the identical rule.
Partial buffer storage order itself is unchanged. There is no kAll widening,
new barrier or numeric-tolerance change. The independent compile control is
`TILEMEGA_CG_SPLIT_TASK_ORDER`, default 1; 0 preserves the diagnosed failure.

✅ Focused gqa2/seq512/past0/split16, 50 fresh processes **per state**, with
states interleaved within rounds: old order **0/50**, corrected order
**50/50**. Every L0.5/L1 hash remains `8b8a3de9e7f35f9d`; corrected L2
matches it. Corrected waits are 314710 versus 314344 in the old order.
Both numbers describe their respective queue materializations; equality to
old buggy counts is not the corrected projection gate.
Raw data: `../split_order_repair/{correctness.tsv,logs,ptxas,build_manifest.json}`.
`verification.json` independently rereads all 100 process logs, checks binary
hashes, rejects duplicate/missing process keys and confirms PASS/hash records.

✅ CPU `runtime_projection_test` covers both coordinate orders, including
tile-id/chunk decoding and the resulting lifted wait differences; the report
is `../split_order_repair/cpu_projection.{txt,stderr}`, remaining references 0.
Full CTest after A3/A4/A5 additions is 29/29, policy PASS, target-audit five
targets/zero failures (`COST_MODEL/round5_*`). These are not GPU acceptance.

✅ Full repaired matrix **7500/7500 fresh processes** passed: two BF16 models,
seq={1,4,128,512,2048}, past={0,3,512}, split={1,2,4,8,16}, 50 processes per
cell. `../split_order_matrix/verification.json` independently rereads all
7500 logs, verifies the frozen binary hashes and exact Cartesian process
coverage, and requires L0.5/L1/L2 hashes to agree in every repaired process.
This does not complete A2's remaining ownership/variant coverage. Full
symbolic counter comparison has now completed: **15000/15000** task-ref/wait
values match all 7500 repaired process logs, not old split>1 archives.
`../split_order_matrix/symbolic_verification.json` is the independent audit.
After interruption at eight completed CPU queries, two missing queries ran
in `period_symbolic_continuation/`; `merge_projection.py` validates the exact
150-cell union in `period_symbolic_complete/` without overwriting either batch.
These GPU binaries were frozen before the resource-trait aliases and shared
decode helper refactor; the build manifest identifies the precise tested
headers. The ten final-header builds have identical device instructions and
per-function resource records to those frozen binaries, **10/10**
(`../split_order_final_headers/device_comparison.json`). This is device-build
equivalence evidence, not another GPU process claim or host-byte identity.
No performance claim uses correctness-sweep times. CPU counting detours,
timeouts and exact decomposition controls are in `count_decomposition.md`.

The sections below preserve the initial failure and the evidence that led to
the repair; references to “not repaired” describe that historical run only.

## Implemented path

✅ `include/tilemega/Codegen/RuntimePlan.h` and
`lib/Codegen/Codegen.cpp` expose `ReadRuntimePlan`: the same verified CG
analysis used by variant lowering supplies stage identities, dependency
windows, GEMM variants and ownership flags. The parameter domain is read
through the existing SymbolicShapeBridge parser. No generated CUDA is read
by the symbolic implementation.

✅ `lib/Solver/RuntimeProjection.cpp:35` constructs ISL relations and QPs for
runtime task references, lifted wait entries and the longest worker queue.
It covers the existing split rewrite and both tile/element ownership forms,
including the existing attention `(token,head)` stage. Future attention
chunking is not implemented here. Runtime dimensions are restricted to the
exported parameter domain, not a sampled substitute domain.

The wait relation maps each consumer task to `(worker, producer-stage,
event-kind, group)`. Taking its image removes task-local duplicate polls
and repeated polls lifted by `seen[worker]`. Producer and event-kind keys
are disjoint, so their cardinalities can be summed. For singleton events,
same-worker entries are a subset determined solely by `(worker,group)`;
subtracting that subset after union preserves exact deduplication. This
models the current host materializer, including the coordinate bug below.

The longest queue equals the sum of stage counts owned by placed worker 0:
under stage-major modulo ownership, that worker attains the maximum for
every stage. Existing placement permutations only rename workers. A future
task-dependent placement must replace this mapping, not reuse that maximum
formula as an estimate.

## Verified controls and limits

| Control / gate | Observed result | Limit |
|---|---|---|
| Shared-plan refactor, 2 BF16 models × plain/variant CUDA | 4/4 byte-identical | No runtime correctness conclusion |
| Split1, 2 models × 5 seq × 3 past | 3000/3000 exact task_refs/waits comparisons against 1500 archived processes | Archived, not new synchronization evidence |
| gqa2 split2, symbolic seq and past | 15 cells evaluated; seq128/past3 agrees with 50 archived processes | 14 cross-cells initially lacked archives |
| Early QP binding OFF/ON | Imported 1920-row output byte-identical, both unit tests pass | Count/evaluation control, not event pricing |
| BF16 input regression | 1540/1540 bit patterns | Not A6's per-GEMM-stage gate |
| FP32 input regression | 2154/2154 bit patterns | Same limitation |
| Full CTest after changes | 27/27 | CPU/unit/import checks |
| Policy / target audit | PASS; 5 targets, 0 failures | sm120 execution not performed |
| New runtime counter capture | 94 PASS, then 1 FAIL; stopped at 95/150 cells | One process per cell, not 50 per configuration |

Raw evidence: `codegen_equivalence.json`, `matrix/`, `split2_sp/`,
`binding_control/`, `input_bits_{bf16,f32}.{tsv,log}`, `capture/manifest.json`.
`check_codegen_projection.py` preserves the pre-change compiler as a
control; the final rerun still produces identical CUDA in all four cases.

Independent switches: `TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION=0` explicitly
rejects projection rather than returning fallback counts;
`TILEMEGA_EARLY_QP_BINDING=0` preserves late ISL parameter binding. The
runtime kernel itself was not changed by A2. The projection unit exercises
tile/element ownership, force-all, kappa 0/1/2 and symbolic KV `max(S,past)`;
an invalid-grid error exit reports references before=0/after=0. Normal
tools also report `ISL_CONTEXT remaining=0`; this is not a blanket assertion
about every unexecuted error branch (A12.1 remains open).

## Correctness stop: split coordinates do not match the emitted window

✅ `capture_runtime_projection.py:21` uses the archived FP32-partial BF16
binaries and existing SEQSCAN fixtures without changes. It records their
SHA256 values and stops on the first nonzero exit. The failing cell is
gqa2, seq=512, past=0, split=16. The binary SHA256 is
`05089f360fe558868952ca900694a62cb9e200866d026b1599f6f596c7a88346`.

| Comparison | Mismatches | max_abs |
|---|---:|---:|
| L0.5 vs PyTorch golden | 0 | 0.03125 |
| L1 vs L0.5 | 0 | 0 |
| L2 vs L1 | 223287 | 1.1054688 |
| L2 iteration1 vs iteration0 | 222217 | 1.1210938 |

L0.5/L1 hash=`8b8a3de9e7f35f9d`; L2 hash=`f969e6fee82ff801`.
The raw log is `capture/gqa2_s512_p0_k16.txt`. No subsequent GPU cell ran,
no numeric tolerance changed, and no new performance claim uses these times.
This is an inter-level discrepancy, not the closed common-FP32 criterion
artifact. It does not reopen the depth/width BF16 noise-floor investigation.

✅ A static counterexample is independently executable with
`python3 docs/experiments/EVENT_COST/explain_split_projection.py`:

- `lib/Analysis/TaskInstantiation.cpp:170` appends the split chunk axis to
  the output axes. Row-major window construction therefore uses chunk as
  the fastest coordinate (`lib/Analysis/DependencyForm.cpp:43`).
- The archived CUDA's first edge, `gqa2_k16.cu:153`, is
  `at=floor(logical/64)*128`, count=128.
- `GemmStageTaskBody.h:483` instead decodes chunk-major: `chunk=task/tiles`,
  `local=task-chunk*tiles`, then `m=local/tiles_n` at line444.
- `ModelHarness.cuh:1263` applies the window directly to that runtime task
  number, without converting coordinate order.
- With seq512, tile128×128, N512 and split16, task4 is chunk0, M-tile1.
  It reads rows128–255 but waits for rows0–127. 192/256 tasks on this first
  incoming edge have missing required rows, even before poll lifting.

`split_coordinate_witness.json` contains the concrete result and archived
source hash. The parser is diagnostic only, not a new pricing input path.
❌ Inference: the missing producer waits can explain the dynamic failure;
exclusive attribution still requires a coordinate-corrected implementation
and >=50 fresh-process controls. That repair has **not** been made in this
stopped run. Neither broad kAll replacement nor a numeric tolerance change
is a valid resolution.

Thus equality to E2E_SCHEDULE alone cannot certify semantic validity when
the schedule already encodes the wrong task coordinates. A2 must first
reconcile CG and runtime task order, then rerun both exact counter checks
and synchronization correctness; the previously passing split128/3 grid
does not cover this multi-M-tile case.

## Kappa direction: user-approved interpretation, not a passed price gate

✅ Source semantics: kappa0 is the whole-stage aggregate special case;
kappa1 uses singleton fine events and can elide same-worker polls.
The gqa2 counts at seq4/128 are respectively 244/4520 (kappa0) and
500/16292 (kappa1). Kappa1 values match the archived runtime logs; kappa0
values are symbolic/source evidence, not a newly matched GPU sweep.

The user approved retaining these semantics and recording the reasoning.
A9.3 must show a nonzero event-price difference whose direction follows the
actual exact wait counts, not assume 0→1 reduces polls. Positive-kappa
coarsening will be reported separately; even there, the kappa1 same-worker
elision means monotonicity must be checked, not assumed. A9 is unimplemented
and the count difference is **not** its functional acceptance.

## Detours and corrections

Initial whole-graph image counting and late parsing of large unbound QPs
were prohibitively expensive. Interrupted probes were not counted as
successful runs. Splitting disjoint event keys, coalescing images, caching
repeated layer forms and subtracting the local subset reduced the cost.
The split2 symbolic-S/past pilot still takes 136.564 seconds: no claim that
production solve-time cost is solved. The split1 concrete-past runs take
roughly 75–136 seconds each (individual durations in their manifest).

Early binding substitutes complete parameter tokens only, retaining ISL
as the arithmetic/parser authority (`QuasiPolynomial.cpp:54`), and handles
printed implicit coefficients such as `31S`. A local 1920-row OFF/ON control
is byte-identical. Another correction was preserving the CG-exported
parameter domain; unbounded-S exploratory outputs are retained as such,
not treated as full-domain proofs for a different model.

Open: complete split/ownership/variant-shape runtime matrix, semantic split
coordinate repair, A3–A9, all B items, full error-path coverage and actual
FP32-partial combine calibration. See `../../ROUND5_LEDGER.md`.
