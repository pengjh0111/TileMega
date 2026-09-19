# R7 B1 — cross-task shared-memory pipelining

Written for: the TileMega maintainers reviewing R7 §5.2.

This file is built up as B1 proceeds. **B1-a** (correctness on the six cells),
**B1-b** (the occupancy budget, before and after) and **B1-d** (sigma wired
into the solver) are complete; B1-a's Llama half, B1-c and B1-e are being
measured and are reported as their runs land.

## B1-a — correctness, and how much of each cell the mechanism reaches

Every cell, both arms, 50 fresh processes each: **600/600 PASS**, one binary
per arm (`correctness_head/*/r*.json` carries the sha256; each arm has exactly
one). Every binary in this table was built at `49098a425`, so the arms differ
in the mechanism and in nothing else.

Each run prints `E2E_PREFETCH`, which walks the whole schedule applying the
same rule the worker applies per slot: `declared` counts slots whose kind
names a prefetch operand, `issued` those where the operand is also on the
derived frontier, 16-byte aligned and inside the page, and `queue_heads`
those issued slots that are first in their worker's queue and therefore have
no earlier body to hide under.

| cell | slots | declared | issued | queue heads | page | issued/slots | overlapping |
|---|---:|---:|---:|---:|---:|---:|---:|
| gqa2_s4   | 648   | 16   | 16   | 4   | 1024 | 2.47% | 12 |
| gqa2_s128 | 6400  | 512  | 512  | 128 | 1024 | 8.00% | 384 |
| mha4_s4   | 1520  | 32   | 32   | 4   | 1024 | 2.11% | 28 |
| mha4_s128 | 16384 | 1024 | 1024 | 128 | 1024 | 6.25% | 896 |
| real_s4   | 12208 | 32   | 32   | 4   | 8192 | 0.26% | 28 |
| real_s128 | 49280 | 1024 | 1024 | 128 | 8192 | 2.08% | 896 |

Two things to read out of it, both of which matter before any timing is
interpreted:

1. `declared == issued` in every cell. Nothing is compiled in and then refused
   at runtime, which is the failure the page measurement found at 1024 bytes on
   the real cells (`declared=32 issued=0`, `raw/real_s4/page_probe/`) and
   which the page in `run.py` exists to avoid.
2. The mechanism reaches a **small fraction of the schedule**, and the fraction
   that can actually overlap is smaller still: 28 of 12208 slots on `real_s4`.
   Only RMSNorm and QKNorm bodies declare a prefetch, and only their scale
   operand is on the frontier. This is the leading input to B1-e's diagnosis
   and is stated here, before the timings, so that it cannot be recruited
   afterwards as an explanation of whatever they show.

### The Llama population

B1-a's third population is the maximal connected Llama graph: R6's admitted
geometry, fixture, seed and 0.0231875014 tolerance, replayed at HEAD by
`llama.py` and built with the mechanism at a 4096-byte page. **50/50 PASS on
`prefetch`, 50/50 on `inline`.**

The same run reports `E2E_PREFETCH slots=15486 declared=0 issued=0`. That is
`declared`, not `issued`: this graph carries no normalization stage at all --
R6 admitted it with both per-layer norms passed in as graph inputs -- and only
RMSNorm and QKNorm bodies declare a prefetch operand. The mechanism is compiled
in and has no client here, the solver prices it at zero for the same reason,
and the replayed geometry's drift from R6's is neither this round's pricing nor
the page. F-228 carries the controls and the stage census.

### The SEQSCAN subset

B1-a's second population is R5's twelve-case sequence scan: the four reference
cells at `s1_p0`, `s128_p512` and `s2048_p0`, both arms. **24 arms, 50 fresh
processes each, 1200/1200 PASS**, 24 distinct binaries
(`raw/seqscan/*/seqscan/*/*/r*.json`).

The subset is R5's only if the plans are R5's. JOINT's plans were projected
before `no_producer` existed, so a prefetch build against them fails on the
guarded field rather than quietly prefetching nothing; `seqscan.py project`
replays JOINT's own recorded command and environment against a driver built at
HEAD, and `seqscan.py verify` strips the guarded field back out and compares
byte for byte. **12/12 cases identical without the guarded field**
(`raw/seqscan/regeneration.tsv`). Nothing else about the plan was touched.

What the subset adds beyond a second pass count is the sequence sweep of the
mechanism's reach, which the six timed cells do not cover:

| case | slots (gqa2 / mha4) | issued | queue heads | issued/slots |
|---|---:|---:|---:|---:|
| `s1_p0`     | 546 / 1244       | 4 / 8         | 1 / 1     | 0.73% / 0.64% |
| `s128_p512` | 6400 / 16384     | 512 / 1024    | 128 / 128 | 8.00% / 6.25% |
| `s2048_p0`  | 102400 / 262144  | 8192 / 16384  | 128 / 640 | 8.00% / 6.25% |

`declared == issued` in all twelve, as in the timed cells. The share saturates
at the seq-128 value and does not keep growing with sequence length: the
declaring bodies are per-token normalizations, so they scale with the schedule
rather than within it. The `s1_p0` column is the decode shape, and it is the
one where the mechanism reaches almost nothing — four issuing slots out of 546,
of which one is a queue head with no body to hide under. The two `s2048_p0`
rows differ only in `queue_heads` (128 vs 640) because the two cells carry
different placements of the same graph; the issued counts are a property of the
model and the case, not of the cell.

## B1-b — what a prefetch page costs

✅ Verified on RTX 4090 / sm_89. Reproduce with `occupancy.py`; raw tables in
`raw/occupancy.tsv` (per page) and `raw/occupancy_cells.tsv` (per cell).

B1-b was measured **before** the mechanism was built, on purpose. The page B1
appends after the `TaskSmem` union is dynamic shared memory on the L2 worker
kernel, so it can cost residency, and the grid is sized from that residency
(`ModelHarness.cuh:2931-2946`) — a cap the launch cannot meet is *rejected*,
not silently lowered. Building the mechanism first and discovering the budget
afterwards would repeat the error B0 was created to fix.

### Method

Two independent answers per cell, cross-checked:

- **the driver**, `cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real cubin
  built with the shipped flags (`JOINT/measure.py:32-41`'s `PROTOCOL`, plus the
  cell's frozen `kappa`/`residency`/`placement`). This is the same query
  production makes through `TargetSpec::ActiveBlocksPerSM`
  (`lib/Target/TargetSpec.cpp:461-471`). Going through the driver API on a
  cubin is what lets the page size vary without editing the cell source;
- **F-40's closed form** (`ORACLE/occupancy.sh:23-45`), generalised off its
  hard-coded 256-thread/8-warp shape, because these kernels run **128** threads.

Page 0 is anchored against the harness's own `E2E_RESOURCE` line from the runs
that actually executed (`PHASE2/raw/<cell>/correctness/selected/r0.log`). It
agrees on all four cells, for both the L1 and L2 kernels.

### Before

| cell | configuration | residency used | regs (prod / instrumented) | smem | L2 CTA/SM |
|---|---|---:|---|---:|---:|
| gqa2 s4 | `32x16x64s2k1_k1_r1` | 1 | 94 / 96 | 16384 | 5 |
| gqa2 s128 | `32x16x32s2k1_k1_r5` | 5 | 86 / 96 | 16384 | 5 |
| mha4 s4 | `32x16x64s2k1_k1_r1` | 1 | 94 / 96 | 16384 | 5 |
| mha4 s128 | `32x16x16s2k1_k2_r5` | 5 | 85 / 87 | 16384 | 5 |

All four are **register bound at 5** and all four run 128-thread CTAs. The
`ctas_per_sm=1` the s4 cells print is their chosen `RESIDENCY_CAP`, not a
hardware limit: the hardware allows 5 there too.

Two register counts are given because the numbers the harness logged come from
the **instrumented** build (`TRACE_PHASE`+`TRACE_KLOOP`+`TRACE_SIMT`, B0) while
the budget below is the production build. They differ by up to 10 registers and
both give 5 CTA/SM — which also says B0's probe cost no occupancy.

### The cost curve

Driver CTA/SM against page bytes appended after the union (`raw/occupancy.tsv`;
gqa2 s128 shown, the others differ only in where registers bind):

| page | 0 | 1024 | 2048 | **3072** | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CTA/SM | 5 | 5 | 5 | **5** | 4 | 4 | 3 | 2 |

| cell | free page at 5 CTA/SM | affordable at the residency it runs |
|---|---:|---:|
| gqa2 s4 | 3072 | 84992 |
| gqa2 s128 | 3072 | 3072 |
| mha4 s4 | 3072 | 84992 |
| mha4 s128 | 3072 | 3072 |

The two s4 cells run at `RESIDENCY_CAP=1`, so a page costs them nothing until
it reaches the opt-in ceiling — 84992 B, i.e. 101376 less the union. The two
s128 cells run at 5 and are the binding constraint: **3072 bytes**.

### Is 3072 enough?

The page must hold the successor's no-in-edge operand. For these GEMMs that is
the B tile, `K x 16 x 2` bytes:

| cell | tile K | B tile bytes | stages that fit in the free page |
|---|---:|---:|---:|
| gqa2 s4 | 64 | 2048 | 1 free, 41 within its own budget |
| gqa2 s128 | 32 | 1024 | 3 |
| mha4 s4 | 64 | 2048 | 1 free, 41 within its own budget |
| mha4 s128 | 16 | 512 | 6 |

So yes, and with room: **every cell fits at least one B stage at zero occupancy
cost**, and the constraint falls the convenient way round — the cells that want
the largest page (s4, 2048 B) are the ones running at residency 1 where shared
memory is nearly free, while the cells pinned at 5 want the smallest pages.

### F-40 needs one correction to be used as a shared-memory budget

F-40's closed form predicts the driver exactly on 52 of 56 rows. The four it
misses are the **same row in every cell**: `page=4096`, i.e. `smem=20480`,
where `102400/20480 = 5` exactly. The driver says 4.

The cause is a per-CTA shared-memory reservation the driver adds on top of the
request. Adding 1024 B per CTA makes the form exact on **56 of 56**:

```
ctas_smem = floor(102400 / (smem + 1024))
```

Bisected cliff on gqa2 s128: `dyn=19456` holds 5, `dyn=19584` drops to 4, and
`(19456 + 1024) * 5 = 102400` exactly.

F-40 was fit on sm_89 configurations that were overwhelmingly register bound
(605 register-bound, 150 smem-bound, 322 tie), and the reservation only shows
up when the shared-memory term is the binding one and lands on an exact
boundary — which is precisely the regime B1 moves these kernels into. Quoting
F-40 unamended here would have authorised a 4096 B page and silently cost 20%
of residency. Recorded as F-223.

### After — the page actually in place

✅ Verified on RTX 4090 / sm_89. Same run of `occupancy.py`; raw table in
`raw/occupancy_arms.tsv`, one row per cell per arm, built from the same
sources and macros the timed binaries use.

The mechanism holds **two** pages so slot and slot+1 can be live at once, so
the appended bytes are twice `TILEMEGA_PREFETCH_PAGE_BYTES`.

| cell | smem before → after | appended | regs before → after | driver CTA/SM before → after | residency it runs at | keeps it |
|---|---:|---:|---|---:|---:|:--:|
| gqa2 s4 | 16384 → 18432 | 2048 | 94 → 96 | 5 → 5 | 1 | ✅ |
| gqa2 s128 | 16384 → 18432 | 2048 | 86 → 96 | 5 → 5 | 5 | ✅ |
| mha4 s4 | 16384 → 18432 | 2048 | 94 → 96 | 5 → 5 | 1 | ✅ |
| mha4 s128 | 16384 → 18432 | 2048 | 85 → 88 | 5 → 5 | 5 | ✅ |
| real s4 | 16384 → 32768 | 16384 | 85 → 88 | 5 → **3** | 2 | ✅ |
| real s128 | 16384 → 32768 | 16384 | 144 → 146 | 3 → 3 | 2 | ✅ |

`prefetch` and `inline` are identical in every column: they differ only in when
the wait is issued, which is why `inline` is the storage-matched control.

Three things this says that the "before" half could not:

1. **The ABI split costs registers, and the cost is small.** Splitting the
   TaskBody into `Prefetch`/`Wait`/`Compute` adds 2 registers on four cells, 3
   on `mha4 s128` and `real s4`, and 10 on `gqa2 s128` — which was the one cell
   sitting at 86 with slack to 96. No cell crosses a residency boundary on
   registers: every `reg_lim` stays where it was.
2. **The real model needs a page the budget did not price.** B1-b's budget was
   derived on the reference cells, where the free page at full residency is
   3072 B. The real model's scale row is 4096 B, so the page is 8192 and the
   two of them cost 16384 — five times the free budget. It is affordable only
   because both real cells run at `RESIDENCY_CAP=2`: `real s4` gives up driver
   CTAs 5 → 3 and still meets its cap. Had the solver picked residency 4 or 5
   for that cell, the page would have been rejected at launch.
3. **Nothing was silently lowered.** `keeps_residency` is 1 on 18 of 18 rows:
   every arm can still field the residency its cell was selected at, so the
   B1-e timings compare configurations that R6 would also have accepted.

### What this does not say

- It is a **prediction of resident CTAs**, from the same API production sizes
  the grid with — not an observed count of concurrently running CTAs. It is the
  operative number because the launch is accepted or rejected on it, but it is
  not a measurement of achieved overlap. That is B1-c.
- It is sm_89 only. `optin_cap`, the per-CTA reservation and the register file
  all differ on sm_120; re-derive before quoting.
- The "after" half of B1-b's report — shared bytes, registers and CTA/SM with
  the page actually in place — needs the mechanism and is not in this file yet.
  The page cost is bounded above here; the register cost of splitting the
  TaskBody ABI into `Prefetch`/`Wait`/`Compute` is not, and registers are what
  binds these kernels today.

## B1-c — the overlap itself, measured between adjacent slots

R7 §5.2 asks for direct evidence that the mechanism works, not an end-to-end
number. B0's phase trace supplies it: `prefetch_wait_cycles` brackets the issue
and the `cp.async.wait_group` in *both* timed arms, so the two arms can be
compared slot by slot. The pipelined arm waits on bytes issued during slot
`s-1`'s body; the inline arm issues and waits for the same bytes in place. All
three arms, 50 rounds each, six cells: **900 processes, 900 PASS**, one dump
per round (`raw/*/phase_head/`), medians per slot across rounds so a single
scheduling hiccup cannot move a slot.

| cell | issuing slots | prefetch wait | inline wait | difference |
|---|---:|---:|---:|---:|
| gqa2_s4   | 16   | 430.8  | 2134.5 | **1682.5** |
| gqa2_s128 | 512  | 483.0  | 2090.5 | **1591.5** |
| mha4_s4   | 32   | 448.5  | 2196.5 | **1665.5** |
| mha4_s128 | 1024 | 488.5  | 2236.5 | **1707.2** |
| real_s4   | 32   | 1179.5 | 3689.5 | **2578.5** |
| real_s128 | 1024 | 299.0  | 3347.0 | **2936.8** |

Cycles, not nanoseconds: `clock64()` is per-SM and the dump carries no worker
column, so the schedule has no common time base and no conversion is attempted.
Every number here is a difference of two reads taken by one slot on one SM,
which is the only comparison that clock supports.

**The mechanism fires.** The wait in front of the copy collapses by 1591-2937
cycles on every issuing slot of every cell, and it does so on *every* issuing
slot: `overlap_slots_positive` equals `issued_slots` in all six rows, 2640 of
2640. On the slots the rule refuses the two arms agree to within 5-9%
(`prefetch_wait_idle` 275-556 vs `inline_wait_idle` 280-590), which is the
instrument's floor with nothing in flight and confirms the difference above is
the copy and not the bracket.

### The fetch is relocated, not removed

The wait bracket cannot see the issue in the pipelined arm, because that issue
happens inside the *previous* slot's body. Charging it there is what separates
a fetch that was removed from one that was merely moved:

| cell | overlap | prev body (prefetch) | prev body (inline) | Δ | net per issue |
|---|---:|---:|---:|---:|---:|
| gqa2_s4   | 1682.5 | 16970.0  | 15383.5  | **+1586.5** | 96.0 |
| gqa2_s128 | 1591.5 | 36401.0  | 35492.0  | **+909.0**  | 682.5 |
| mha4_s4   | 1665.5 | 17066.5  | 15488.0  | **+1578.5** | 87.0 |
| mha4_s128 | 1707.2 | 45774.5  | 46365.5  | -591.0      | 2298.2 |
| real_s4   | 2578.5 | 181066.5 | 181247.5 | -181.0      | 2759.5 |
| real_s128 | 2936.8 | 381926.0 | 380517.0 | **+1409.0** | 1527.8 |

On the two `s4` reference cells the predecessor's body grows by 94-95% of what
the wait lost: 1586.5 of 1682.5, 1578.5 of 1665.5. Almost nothing was hidden --
the copy was carried out of the wait and into the body it was supposed to hide
under, leaving 87-96 cycles. **That is the direct answer to B1-e's open
question**: `prefetch/inline` is 1.0154 and 1.0142 on exactly these two cells,
and the trace shows why the overlap did not pay for itself there.

The three larger cells do not all behave that way. `gqa2_s128` relocates 57% of
the overlap; `real_s128` 48%; on `mha4_s128` and `real_s4` the predecessor's
body is *shorter* in the pipelined arm (-591, -181), so no relocation cost is
visible at all and the full overlap survives. The relocation is real but it is
not universal, and this table says so rather than generalizing the two cells
that show it most.

### Even the most favourable reading is an order of magnitude too small

Take the cells at their best -- net recovery per issue, ignoring relocation
where the delta is negative -- and multiply by the slots that issue:

| cell | net per issue | issuing slots | recovered | all bodies summed | share |
|---|---:|---:|---:|---:|---:|
| gqa2_s4   | 96.0   | 16   | 1 536     | 8 908 861     | 0.017% |
| gqa2_s128 | 682.5  | 512  | 349 440   | 151 397 291   | 0.231% |
| mha4_s4   | 87.0   | 32   | 2 784     | 20 122 179    | 0.014% |
| mha4_s128 | 2298.2 | 1024 | 2 353 357 | 342 350 643   | 0.687% |
| real_s4   | 2759.5 | 32   | 88 304    | 2 292 666 857 | 0.004% |
| real_s128 | 1527.8 | 1024 | 1 564 467 | 2 552 256 443 | 0.061% |

The denominator is aggregate body work across all workers, not a makespan --
none is available from a per-SM clock. With the workers balanced the two are
proportional, so the right column is an order-of-magnitude *ceiling* on what
this overlap could move end to end, not a prediction of it.

**0.004%-0.69%.** B1-e measures the mechanism's always-on price at 2.1-5.0%
on five of six cells (`inline/control`; on `real_s4` it is -0.7%). The ceiling
on what the overlap can return is smaller than that price by one to three
orders of magnitude on every cell, and the two
measurements are independent: one comes from the phase trace, the other from
end-to-end rounds. That is the whole diagnosis, and it does not depend on the
relocation being universal.

### What B1-c does not say

It does not say the copy is slow; 0.8-9.9% of the body it hides under
(`overlap_share_of_prev_body`) is a plausible cost for the bytes involved. It does not say overlapping is impossible on this
hardware -- it fires on 2640 of 2640 eligible slots. What it says is that the
*population* of eligible slots is too thin (B1-a: 0.26-8% of slots) for any
per-slot gain of this size to matter against a cost every slot pays. The number
that would have to change first is the population, not the copy.

## B1-d — pipelining as a dimension of sigma

✅ Verified (unit test, `test/unit/pipeline_sigma_test.cpp`, CTest target
`pipeline_sigma`). Reproduce with `build-portable/pipeline_sigma_test`:

```
PIPELINE_FRONTIER stages=102 with_frontier=68 rope_element_reads=12
PIPELINE_PRICE tasks=232 priced=16 mean_share=0.016181 page512_priced=0
PIPELINE_BOUNDS queue_lb binding_path makespan flags
PIPELINE_TABLE accepted rejected-head rejected-length
PIPELINE_SIGMA PASS frontier pricing bounds table
```

R7 §5.2(b) asks that pipelining be a dimension of sigma rather than a
post-pass, and names three places it has to appear. Each is below with the
example that fails if the wiring is removed.

### Precondition — the frontier is derived, never annotated

`lib/Frontend/Frontend.cpp:227-229` sets `no_producer` from the write set of
the plan, and `lib/Analysis/TaskWork.cpp` splits the read-only frontier out of
each task's read work (`b4b1dac38`). No hand annotation exists anywhere on the
path; `lib/Codegen/Codegen.cpp:436-442` only *emits* what the CG already
carries.

*Verifiable example*: `CheckReferenceModels` walks **all 102 stages** of both
reference models and compares the derived per-operand frontier against EX-E4's
hand rule (a tensor is off the frontier exactly when some stage writes it).
102/102 agree, 68 stages carry at least one frontier operand, and the
element-wise RoPE reads are counted separately (12) because they read a phase
table rather than a producer's output.

### Location 1 — the cost model prices the frontier share

`include/tilemega/Solver/TaskModel.h:36-45` (`PrefetchPricing`) and
`lib/Solver/TaskModel.cpp:58-105`. A task instance is priced twice — once
reading its prefetch operand from global, once reading it locally — and the
difference is the credit. The credit is granted only where the executor would
actually issue: the operand is on the derived frontier, the body declares it
(`ScalarPrefetchOperand`, `include/tilemega/Codegen/tasks/ScalarDataflow.h`),
its footprint is whole 16-byte lines, and it fits the page the binary was
built with.

*Verifiable example*: `CheckPricing` solves `gqa2` at sm_89 and finds
`priced=16` of 232 tasks credited, mean share **1.6%** of the task's own time,
with every credit inside `[0, task_ns]` and zero on every body that declares no
Prefetch. Re-preparing the same problem with `prefetch_page_bytes=512` — a page
the 1 KiB scale row does not fit — gives **0** credited tasks: the solver
refuses exactly what the executor refuses.

### Location 2 — the objective counts the overlap in `max(CP, queue_lb)`

`lib/Solver/ExecutionSimulator.cpp:272-286` builds `pipeline_gain[next] =
min(prefetch_ns[next], task_ns[n])` for every queue adjacency `n -> next` and
routes every duration through one `duration()` lambda, so the simulator's
makespan, critical path, block time and stretch all see the same number.
`:594-635` applies the same gain inside `EvaluatePlanBounds`, in **both**
terms: the queue lower bound and the binding path. The distinction R7 asks for
— "same worker, adjacent, pipelinable" against "same worker, not pipelinable" —
is exactly the `queue_next[n] >= 0` gate: a queue head gets no credit because
there is no earlier body to hide under.

*Verifiable example*: `CheckBounds`, two 10 ns tasks.

| plan | prefetch_ns | queue_lb | binding path | simulated makespan | flags |
|---|---|---:|---:|---:|---|
| both on worker 0 | — | 20 | 20 | 20 | 0,0 |
| both on worker 0 | 4,4 | **16** | **16** | **16** | 0,**1** |
| one per worker | 4,4 | — | 10 | — | 0,0 |
| both on worker 0, `task_ns={1,10}` | 4,4 | **10** | — | — | — |

The second row is the overlap; the third is the same tasks with the same
frontier split across two workers, where both are queue heads and the credit
vanishes — which is what makes this a dimension of sigma rather than a property
of the task. The fourth clamps the credit to the predecessor's body: a 4 ns
fetch cannot hide under a 1 ns task, so the bound falls to 10, not 9.

### Location 3 — the plan carries the flags back to CG

`include/tilemega/Solver/ExecutionSimulator.h:221` (`PipelinedSlots`) reports
which slots the chosen plan pipelines;
`include/tilemega/Dialect/CouplingGraph/PlacementPlan.h:57-63` carries them on
`PlacementTable`; `PlacementSolvePass.h:46-59` writes them into the
`tilemega.placement` attribute (and omits the field entirely when nothing
pipelines, so a model without a frontier keeps its published attribute shape);
`lib/Dialect/CouplingGraph/CGDialect.cpp:382-392` parses them back, and
`lib/Dialect/CouplingGraph/PlacementPlan.cpp:83-92` verifies them.

*Verifiable example*: `CheckPlacementTable` accepts `pipeline={0,1}` on a
two-slot worker, and **rejects** `{1,1}` (slot 0 is a queue head and cannot
pipeline) and `{0}` (the flags must cover every node). The write-back is
therefore not free-form: an inconsistent sigma is refused by the dialect
verifier, not by a convention.

### What B1-d does not say

- The credit is a **bandwidth share**, not a latency model: it is the
  difference between pricing the operand globally and locally under the
  calibrated cost model. It carries no L1-hit term and no queueing term, so it
  is an upper bound on what the overlap can be worth in the cost model's own
  units, not a prediction of the measured gain. B1-c and B1-e measure that.
- Nothing here re-solves R6's cells. B1-e times R6's *selected* configurations
  with and without the mechanism; whether the enlarged sigma would have chosen
  differently is a separate experiment and is not claimed.

## B1-e — end to end, six cells, no threshold

R7 §5.2 asks for the paired ratios and a diagnosis, not a verdict, so none is
set. `control` is FORK6's *selected* configuration verbatim, which makes every
ratio below a ratio against R6's own choice. Each round runs all three arms in
rotation (`run.py e2e`), the ratio is formed inside the round, and the interval
is a 20000-draw bootstrap of the median (seed 20260919). 50 rounds per arm per
cell, 900 fresh processes, 900 PASS.

| cell | control median (ms) | prefetch/control | inline/control | prefetch/inline |
|---|---:|---|---|---|
| gqa2_s4 | 0.116736 | 1.0614 [1.0604, 1.0619] | 1.0442 [1.0435, 1.0463] | 1.0154 [1.0106, 1.0168] |
| gqa2_s128 | 0.244736 | 1.0293 [1.0291, 1.0295] | 1.0286 [1.0264, 1.0291] | 1.0004 [1.0000, 1.0027] |
| mha4_s4 | 0.228336 | 1.0673 [1.0640, 1.0678] | 1.0497 [1.0493, 1.0516] | 1.0142 [1.0129, 1.0169] |
| mha4_s128 | 0.502864 | 1.0326 [1.0310, 1.0334] | 1.0284 [1.0268, 1.0285] | 1.0040 [1.0039, 1.0050] |
| real_s4 | 3.879920 | **0.9918** [0.9913, 0.9923] | 0.9934 [0.9928, 0.9941] | 0.9981 [0.9976, 0.9989] |
| real_s128 | 6.524448 | 1.0225 [1.0215, 1.0239] | 1.0211 [1.0206, 1.0234] | 1.0009 [1.0000, 1.0013] |

Above 1 is slower. Every interval excludes 1 except the two `prefetch/inline`
rows that touch it from above. Read as a whole: **the mechanism costs 2.9-6.7%
on the four reference cells, buys 0.8% on `real_s4`, and costs 2.3% on
`real_s128`.** This is a negative result on five of six cells.

### Where the cost is, and where it is not

The third column is the decisive one. `inline` pays the same two pages, issues
the same `cp.async` and waits on it immediately; it differs from `prefetch`
only in overlapping nothing. On every cell `inline/control` accounts for almost
all of `prefetch/control`:

- gqa2_s128: 1.0286 of 1.0293. mha4_s128: 1.0284 of 1.0326. real_s128: 1.0211
  of 1.0225. The residual `prefetch/inline` is 0.04-0.4%.
- The two `s4` reference cells are the exception in size, not in direction:
  `prefetch/inline` is 1.0154 and 1.0142, so overlapping made them *slower*
  than not overlapping at identical storage.

So the regression is **the mechanism's fixed cost, not a failure of the
overlap**: the ABI split and the page are paid by every slot (B1-b: 2-10 extra
registers on every body, 2048-16384 appended bytes), while only 12-896 slots of
648-49280 are eligible to overlap at all (B1-a). On `real_s4` the same fixed
cost lands on the other side of zero -- both `inline` and `prefetch` are
slightly *faster* than control -- which is the honest reading of a 0.7%
storage-only effect at a 3.88 ms cell: not an overlap gain.

`real_s4` is also the cell whose driver occupancy fell 5 CTAs/SM to 3 for the
page (B1-b) and got faster anyway, which is consistent with both real cells
running at `RESIDENCY_CAP=2` -- the CTAs the page cost were never used.

### What is not yet said here

Why `prefetch/inline` is above 1 on the `s4` reference cells -- whether the
`cp.async.wait_group 1` left outstanding across the slot boundary is itself the
cost -- is B1-c's question, answered above from the phase trace rather than by
inference: on those two cells the predecessor's body grows by 94-95% of what
the wait lost.

Per R7 §0 item 2 and EX-E4: this is a negative result under a mechanism that is
*present and firing* (`declared == issued` on all six cells), on models whose
weights are small enough to sit in L2 on the reference cells, which is why
§5.2's own evidence note makes the real-width cells primary. It is not evidence
that cross-task pipelining is worthless; it is evidence that **this** page-based
frontier prefetch, over the 0.2-6% of slots whose operands are on the frontier,
does not pay for the storage and the ABI split it costs on these six cells.
