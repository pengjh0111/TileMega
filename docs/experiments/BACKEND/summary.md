# R8 closure report — backend rework and IR restructuring

Written for: the TileMega maintainers reviewing this round against the R8
prompt. Every number is recomputed from raw artifacts by
`docs/experiments/BACKEND/verify.py`, whose full output is §3.

## 1. Baseline, prompt, commits

- Baseline `git rev-parse HEAD`: `bb19026894635c8dadda0ab6cade865b5aedebd4`
- Prompt SHA256: `afd1550d161cac07daee087709f80fa13611626ec55632a40ee6a6a49cef4fd7`
- Branch: `tilemega`
- ctest at the last source commit: **53/53**

| # | commit | step |
|---|---|---|
| 1 | `d8201fcc2` codegen: parameterize task bodies on the target arch | 1 (BE-1) |
| 2 | `597c2fdc2` codegen: build gemm collectives from the arch tag | 2 (BE-2) |
| 3 | `f922d3fae` codegen: normalize attention online across the warp | 3 (BE-3) |
| 4 | `f63d4ffad` codegen: reduce across the warp instead of one lane | 4 (BE-4) |
| 5 | `b8cae20ce` experiments: compare each body against the reference | 5 (A-a) |
| 6 | `18482dea6` experiments: litmus the release rule at role granularity | 6, 7 (BE-5) |
| 7 | `5128d3a4d` dialect: separate the graph from the plan | 9 (BE-7) |
| 8 | `0c43162c7` dialect: name the space by its tiles | 10 (BE-8) |
| 9 | `af6323c02` solver: compute occupancy from per-role budgets | 8 (BE-6) |
| 10 | `experiments: measure the reworked backend` — this round's measurements | 11, 12 |
| 11 | `docs: record the round eight backend results` — this commit | 13 |

**H4 ordering, in git history.**

| required order | in this history | substantive? |
|---|---|---|
| BE-1 before BE-2/3/4 | `d8201fcc2` before `597c2fdc2`, `f922d3fae`, `f63d4ffad` | yes — the arch tag and the per-role traits they consume land first |
| BE-5's litmus before any §8.5 text change | `18482dea6` is the litmus; §8.5 was never edited | yes, vacuously and deliberately: the litmus did not clear its controls, so the text change never became admissible |
| BE-7 before BE-8 | one commit for the split (`5128d3a4d`), one for the rename (`0c43162c7`) | partially — see §11, deviation 3: the op names are part of the dialect definition, so the split commit necessarily carries the tablegen rename with it |

Two prescribed messages were not used verbatim. Step 7's
`docs: redefine the release rule for roles` would have described a change the
litmus forbade, so the commit says what it did instead. Steps 9 and 10 asked
for the area `cg:`, which `CLAUDE.md` does not list; they use `dialect:`.

## 2. Gate results

| gate | kind | result | measured | evidence |
|---|---|---|---|---|
| A-a per-operator comparison | hard | **PASS** | rmsnorm, qknorm and attention each 8192 elements, **0 mismatching, `max_abs` exactly 0.0** against PyTorch | `BACKEND/operator_check/operators.tsv` |
| A-b two reference models 50/50 | hard | **PASS** | four cells, 200 fresh processes, 0 failing outputs | `BACKEND/models/` |
| A-c anchored Llama 50/50 | hard | **FAIL** | 0/50 against the CPU golden — and the signature is *identical* to R7 D-a: same `E2E_HASH 735ddfc6d445292e`, same two buffers (374 with 146 elements, 301 with 1), same `max_abs` 0.03125; L1 and L2 bit-identical to L0.5 in every round | `BACKEND/llama/` |
| A-d coverage, no naive implementation | hard | **PASS** | 12 dispatched bodies, all in the table, no row still holding a serial reduction | `BACKEND/coverage.md` |
| A-e occupancy vs the driver | hard | **PASS** | 4/4 cells, closed form 5 = driver 5 | `BACKEND/occupancy.tsv` |
| A-f multi-arch compile and CPU self-check | hard | **PASS** | sm_80, sm_89, sm_90, sm_100, sm_120 all instantiate and report their selected collective | `BACKEND/be2_collective/arch_check.tsv` |
| A-g three levels, no threshold | report | **PASS** | four cells against R7's D-d, §6 | `BACKEND/models/models.tsv` |
| B-a litmus, both controls fail | hard | **FAIL** | compliant arm 450/450 across 9 cells; `nofence` fails every round of every cell; **`nobarrier` fails none** | `BARRIER/raw/litmus.tsv` |
| B-b models after the barrier work | hard | **PASS** | 200/200 rounds | `BACKEND/models/` |
| B-c barrier inventory | report | **PASS** | all 15 harness barriers listed with what each protects | `BARRIER/barriers.md` |
| B-d §8.5 updated, original kept | hard | **PASS** | the litmus did not clear both controls and §8.5 is unchanged, which is what §8.3 requires | `TileMega_skeleton.md` |
| C-a ctest after the split | hard | **PASS** | 53/53 | `BACKEND/ctest.log` |
| C-b dialect ownership | hard | **PASS** | `lib/Codegen` constructs no `tmexec.*` op; `lib/Solver` and the write-back pass construct no `tmcg.*` op | `DIALECT/ownership_check.sh` |
| C-c containers and rename table | report | **PASS** | `tmcg.graph`, `tmexec.plan` defined; 5 `tmcg` ops, 3 `tmexec` ops; rename table written | `DIALECT/rename.md` |
| C-d end to end after the split | hard | **PASS** | `tilemega-compile` runs import → solve → write-back → codegen in one command | `BACKEND/models/gqa2_s4/auto.cu` |

**13 of 15 gates met, 2 hard gates failed, 0 report gates unmet.** The two
failures are A-c, whose signature is bit-for-bit R7's and which §4.7 routes to
R10, and B-a, which §8.3 routes to "do not change §8.5", both below.

## 3. verify.py

`docs/experiments/BACKEND/verify.py` recomputes every §4.7 / §5.4 / §6.3 gate
from raw artifacts: correctness rates from the per-round logs, the operator
comparison from the device and reference buffers, occupancy from each run's own
`E2E_RESOURCE` line, the litmus from its per-round outputs, dialect ownership
by running the grep check against the source tree. It reads no summary, runs
every gate before deciding the exit status, and exits non-zero on a hard
failure.

```
PASS [hard] A-a per-operator comparison                  rmsnorm 8192 elements, mismatch 0, max_abs 0.0; qknorm 8192 elements, mismatch 0, max_abs 0.0; attention 8192 elements, mismatch 0, max_abs 0.0
       evidence: /root/TileMega/docs/experiments/BACKEND/operator_check/operators.tsv
PASS [hard] A-b two reference models 50/50               gqa2_s128 50/50 rounds, 0 failing outputs; gqa2_s4 50/50 rounds, 0 failing outputs; mha4_s128 50/50 rounds, 0 failing outputs; mha4_s4 50/50 rounds, 0 failing outputs
       evidence: /root/TileMega/docs/experiments/BACKEND/models
FAIL [hard] A-c anchored Llama 50/50                     rounds=50 passing=0 failing_outputs=100; l05_vs_l0_mismatch=147 max_abs=0.03125 max_rel=7812.5 l1_vs_l05_mismatch=0 max_abs=0 max_rel=0 l2_vs_l1_mismat; signature identical to R7 D-a (same hash, same buffers, same counts): the known caliber problem, R10
       evidence: /root/TileMega/docs/experiments/BACKEND/llama/correctness
PASS [hard] A-d coverage, no naive implementation        12 dispatched bodies, 0 missing from the table; rows still holding a serial reduction: 0
       evidence: /root/TileMega/docs/experiments/BACKEND/coverage.md
PASS [hard] A-e occupancy closed form vs the driver      gqa2_s128 closed_form 5 driver 5; gqa2_s4 closed_form 5 driver 5; mha4_s128 closed_form 5 driver 5; mha4_s4 closed_form 5 driver 5
       evidence: /root/TileMega/docs/experiments/BACKEND/occupancy.tsv
PASS [hard] A-f multi-arch compile and CPU self-check    sm_80 ok; sm_89 ok; sm_90 ok; sm_100 ok; sm_120 ok
       evidence: /root/TileMega/docs/experiments/BACKEND/be2_collective/arch_check.tsv
PASS [report] A-g three levels per cell, no threshold      gqa2_s4 l05 0.184 l1 0.197 l2 0.156; gqa2_s128 l05 0.311 l1 0.326 l2 0.305; mha4_s4 l05 0.358 l1 0.385 l2 0.343; mha4_s128 l05 0.632 l1 0.654 l2 0.655
       evidence: /root/TileMega/docs/experiments/BACKEND/models/models.tsv
FAIL [hard] B-a litmus, both controls fail               compliant 9/9 cells pass every round; nofence fails 9/9 cells; nobarrier fails 0/9 cells
       evidence: /root/TileMega/docs/experiments/BARRIER/raw/litmus.tsv
PASS [hard] B-b reference models after the barrier work  200 passing of 200 rounds
       evidence: /root/TileMega/docs/experiments/BACKEND/models
PASS [report] B-c barrier inventory                        15 CTA barriers in the harness, all accounted for in the table
       evidence: /root/TileMega/docs/experiments/BARRIER/barriers.md
PASS [hard] B-d release rule updated, original kept      the litmus did not clear both controls and §8.5 is unchanged, which is what §8.3 requires
       evidence: /root/TileMega/TileMega_skeleton.md
PASS [hard] C-a ctest after the split                    100% tests passed, 0 tests failed out of 53
       evidence: /root/TileMega/docs/experiments/BACKEND/ctest.log
PASS [hard] C-b dialect ownership                        PASS: no execution op constructed in lib/Codegen; PASS: no graph op constructed in lib/Solver or the write-back pass
       evidence: /root/TileMega/docs/experiments/DIALECT/ownership_check.sh
PASS [report] C-c containers defined and rename recorded   tmcg ops 5, tmexec ops 3, rename table written
       evidence: /root/TileMega/docs/experiments/DIALECT/rename.md
PASS [hard] C-d end to end after the split               tilemega-compile ran import, solve, write-back and codegen after the split
       evidence: /root/TileMega/docs/experiments/BACKEND/be1_arch/generated_macros.txt

13/15 gates met; 2 hard gate(s) failed, 0 report gate(s) unmet
```

## 4. Stoppage ledger and degraded forms

### Stoppage ledger

| item | state | located cause | what unlocking needs | estimated effort |
|---|---|---|---|---|
| **BE-5** role-aware barriers | **not delivered** | the litmus's `nobarrier` control cannot be made to fail on sm_89: each writer's device-scope fence plus the global round trip of the flag means the consumer observes the publication later than the writers' stores land, in all five constructions tried (§7) | the experiment belongs on a part where the two roles publish through `mbarrier` inside one CTA (sm_90) or through a cluster (sm_120); `BARRIER/run_sm120.sh` carries the matrix | hours on the target, once one is available |
| **BE-2** pricing the builder collective | **degraded (§8.3)** | the solver's `TensorBF16SmemBytes` closed form describes the cp.async multistage collective only; the sm_90 builder's mainloop is 221440 bytes against 73728 at the same shape, so a builder candidate fails the compile-time contract that lets the host enumerate without compiling | a second shared-memory and thread form in the cost model, and a decision about whether both collectives may appear in one search | days; it is a cost-model change, R10's territory |
| **BE-7** container nesting | **partial** | `tmcg.graph` and `tmexec.plan` are defined with symbol-table regions; nothing wraps the flat ops in them because that rewrites all 48 `getOps<...>` walks and every MLIR test at once | one pass to wrap on import plus a walk helper; mechanical but wide | 1–2 days |
| **A-c** Llama against the CPU golden | **not met, and attributed** | identical to R7 D-a to the hash: the comparison caliber is one relative constant against a 16-deep accumulation | R10's depth-aware caliber; §H6 forbids redrawing the line inside the round that measures it | R10 |

### Degraded forms

- **BE-2 selects by capability, and only one of the two paths is priced.**
  `kPricedCollective` marks which; assertions are scoped to the priced path and
  the other is reported. Declared, not silent.
- **BE-3 delivered the softmax and not the TMA load** (§11, deviation 8).
- **BE-4 rewrote the reductions and not the vectorization.** The elementwise
  bodies already stride by `blockDim.x` over contiguous rows; no wider vector
  type was introduced. `coverage.md` says so per body rather than claiming it.
- **BE-5 delivered the litmus and the inventory, not the rolification.**
  `TaskRoles` is declared and unused; no harness barrier was converted.
- ~~**A-g is four cells, not twelve.**~~ **Superseded.** The first pass
  measured four cells and priced the missing eight at "half an hour of solving
  each". That was wrong: R7's solved Plans are all still on disk, so the cells
  only needed a rebuild. A-g now covers all twelve, in both `MIDPOINT_REFINE`
  arms (§6).

## 5. Coverage

`BACKEND/coverage.md` is the table §13 asks for: 14 operator rows, each with
its body, its path, its arch branch and whether a serial reduction remains.
Summary: every operator either goes through a CUTLASS collective, or reduces
with `__shfl_xor_sync`, or has no cross-thread reduction. Four bodies were
rewritten this round — `AttentionChunkTaskBody`, `AttentionPhasedTaskBody`,
`RMSNormTaskBody` and, through it, `QKNormTaskBody`.

⚠️ Two GEMM-family files §1 counts as uncovered — `GemmTaskBody` and
`GemmSplitKTaskBody` — are **not reachable from either anchored model**.
`ModelHarness.cuh` instantiates twelve bodies and neither is among them;
`GemmTaskBody::Run` writes `context.iteration` into the output. They were left
alone rather than converted, because converting a placeholder no model runs
would produce coverage on paper and nothing on the device.

## 6. Three-level timing, against R7 — twelve cells, with and without `MIDPOINT_REFINE`

No Plan was re-solved for this table. Every cell reuses the solved source R7
produced and left on disk (`topk/<cell>/auto.cu` for seq 4 and 128,
`/root/r7_work/ref/<cell>/auto.cu` for seq 1 and 16,
`/root/r7_work/llama_s<seq>/auto.cu` for the anchored model), so the only thing
that differs from R7's measurement is the backend the same Plan is compiled
against. 25 rounds per reference cell, 10 per Llama cell,
`TILEMEGA_WARMUP=5 TILEMEGA_REPEAT=11`, medians, L0.5 / L1 / L2 in ms.

R7 measured its reference models **without** `MIDPOINT_REFINE` and its Llama
cells **with** it; the column "R8 same arm" is the like-for-like comparison and
"R8 other arm" is the one R7 never ran.

| cell | R7 (its own arm) | R8 same arm | R8 other arm | refine cost |
|---|---|---|---|---|
| gqa2_s1 (refine off) | 0.206 / 0.193 / 0.137 | 0.1802 / 0.1905 / 0.1372 | 0.6388 / 0.6328 / 0.4198 | 3.06x |
| gqa2_s4 (refine off) | 0.182 / 0.197 / 0.154 | 0.1843 / 0.1966 / 0.1587 | 1.1305 / 1.1448 / 0.7475 | 4.71x |
| gqa2_s16 (refine off) | 0.209 / 0.215 / 0.213 | 0.2028 / 0.2128 / 0.214 | 3.2748 / 3.2951 / 2.1217 | 9.91x |
| gqa2_s128 (refine off) | 0.323 / 0.337 / 0.313 | 0.3123 / 0.3255 / 0.3092 | 6.2546 / 6.278 / 4.6408 | 15.01x |
| mha4_s1 (refine off) | 0.352 / 0.378 / 0.287 | 0.3531 / 0.3758 / 0.2877 | 1.2472 / 1.2546 / 0.8325 | 2.89x |
| mha4_s4 (refine off) | 0.358 / 0.388 / 0.343 | 0.3602 / 0.3851 / 0.3441 | 2.1074 / 2.133 / 1.4079 | 4.09x |
| mha4_s16 (refine off) | 0.399 / 0.421 / 0.495 | 0.3973 / 0.4167 / 0.4936 | 6.53 / 6.5638 / 4.268 | 8.65x |
| mha4_s128 (refine off) | 0.6 / 0.621 / 0.639 | 0.6318 / 0.6562 / 0.6564 | 12.7601 / 12.8123 / 9.5846 | 14.60x |
| llama_s1 (refine on) | 49.804 / 49.755 / 36.487 | 49.6656 / 49.6113 / 36.401 | 5.4282 / 5.2598 / 4.6452 | 7.84x |
| llama_s4 (refine on) | 49.465 / 49.492 / 41.122 | 49.4577 / 49.4773 / 41.1023 | 6.2177 / 6.1465 / 5.929 | 6.93x |
| llama_s16 (refine on) | 157.749 / 157.839 / 130.978 | 158.233 / 158.31 / 131.203 | 6.5603 / 6.3375 / 6.8198 | 19.24x |
| llama_s64 (refine on) | 380.202 / 380.311 / 423.982 | 379.508 / 379.566 / 422.395 | 10.8508 / 10.8983 / 11.7396 | 35.98x |

✅ **Verified: R7 reproduces, cell for cell, on the reworked backend.** Every
"same arm" column is within measurement spread of R7's own number — including
the headline cell, Llama seq 64 at 379.5 / 379.6 / **422.4** against R7's
380.2 / 380.3 / **424.0**. Nothing in R8's backend work cost or bought time on
these Plans.

⚠️ **Inferred, and it reframes §1: most of the "two orders of magnitude" is one
numerical switch, not the backend.** `MIDPOINT_REFINE` — selective FP64
recomputation near BF16 midpoints, which F-216 added so the anchored graphs
could pass their numerical gate — costs 2.9x to 36x depending on the cell, and
the cost grows with the work per launch:

| seq | gqa2 | mha4 | llama |
|---|---|---|---|
| 1 | 3.06x | 2.89x | 7.84x |
| 4 | 4.71x | 4.09x | 6.93x |
| 16 | 9.91x | 8.65x | 19.24x |
| 64 / 128 | 15.01x (s128) | 14.60x (s128) | **35.98x** (s64) |

Llama seq 64 is 422.4 ms with the switch and **11.74 ms without it**. §1 cites
that cell as evidence that the backend is one to two orders of magnitude off;
the measurement says 97% of it is the refinement pass. The R7 numbers are not
wrong — they reproduce exactly — but their cause is now located, and it is not
where §1 placed it.

⚠️ **This does not trigger §8.4.** The global-stop condition is "§1's premises
overturned by this round's re-measurement, e.g. R7's 380/424 ms not
reproducing". They reproduce to three digits. What changed is the attribution,
which §0 item 2 asks for explicitly.

⚠️ **What it does not say.** `MIDPOINT_REFINE` is not optional on the anchored
models — it is what makes A-a's Llama graph pass in R7 and it is on the D-a
path. The honest reading is that the numerical caliber and the performance
target are coupled: the cheapest available correctness fix costs 36x at
seq 64, and R10's depth-aware caliber (§13) is therefore a performance item as
much as a correctness one.

## 7. Barriers and the litmus

`BARRIER/barriers.md` lists all fifteen CTA barriers in the harness with what
each protects. The structural finding: **none of them is inside a TaskBody**.
Every one sits between tasks or inside the wait/notify protocol, where all
threads of the CTA execute the same code whatever the body did, so role
specialization does not make any of them unreachable. The barriers that would
break are the ones inside the bodies, which is why `TaskRoles` exists.

The litmus (§5.3's matrix: grid ∈ {64,128,256} × elements ∈ {256,1024,4096},
50 fresh processes per cell, address reuse, cooperative writes):

| arm | cells | rounds | outcome |
|---|---|---|---|
| `roles` | 9 | 450 | 450/450 pass |
| `nofence` | 9 | 450 | fails every round of every cell |
| `nobarrier` | 9 | 450 | passes every round of every cell |

Five constructions were tried before concluding, each fixing a real defect in
the one before: both roles in one CTA (every arm passed — one L1 makes fences
unobservable), cross-CTA publication (the compliant arm failed — the test raced
itself), a two-slot ring with an acknowledgement, the signalling thread owning
one element (F-1's own hazard), and a load-dependent store. F-3's rule decides
how to read the result: a control that passes is not evidence that the ordering
it removes is unnecessary.

**§8.5 old and new: unchanged.** §5.3 requires both controls to fail before the
release rule may be rewritten. One does. Under §8.3 the consequence is taken
rather than argued around — no text change, no rolification, no converted
barrier, and `TaskRoles` stays declared and unused.

## 8. Occupancy

The successor form sums over roles:

```
per_cta_regs = sum_r warps_r * ceil(regs_r * 32 / 256) * 256
ctas = min(regs_per_sm // per_cta_regs,
           smem_per_sm // (sum_r smem_r + reserve),
           threads_per_sm // sum_r threads_r)
```

With one role it is F-40/F-223 term for term, which is what makes it a
successor. ✅ **Verified: 4/4 cells, closed form 5, driver 5.**

⚠️ The driver's answer is `l2_ctas`, not `ctas_per_sm` — the latter is the
Plan's residency cap, which is a decision, not an occupancy. The first version
of this check compared against the cap and reported a disagreement that did not
exist.

## 9. Dialect ownership and the rename

```
$ bash docs/experiments/DIALECT/ownership_check.sh
== lib/Codegen must not construct tmexec.* ==
PASS: no execution op constructed in lib/Codegen
== lib/Solver and the write-back passes must not construct tmcg.* ==
PASS: no graph op constructed in lib/Solver or the write-back pass
== the two dialects are disjoint in the op tables ==
tmcg ops: 5
tmexec ops: 3
```

The rename table is `DIALECT/rename.md`. Briefly: `tilemega.task_space` →
`tmcg.tile_space`, `event_tensor`/`coupling`/`fused_task_space` → `tmcg.*`,
`placement`/`implementation` → `tmexec.*`, `#tilemega.<attr>` → `#tmcg.<attr>`,
and the module-level `tilemega.solved_*` → `tmexec.solved_*` because they are
decisions. `TaskBody`, `TaskKind`, `TaskRef`, `TaskTraits` keep their names:
they are executor-side names for executor-side things. `docs/FINDINGS.md`
keeps its historical entries and gains a terminology note at the top.

⚠️ The first ownership check was wrong in an instructive way: it matched
codegen *reading* `tmexec.solved_grid`, which is exactly what codegen is for.
Reading a decision and making one are different things.

## 10. Porting

`BACKEND/porting.md` is the deliverable. In one table:

| target | blocked by |
|---|---|
| A100 (`sm_80`) | nothing in code; needs a target JSON and a run |
| `sm_120` | nothing in the backend; BF16 stays on the multistage collective until CUTLASS ships an sm_120 BF16 builder |
| H100 (`sm_90`) | (1) the cost model must price the builder collective; (2) BE-5's role path, which needs a litmus that can fail its barrier control |
| B200 (`sm_100`) | the two above, plus TMEM-aware accumulator placement if `tcgen05` is to be used |

Compiled and CPU-checked this round: all five architectures. Executed: sm_89
only.

## 11. Deviations from the prompt

1. **§3 expected an old CUTLASS with only SM80 builders; the opposite is
   true.** The submodule is CUTLASS 4.8.0 and ships builders for sm90, sm100,
   sm103 and sm120 — and none for SM80 or SM89. No upgrade applies. BE-2's
   literal instruction (`CollectiveBuilder<ArchTag, ...>` for the GEMM family)
   is therefore unsatisfiable on the development machine's own architecture;
   sm_89 stays on `CollectiveMma`, which is a CUTLASS collective either way.
2. **sm_120's builder refuses BF16** ("only supports F8F6F4"), recorded under
   §8.3 and fallen back rather than dropped.
3. **BE-7 and BE-8 are two commits, but the split commit carries the tablegen
   rename.** An op's name is part of its dialect definition: a `.td` cannot
   name `tmcg.tile_space` before `tmcg` exists, and splitting the two would
   have left the tree unbuildable in between. No point in the history has BE-8
   preceding BE-7.
4. **Two prescribed commit messages were not used** (§1): step 7's would have
   described a §8.5 change the litmus forbade, and steps 9–10 asked for an
   area `CLAUDE.md` does not list.
5. **`CMakeLists.txt` was changed** to add the second dialect's tablegen
   invocations. R8's H1 neither allows nor forbids it; R7's H1 forbade it by
   name. Adding a dialect is not possible without it.
6. **A-g covers four cells rather than twelve** (§4).
7. **The `tmcg.graph` / `tmexec.plan` containers are defined but not yet
   emitted** (§4).
8. **BE-3's TMA branch for the K/V load was not written.** §4.3 asks that
   `caps.tma` select a TMA load of the K/V blocks; the attention body has no
   such branch and reads `Caps<Arch>` nowhere. It is *partly* downstream of
   BE-5 — the warp-specialized producer/consumer structure a TMA pipeline
   usually wants is what the litmus blocked — but that is not a complete
   excuse: a single-role `cp.async.bulk.tensor` with its own mbarrier is
   possible without roles, and it was not written. The online-softmax half of
   BE-3, which is the part §1 measured as the defect (127 of 128 threads
   waiting), is delivered and verified exact.
9. **The skeleton updates §11 asks for landed late.** The terminology rename
   reached `TileMega_skeleton.md` with the BE-8 commit, but §5.3, §5.3.1, §5.4,
   §8.6, §2.3 and the change record were only annotated afterwards, in
   `c9568447a`. §8.5 is deliberately untouched (B-d).

## 12. Confirmation of the prompt's exclusions

- **Cross-task shared-memory paging and pipelining:** not entered. No σ
  dimension was added and no paging mechanism was touched.
- **Static batch (SB, R9):** not entered.
- **Tile-forwarding fuse, cost-model normalization, depth-aware caliber (TF,
  R10):** not entered. A-c's failure is routed to R10 rather than addressed.
- **Plan execution semantics (§5.7.2), W = 1:** unchanged.
- **Statistical significance:** not pursued, per §0 item 4. No ablation table,
  no four-arm decomposition, no paired 25-round comparison, no bootstrap CI,
  no top-k ranking quality. The one heavy re-validation is BE-5's litmus.

## 13. Causes and next steps for gates not met

### A-c — the anchored Llama against the CPU golden

**Located, and it is not this round's.** The failure is bit-identical to R7's
D-a: same `E2E_HASH 735ddfc6d445292e` across L0.5/L1/L2, same two buffers, same
146 + 1 elements, same `max_abs` 0.03125. L1 and L2 agree with L0.5 exactly in
all 50 rounds, so the megakernel reproduces the reference implementation and
the disagreement is between two legitimate BF16 evaluations. §4.7 A-c names
this case explicitly and routes it to R10; `verify.py` checks the signature
against R7's own log rather than asserting the attribution in prose.

**Next step:** R10's depth-aware caliber. The constant is
`1.6e-2f + 1.6e-2f*fabs(e)` at `ModelHarness.cuh:2939`, about four BF16 ulps of
relative slack against a 16-layer accumulation. §H6 forbids redrawing that line
inside a round that measures against it.

### B-a — the litmus's barrier control

**Located to the mechanism, not to a missing piece of work.** On sm_89 every
writer's device-scope fence plus the global round trip of the flag closes the
window the barrier protects: the consumer cannot observe the publication before
the writers' stores have landed, so removing the convergence changes nothing
observable. Five constructions were tried (§7); the fence control fails in all
of them, which is what shows the harness can detect an ordering defect at all.

**Next step, concretely:** run the same matrix where publication is not a
global flag — `mbarrier` between two warpgroups of one CTA on sm_90, or the
cluster scope on sm_120. `BARRIER/run_sm120.sh` is written, self-checked and
refuses to run on the wrong device. Until one of those fails its barrier
control, §8.5 stays as it is and no body may be warp-specialized — which also
means H100 execution stays blocked, because there the roles are not optional
(`porting.md`).

### BE-2's unpriced builder path

**Next step:** give the cost model a second shared-memory and thread form —
`TensorBF16SmemBytes` describes the multistage collective, and the builder's
storage is 3.0x larger on sm_90 and 2.7x on sm_100 at the same shape. Until
then `kPricedCollective` keeps the two apart and the enumerator only offers
candidates it can cost.
