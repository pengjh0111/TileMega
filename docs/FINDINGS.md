# Findings

> **Terminology, from R8 on.** A CG-stage object is a *tile*, not a task:
> `task_space` is now `tile_space` and `TaskSpaceOp` is `TileSpaceOp`. The
> single `tilemega` dialect became two — `tmcg` for the graph's structure and
> `tmexec` for the decisions solved on it — so `tilemega.task_space` reads
> `tmcg.tile_space` and `tilemega.placement` reads `tmexec.placement`. The
> executor-side names (`TaskBody`, `TaskKind`, `TaskRef`, `TaskTraits`) are
> unchanged, because a task is still what one worker runs. Entries below keep
> the spelling they were written with: they record runs that happened, and the
> logs they cite still use it.

## F-1 — CTA-wide publication needs a CTA-wide release sequence

- Finding: when all producer threads write the tile but only thread 0 signals,
  a fence performed only by thread 0 does not order writes performed by the
  other threads. Each writer fences before a CTA barrier; thread 0 signals
  after that barrier.
- Evidence: `docs/experiments/V_A/event_sync.cu`; the compliant path passed
  1,250/1,250 runs and the no-barrier control mismatched 100/100 runs across
  the two fill modes.
- Skeleton impact: §8.5 should distinguish single-thread production from
  cooperative CTA production and show the required CTA convergence.
- Confidence: high.

## F-2 — All-thread polling is not inherently a correctness negative control

- Finding: independent polling by every thread generated excess atomic traffic
  but passed 100/100 correctness runs. In the supplement its kernel-time ratio
  versus single-thread polling was 0.996× / 0.971× / 0.946× at grid 64/128/256,
  so this short-wait workload did not measure the expected contention cost.
- Finding: moving `__syncthreads()` into the divergent polling loop is a
  different and genuinely dangerous pattern: it hung 50/50 runs at each of
  grid 64, 128, and 256. PC samples changed without completion, consistent
  with a collective stall/livelock.
- Evidence: `docs/experiments/V_A/negative_controls.txt`,
  `supplement_polling_timing_raw.txt`, `supplement_barrier_spin_raw.txt`, and
  `supplement_hang_probe.txt`.
- Skeleton impact: §8.1 should separate two rules: single-thread polling avoids
  unnecessary atomic traffic; collective barriers must never occur inside a
  thread-divergent spin loop. The former cost was not measurable here, while
  the latter received direct negative-control evidence.
- Confidence: high for correctness on this kernel/GPU; low for the measured
  performance cost outside this short-wait workload.

## F-3 — Fence-removal passing is not evidence that fences are redundant

- Finding: the original no-fence control used fresh addresses each iteration,
  excluding a stale L1 hit. The combined hostile variant (reuse + no backoff +
  8,192-float tile) still passed 150/150, but isolating address reuse at the
  original tile size failed 150/150. Every sampled mismatch was the exact
  previous-iteration value. Removing backoff alone and increasing tile alone
  each passed 150/150; the fenced reuse control passed 150/150.
- Evidence: `docs/experiments/V_A/supplement_hostile_raw.txt`,
  `supplement_hostile_isolation_raw.txt`, `supplement_reuse_correct_control_raw.txt`,
  and `supplement_mismatch_samples.txt`.
- Skeleton impact: retain §8.5. A fence negative control must reuse addresses;
  fresh per-iteration storage structurally suppresses stale-line failures.
  Hostile changes are not monotonic—larger writes changed timing enough to hide
  the reuse-only failure—so each stressor must also be tested independently.
- Confidence: high.

## F-4 — Occupancy capacity is not the same as a co-residency requirement

- Finding: cubin metadata gave 256 threads, 40 registers/thread, 0 static and
  32 bytes dynamic shared memory, hence 6 CTA/SM × 128 SM = 768 resident CTAs.
  Nevertheless the current circular workload passed 50/50 even at 1,536 CTAs.
- Evidence: `docs/experiments/V_A/supplement_occupancy.txt` and
  `supplement_residency_boundary_raw.txt`.
- Skeleton impact: §8.7 should retain the portable capacity formula, but only
  require `grid ≤ resident_limit` when analysis of the generated wait-for graph
  proves full-grid co-residency is necessary. Local successor dependencies can
  drain progressively above the capacity.
- Confidence: high.

## F-5 — Architecture ordering is not a capability lattice

- Finding: sm_120 compiled cluster/TMA/warp-specialized SM120 MMA paths but
  intentionally excluded `tcgen05`. PTX exposes `tcgen05` for the sm_100f and
  sm_101f families, not sm_120f. The four target shared-memory budgets also
  change one TaskBody from 5 to 8 stages.
- Evidence: `include/tilemega/Target/ArchDispatch.h`, `configs/targets/*.json`,
  `docs/DEVICE_MATRIX.md`, and `experiments/V_I/raw/matrix.tsv`.
- Skeleton impact: §3.4/§5.3 should specify exact capability switches and
  target-configured resources. No `arch >= N` feature policy is sound across
  the supported Blackwell targets.
- Confidence: high for compile-time dispatch and vendor feature tables;
  runtime confirmation on sm_90/sm_120 remains pending.

## F-6 — CUTLASS mainloop extraction has family-specific orchestration

- Finding: direct invocation of `MainloopSm80CpAsync` from a persistent kernel
  is practical and reached 1.119× the measured GemmUniversal throughput with
  exact output. The caller must own tensor/residue construction, accumulator,
  shared storage, K scheduling, synchronization, epilogue, and launch policy.
  The current CollectiveBuilder did not cover this SM80/89 combination, while
  SM90 and narrow-type SM120 builders selected TMA warp-specialized families.
- Evidence: `experiments/V_B/collective_persistent.cu`, `builder_probe.cu`, and
  `raw/performance.txt`.
- Skeleton impact: §5.3's TaskBody ABI should retain an architecture-independent
  context/storage/result contract but permit one mainloop adapter per CUTLASS
  family; it cannot assume a single universal `operator()` orchestration.
- Confidence: high on sm_89, compile-only on sm_90/sm_120.

## F-7 — Compile-time traits are an exact shared-storage pruning oracle here

- Finding: 100 real collective types were evaluated in 17.626 seconds. For
  three emitted kernels, `sizeof(SharedStorage)` matched ptxas shared memory
  exactly (zero-byte error).
- Evidence: `experiments/V_D/raw/traits.tsv`,
  `traits_compile_seconds.txt`, and `ptxas_compare.txt`.
- Skeleton impact: §5/§7 candidate pruning can query traits before codegen, but
  register pressure, spills, and merged-megakernel occupancy still require
  ptxas evidence.
- Confidence: high for the tested SM80 cp.async family.

## F-8 — Shared-memory max requires an explicit union lifetime

- Finding: 2/3/5/8-way and nested dispatch emitted exactly the largest branch
  storage. A loop-carried value remained at max with an explicit union, but
  separate escaping storage objects summed to 36,864 bytes and reduced
  occupancy from 6 to 2 CTA/SM.
- Evidence: `experiments/V_G/raw/resources.jsonl` and `occupancy.txt`.
- Skeleton impact: §8.6 should require generated storage to be a single union
  whose lifetime spans dispatch. The variant bound is
  `f(TargetSpec, blockSize, registers, max_i(storage_i), overhead)`, not a
  target-independent count.
- Confidence: high on sm_89; resource projections for migration targets are
  configuration-level only.

## F-9 — Wait-for topology, not grid size alone, determines progress

- Finding: the backward dependency passed at resident_limit+1 (20/20) but hung
  at 2×resident_limit (20/20). Thus even a non-streaming-looking global graph
  does not make every over-capacity launch hang; the initially scheduled CTA
  subset may still expose a progress frontier.
- Evidence: `experiments/V_J/raw/backward_scan.txt` and
  `hang_probe_summary.txt`.
- Skeleton impact: §2's derived quantities cannot be expressed from local
  `(theta, g)` mapping alone. Co-residency legality needs the global realized
  wait relation, cluster shape, launch order assumptions, and a proof over
  possible resident subsets. §8.7 should use occupancy only as the capacity
  term in that proof.
- Confidence: high for the observed schedule; necessity/sufficiency of a
  general progress test remains open Phase 2 work.

## F-10 — Larger tiles can hide a missing fence

- Finding: under address reuse, no fence, and no backoff, mismatch rate was
  100% through 4,096 floats, 2% at 8,192, and 0% at 16,384 (50 fresh processes
  per cell).
- Evidence: `experiments/V_J/raw/tile_scan.txt` and per-run logs.
- Skeleton impact: §8.5 should say that large GEMM tiles are a weak negative
  control for stale-cache behavior; small norm/RoPE-like TaskBodies must be in
  the validation mix. L1 eviction is a plausible explanation, not a measured
  mechanism.
- Confidence: high for the cliff, medium for the cache interpretation.

## F-11 — Frontend/template work dominates candidate compile cost

- Finding: a variant took 4.658 seconds median from source to cubin, while a
  standalone ptxas run took 0.035 seconds and device link took 0.104 seconds.
  Four-process compilation improved wall time 3.61×. A 170-candidate serial
  query projects to 13.2 minutes.
- Evidence: `experiments/V_E/raw/summary.txt`.
- Skeleton impact: §5/§7 should batch constexpr trait queries and parallelize
  survivor compilation. Compilation caching is still unverified locally.
- Confidence: high for this translation unit and machine.

## F-12 — CuTe IR retains most dynamic algebra, but inverse requires static shape

- Finding: the pinned LLVM/MLIR toolchain built successfully; 236/236 LIT and
  106/106 C++ unit tests passed. Dynamic composition, logical/zipped divide,
  flatten/coalesce, ceil-div, and shape-div are legal and remain explicit when
  not foldable. `right_inverse` rejects a dynamic-shape layout; `left_inverse`
  also rejects dynamic stride. Static results agree with pycute.
- Evidence: `experiments/V_F/raw/test_summary.txt`,
  `dynamic_after_fold.mlir`, `flatten_after_expand.mlir`, inverse diagnostic
  tests, and `pycute.json`.
- Skeleton impact: use CuTe MLIR as the imported analysis representation and
  Presburger/ISL as semantic authority. Specialize the intra-tile `W(g)` before
  dialect inversion; if `W` remains dynamic, invert its relation in the bridge
  or raise Tier. Do not depend on CuTe IR for codegen.
- Confidence: high for the pinned revision and tested algebra; general
  dynamic-stride/swizzle conversion remains outside the affine bridge.

## F-13 — TargetSpec needs launch-topology and provenance fields

- Finding: the present schema is enough for stage sizing and CTA occupancy but
  not for portable cluster capacity or reproducible mainloop selection. Needed
  fields include device/SKU identity and source provenance, max resident blocks
  per SM, cluster occupancy/GPC limits, opt-in shared-memory status, supported
  collective datatype families, and the CUDA/CUTLASS version used to derive
  them.
- Evidence: V-C required cluster-specific occupancy reasoning; V-B selected
  different datatype families on sm_120; H100 SM count differs by SKU.
- Skeleton impact: extend §3.4's target contract before Phase 2 launch solving.
  Keep these as queried/configured fields, not hardware literals in business
  code.
- Confidence: high.

## F-14 — Coupling semantics must remain symbolic in the C++ type system

- Finding: the initial Analysis stub represented `C` as a diagnostic string
  and Definition 4 as concrete `size_t`/`double` fields named
  `fan_in/fan_out/locality/reuse`. This erased invariant I1 before Solver or
  Codegen could observe it and did not match `wait/fanout/volume/count`.
  `ClosedForm` now retains theta/g symbols in an immutable AST, while
  `AffineRelation` represents affine coordinates, quantified producer ranges,
  and multi-producer images. An explicit range-partition operation preserves a
  structural key across split-K reparameterization.
- Evidence: `test/unit/coupling_types_test.cpp` evaluates §2.7 edges 1, 7, and
  11 as wait 1, 4096, and 2. Edge 7 retains the same `StructureKey`, adds only
  parameter `Kc`, and changes wait to 1024 for `Kc=4`; the Release-mode CTest
  passes.
- Skeleton impact: §2 Definition 4 should make all four metrics closed forms,
  and the CG C++/MLIR contracts should distinguish structured semantic
  relations from printable text. Tier (analyzability) and SyncKind (placement)
  must remain separate types.
- Confidence: high for the minimum algebra and I1 machine check; Phase 3 must
  replace the AST evaluator with barvinok-backed piecewise quasi-polynomials
  without changing this public contract.

## F-15 — Export symbols and executable guards are separate frontend inputs

- Finding: a strict two-layer Llama export was stable across three runs and
  propagated symbolic sequence/cache extents through all 160 tensor-producing
  compute nodes. Reusing one `Dim` for cache inputs still produced multiple
  symbols; four executable equality guards carry their required identity.
  Dense KV cache appears as explicit graph inputs/outputs without mutation.
- Evidence: `experiments/V_H/raw/report.json`, `node_shapes.tsv`, and the saved
  `exported_program.pt2`; three boundary-shape executions passed and an unequal
  K/V cache was rejected by a guard.
- Skeleton impact: the Phase 1 frontend must import both range constraints and
  guards, canonicalize equivalent symbols, and treat view/type/metadata nodes
  separately from mathematical TaskBodies. `ClosedForm` represents dimension
  arithmetic, while equality/inequality/modulo guards require a Presburger
  constraint domain. Dense append is Tier 1 only after its interval mapping is
  retained.
- Confidence: high for torch 2.13.0 and this graph; the guard accessor observed
  here is private and needs a versioned adapter.

## F-16 — The L0-to-L1 lowering contract is executable with CUTLASS collectives

- Finding: the V-H `ExportedProgram` yielded 179 task spaces and 222 tensor
  couplings. A 24-stage L0.5 lowering matched PyTorch with maximum absolute
  error `1.5497208e-6`; the single-kernel L1 result was bitwise identical to
  L0.5 and passed 50/50 fresh processes. All 14 projections directly invoke
  the FP32 `MainloopSm80CpAsync<3>` collective family.
- Evidence: `experiments/E2E/e2e.cu`, `raw/first_run.txt`,
  `fresh_process_summary.txt`, and `fresh_process_raw.txt`.
- Skeleton impact: §5's architecture-independent TaskBody/pipeline/event
  contracts are sufficient for this fixed lowering. The frontend still needs
  explicit rules for guard/meta/layout operations; the experiment adapter is
  not a general importer. Full-stage barriers require
  `grid = resident_limit`, unlike streaming wait graphs.
- Confidence: high on sm_89 for the fixed two-layer graph and global-barrier
  event scheme; performance and finer-grained events are not generalized.

## F-17 — Mainloop adapters must encode operand-coordinate conventions

- Finding: the first E2E collective substitution used CUTLASS B's wrong layout
  tag and produced 6,143 mismatches. The collective presents B in logical
  `(N,K)` coordinates; `ColumnMajor` maps PyTorch's contiguous `[N,K]` weight
  storage to logical `(K,1)` strides in this adapter. Passing the 14-invocation
  parameter bundle by value also created a 2,592-byte thread stack frame;
  passing one device pointer reduced it to 32 bytes.
- Evidence: the recorded correction in `experiments/E2E/result.md`, final
  source comments, and `raw/ptxas.txt` (168 registers, zero spills, 32-byte
  stack).
- Skeleton impact: §5.3's per-family adapter contract should include logical
  operand coordinates/strides and require large parameter tables by pointer.
  A C++ layout tag name alone is not a frontend layout proof.
- Confidence: high for the tested SM80 cp.async SIMT adapter.

## F-18 — The CG dialect can enforce the L4-to-L1 semantic contract

- Finding: `ClosedFormAttr` now stores the L3a `ClosedForm` value itself and
  round-trips through MLIR without exposing a builtin string attribute to
  consumers. `CouplingMapAttr` stores structured consumer, producer,
  coordinate, range, parameter, fiber, and image fields. Verifiers reject an
  event extent different from `image(C_kappa)`, a wait value different from
  the relation fiber cardinality after theta/g binding, and Tier 3 + cluster.
- Evidence: `test/Dialect/CouplingGraph/{valid,event_shape_mismatch,
  tier3_cluster}.mlir` pass 3/3 under lit; `cg_attr_roundtrip` preserves and
  evaluates a symbolic ceil-div expression.
- Skeleton impact: §2.6's “one structure” rule is now an actual API boundary:
  importer, future Analysis/Solver, and Codegen exchange only a verified CG
  `ModuleOp`; the old string input to Codegen was removed.
- Confidence: high for the implemented Phase-1 schema and fixed cardinality
  verifier; barvinok-backed general counting remains Phase 3.

## F-19 — Guard identity must be normalized in the C++ importer

- Finding: the thin Python bridge serializes nodes, tensor dependencies,
  symbolic metadata, ranges, and raw guards without classification. The C++
  importer preserves all 179 call-function tasks and 222 tensor couplings,
  retains view/transpose access-map tasks, assigns 24 stages with an explicit
  rule, and imports all four equality guards. Cancellation/union normalizes
  both `s61` and `s65` to `s14`.
- Evidence: `python/tilemega/export_bridge.py`,
  `frontend_import_test`, `experiments/E2E_GEN/raw/bridge_summary.txt`,
  `experiments/E2E_GEN/raw/import_summary.txt`, and
  `experiments/E2E_GEN/raw/cg.mlir`.
  An unsupported fixture reports `aten.imaginary.default` by name.
- Skeleton impact: §4.2/P1.2 must keep a versioned guard adapter and a separate
  constraint domain; P1.4 stage policy must be explicit and testable. Layout
  operations are semantic graph structure, not importer noise.
- Confidence: high on torch 2.13 and the V-H two-layer Llama; stage-rule
  generalization remains a proposal.

## F-20 — Generated Phase-2 CUDA reproduces the handwritten ladder

- Finding: `CouplingGraphToCUDA::Lower(ModuleOp)` traverses task-space,
  coupling, placement, symbolic metrics, and stage data, then emits the
  TaskBody specialization, schedule counts, §8 synchronization, and a
  TargetSpec-driven host launcher. Generated and handwritten L0.5 outputs have
  the same bit hash `5245714bc5d3ab4d`; generated L1 is bitwise equal to its
  L0.5 and passed 50/50 fresh processes with no timeout.
- Evidence: `experiments/E2E_GEN/raw/hash_compare.txt`,
  `fresh_process_summary.txt`, `generated_ptxas.txt`, and retained per-process
  logs. Both variants use 168 registers, 49,536B dynamic shared memory,
  1 CTA/SM, and grid 128 on the probed sm_89 GPU.
- Skeleton impact: P2.3/P2.4 are implemented for the fixed two-layer Llama
  Phase-2 scope. The handwritten `experiments/E2E/e2e.cu` is now only a
  reference; product lowering starts at the real ExportedProgram and passes
  through the verified CG dialect. Fine-grained event generation remains P3.5.
- Confidence: high on RTX 4090 for this fixed model/configuration; performance
  optimization and broader model-stage policies are not claimed.

## F-21 — A generated static schedule has two CUDA address-space consumers

- Finding: the first direct shared-object build failed because L1 referenced a
  host `constexpr` schedule from device code. L0.5 needs that table on the
  host, while L1 needs the same entries in device-visible storage. The final
  emitter derives a host constexpr table and a device constant-memory table
  from one initializer, preventing semantic drift between the two ladders.
- Evidence: `experiments/E2E_GEN/raw/compile_shared.txt`; after the correction,
  `tilemega-compile` produced a 647 KiB ELF `.so`, and the generated executable
  retained bit hash `5245714bc5d3ab4d` for both L0.5 and L1.
- Skeleton impact: §5.4 schedule-table lowering must state the address-space
  duplication explicitly; it is a backend representation detail, not two
  schedules. Schedule entries remain one CG-derived source of truth.
- Confidence: high for nvcc 12.8 and the generated sm_89 Phase-2 path.

## F-22 — Closed-form `W^-1 o R` covers §2.7 exactly, without a Presburger solver

- Finding: every producer's `W` is a tiling, so `W^-1` reduces to per-axis
  `floor(./tile)`; projecting a consumer's read interval through it has
  exactly three outcomes — an exact quotient (possibly quantified over a
  range), an exact `floordiv` when the read is a single element, or an
  explicit relaxation that widens `C` and marks the edge inexact. This closed
  form, not a general Presburger/barvinok solver, is what
  `lib/Analysis/CouplingDerivation.cpp` implements, and it is sufficient: all
  13 rows of §2.7's table are derived automatically and machine-asserted
  (`table27_test`), plus a 14th coupling (`add1 -> add2`) the table omits.
  Three places where the derived form and the table's notation differ are
  recorded, not absorbed by adjusting the derivation to match: (a) the table
  groups by consumer operator while the derivation emits one edge per
  `(consumer, operand)` pair — a presentation difference, since event
  synthesis needs the per-operand granularity; (b) the table's row-3 `C`
  reuses one index for both the token block and the KV cache row, while the
  derivation keeps the real per-row append granularity and projects through
  `floordiv(row, Tm)`, which is the more precise form and agrees exactly when
  the append is tiled at `Tm` rows; (c) the table's row-4 `wait = 1` is the
  decode instantiation (`S = 1`); the derived form is the symbolic `Tkv` a
  prefill pass with `S > Tkv` actually needs.
- Evidence: `docs/experiments/P3/table27.md`; `table27_test` asserts all 13
  rows plus the 14th omitted edge and an I1 split-K reparameterization
  (`PartitionRange` leaves `StructureKey` unchanged and grows `wait` to 1024
  at `Kc=4`) without re-deriving `C`.
- Skeleton impact: §2.7's acceptance criterion ("13 rows derived, each
  derived quantity matches") is met; the deferred general barvinok
  quasi-polynomial authority (§3.5) is still not needed for any access
  pattern this codebase currently generates (rectangular, grouped, or
  structured-ragged reference domains).
- Confidence: high for the covered access patterns; a data-dependent index
  degrades correctly to an operator-level Tier-3 barrier (`derived-gather.md`)
  rather than a fabricated affine relation, but no example in this codebase
  needs a true piecewise quasi-polynomial count.

## F-23 — I2 substitutability needs a machine check, not a visual one

- Finding: a relaxed coupling is only sound if the widened relation `C'`
  actually contains the exact one, `C' ⊇ C`. `Contains(wide, narrow,
  producer)` checks this structurally: a wide position covers a narrow one
  either by being the literal same expression, or by being a quantified
  variable ranging over `[0, full extent of that producer axis)` — exactly
  the shape `DeriveCoupling`'s relaxation fallback emits. The check is
  conservative in the safe direction: `true` means containment was
  *established*, `false` means "not established," never "disproved."
- Evidence: `containment_test` covers both directions (a relaxed `C`
  correctly contains its own exact source; a mismatched producer/extent
  correctly reports "not established" rather than a false positive).
- Skeleton impact: §2.2's invariant I2 is now an executable predicate that
  Codegen or a future Solver can call before accepting a Relax, rather than a
  property asserted only in prose.
- Confidence: high for the relaxation shapes `DeriveCoupling` currently
  produces; a future relaxation strategy that does not follow the
  "quantified over the full axis" shape would need `Contains` extended, not
  bypassed.

## F-24 — Event synthesis is correct; the conservative relaxation costs performance, not correctness

- Finding: `EventSynthesis::Synthesize` turns each derived `CouplingEdge`
  into an `EventRequirement` whose `shape` is `image(C_kappa)` for `kappa=1`
  — the product of consumer-coordinate ranges that actually occur in `C` —
  verified against the CG dialect's own verifier (an event extent that
  disagrees with `image(C_kappa)` is rejected). `CouplingGraphToCUDA::Lower`
  turns every producer-stage-precedes-consumer-stage coupling into a device
  `StageDependency` entry that `ModelHarness.cuh`'s `WaitDependencies` polls
  per-edge instead of `GridBarrier`'s one wait per stage. L2 is bitwise
  identical to L1 on both accepted models (50/50 fresh processes on the
  2-layer GQA model; single-run bitwise match on the 4-layer MHA model). But
  every emitted dependency uses `StageDependency::Map::kAll`, which waits for
  a whole producer stage's grid rather than the exact producer CTAs `C`
  identifies, because proving CTA-level identity (`blockIdx.x` names the same
  tile in two independently compiled TaskBody specializations) needs an
  explicit CTA->task ownership entry in the TaskBody ABI that does not exist
  yet (`Map::kIdentity` is wired into the harness but never emitted). The
  measured consequence: L2 is slower than L1, not faster — median `1.182x`
  (2-layer GQA) and `1.355x` (4-layer MHA) of L1's time, because per-edge
  `kAll` waits replace one barrier with several waits per stage that each
  still cover the whole producer grid.
- Evidence: `event_synthesis_test`; `docs/experiments/E2E_L2/result.md`;
  `docs/experiments/E2E_GEN/raw/` and `docs/experiments/P3_GENERALIZATION/raw/`.
- Skeleton impact: §2.3 and §5.5's event-tensor contract is implemented and
  verified end to end; the `kIdentity` fast path is the next concrete step
  toward a performance win from fine-grained events, and it is an ABI change
  (TaskBody must publish which task index each CTA owns), not an algorithm
  change — `Contains`/`DeriveCoupling` already compute what is needed.
- Confidence: high that the current relaxation is I2-safe (`C' ⊇ C` by
  construction: `kAll` is the extreme case of "the full producer stage");
  high that it is not yet a performance win; the register/occupancy cost of
  adding a CTA ownership table to the ABI is not measured.

## F-25 — Removing the hardcoded Llama structure required moving the decision to the frontend, not the emitter

- Finding: the Phase-2 codegen path had absorbed model structure at three
  separate points: `TaskBodyEmitter::Emit` checked `stage % 12` against six
  hardcoded family flags and `#include`d a 715-line handwritten
  `GeneratedLlamaRuntime.cuh`; `ScheduleTableEmitter::EmitStageCounts` wrote
  `(i % 12)` as a stage-family tag into the generated schedule macro; and
  `lib/Frontend/Frontend.cpp`'s `formStages` threw unless the graph had
  exactly 14 `aten.linear.default` ops arranged in the two-layer pattern.
  Fixing the emitter alone could not have removed this: the structural
  knowledge (how many layers, what width, GQA vs. MHA) has to be *derived*
  somewhere, and the emitter is the wrong layer to derive it in, since it
  only sees a verified CG module, not FX shapes. The fix moved structural
  derivation into a new `lib/Frontend/ModelPlan.cpp`, which matches the
  Llama decoder-layer dataflow shape (RMSNorm -> QKV -> RoPE -> KVAppend ->
  Attention -> O-proj -> residual -> RMSNorm -> gated MLP -> residual) and
  `layers.N.*` parameter naming, and derives layer count, hidden/intermediate
  width, and head/kv_head ratio from parameter *shapes*, not from a count.
  The result — `ModelDims`/`BufferDesc`/`GemmDesc`/`StageDesc`/`OutputDesc`/
  `StageDependency` tables — is attached to the CG module as a
  `tilemega.model_plan` attribute; `CouplingGraphToCUDA::Lower` now only
  reads that attribute and emits table *data*, never a `%`/hardcoded-count
  control-flow constant. `TaskBodyEmitter::Emit` now unconditionally emits
  `#include <tilemega/Codegen/tasks/ModelHarness.cuh>`, a model-independent
  runtime; `GeneratedLlamaRuntime.cuh` no longer exists.
- Evidence: two structurally different models pass end to end through this
  one generator (2-layer GQA, 179 task/222 coupling/24 stage; 4-layer MHA
  with `kv_heads == heads`, 355 task/444 coupling/60 stage/11 guard); a
  regression grep (`docs/experiments/P3_GENERALIZATION/run.sh`) confirms the
  generated `.cu` contains none of `% 12`,
  `TILEMEGA_GENERATED_TASK_COUNT 179`, `TILEMEGA_GENERATED_COUPLING_COUNT
  222`, or `GeneratedLlamaRuntime`; `docs/experiments/E2E_GEN/result.md` and
  `docs/experiments/P3_GENERALIZATION/result.md`.
- Skeleton impact: §5.1/§5.2's "handwritten TaskBody only, everything else
  generated" boundary is now real for the layer loop, stage dispatch, and
  launcher, not just for the six TaskBody kernels; P1.4's explicit two-layer
  stage rule is superseded and marked as historical record only.
- Confidence: high for the Llama decoder-layer family (layer count and
  GQA/MHA ratio both vary correctly); no evidence either way for a
  structurally distinct family (e.g. a pure MLP stack, a different norm) —
  `ModelPlan.cpp`'s pattern match would need a new rule, and that rule has
  not been written or tested. The Analysis-layer derivation itself
  (`CouplingDerivation`, independent of this Frontend path) is already
  demonstrated on `MlpStack`/`MhaModel`/`GatherModel` (F-22,
  `docs/experiments/P3/table27.md`'s "Other models" section), so the gap is
  specifically in the FX-to-ModelPlan pattern matcher, not in the
  coupling/codegen algorithms downstream of it.

## F-26 — The CuTe/ISL layout bridge shipped as a verified classifier, not the planned isl_map round trip

- Finding: P3.1 as scoped called for an `ISLContext` RAII wrapper around the
  isl C API, CuTe layout <-> `isl_map` conversion in both directions, and a
  round-trip unit test. What is implemented instead is
  `CuteLayoutBridge::Project`, which classifies a `LayoutDescriptor` into one
  of four `InverseStrategy` values (`kCuteStaticRightInverse`,
  `kPresburgerRelation`, `kCancelSharedLayout`, `kRaiseTier`) using the
  three-level rule from V-F: static `g` uses CuTe's own `RightInverse`;
  unresolved dynamic extent with constant stride would go through a
  Presburger relation (the *decision* to route there is implemented and
  tested; the actual isl_map construction is not); dynamic stride or swizzle
  explicitly raises the Tier rather than approximating. No isl/barvinok
  dependency is linked into the build. The reason this did not block §2.7's
  acceptance is F-22: the coupling derivation actually exercised by every
  covered model uses closed-form `floor(./tile)` algebra, which never needed
  to reach the Presburger path this bridge was meant to own.
- Evidence: `layout_bridge_test` covers all five branches (static ->
  CuTe-right-inverse, symbolic -> Presburger-relation, shared `layout_id` ->
  cancel-shared-layout, dynamic-stride -> Tier-2 floor, swizzle -> Tier-3
  floor).
- Skeleton impact: §3.5's CuTe->ISL conversion rules and P3.1's isl_map round
  trip remain open; recorded as residual debt in `TileMega_skeleton.md`
  §1.5.1 rather than marked done. The three-level policy itself (which
  strategy applies, and the Tier consequence of each) is implemented and
  tested independently of whether the Presburger backend exists yet.
- Confidence: high that the classifier is correct for the four flag
  combinations tested; no evidence the Presburger path works, because
  nothing in this codebase currently forces it to run.

## F-27 — isl/barvinok coexist with MLIR with zero isolation; the real constraint is a literal-divisor rule, not a link conflict

- Finding: the task's premise ("barvinok/isl 不依赖 LLVM... 大概率正交") is
  confirmed rather than assumed. `docs/experiments/P3_ISL/
  crosslink_probe.cpp` builds an `mlir::ModuleOp`, exercises MLIR's own
  bundled `mlir::presburger::IntegerRelation`, and calls barvinok's
  `isl_set_card`/`isl_set_is_subset` in one process; it links and runs
  cleanly as a CMake target (`isl_crosslink_test`, ctest `isl_crosslink`).
  `nm` on every static library in the pinned MLIR build shows zero `__gmp*`
  symbol references — MLIR does not link GMP at all, so there is no GMP
  version to reconcile. The actually load-bearing discovery came from
  building the *first* real expression the migration needs, not from the
  link test: `isl_aff_div(m, tm)` with `tm` a genuine isl parameter fails at
  the isl C API level (`isl_aff.c:3502: second argument should be a
  constant`), and the `iscc` text parser rejects the same thing through
  every syntax tried (`[m/Tm]`, `m = Tm*q`). isl's affine-expression div/floor
  node stores a *literal* rational denominator; a parametric divisor is not
  representable as a single `isl_aff` regardless of spelling. Rebuilding the
  same expression with the tile size as a literal (128) and only the
  workload dimension (`S`) as an isl parameter works immediately and gives
  the expected closed forms (`card` = `S`; `ceildiv(S,128)` prints as
  `floor((127+S)/128)`).
- Evidence: `docs/experiments/P3_ISL/crosslink_probe.cpp`,
  `docs/experiments/P3_ISL/parametric_div_probe.c`, both with raw output
  quoted in `docs/experiments/P3_ISL/result.md`; `ctest -R isl_crosslink`
  1/1 passed.
- Skeleton impact: this is not a new constraint the migration invents — it is
  the same boundary V-F already established for CuTe's `RightInverse` ("`g`
  固定了 tile 内的 W → 特化后用 CuTe 静态 `RightInverse`"), now shown to hold
  identically for isl, and it matches what `lib/Frontend/Frontend.cpp`
  already does (`tilemega.g` is stored as a literal dictionary at import
  time; only `ClosedForm`'s printed form keeps `g` syntactically unevaluated
  until `Codegen.cpp` calls `.Eval(theta, granularity)`). Part 3's
  `DeriveCoupling`/`ComputeMetrics` rewrite must substitute `g`'s concrete
  values when building `isl_map`/`isl_pw_qpolynomial` objects and keep only
  `theta` as genuine isl parameters (preserving I1 through to the generated
  binary, where `theta` is bound at launch). This design point is settled;
  it is recorded here so the eventual rewrite does not have to re-derive it
  under time pressure.
- Confidence: high — both the coexistence claim and the literal-divisor rule
  are demonstrated by running code, not by reading documentation. What is
  NOT done: the actual `ClosedForm`/`AffineRelation` deletion and isl-backed
  rewrite of `CouplingDerivation`/`AccessRelation`/`DerivedMetrics`
  (Part 3.2–3.5), the CuTe↔isl layout bridge (Part 2), Tier judgment via isl
  (Part 4), and the three Part 5 items (monotonic counter, TaskBody ABI CTA
  ownership + `kIdentity`, L2 performance number) — none of these were
  attempted this round; see `TileMega_skeleton.md` §1.5.1.

## F-28 — The pre-isl fanout was a heuristic, and it was wrong for many-to-one couplings

- Finding: before the migration, `fanout(y) = |C^-1(y)|` was not computed as
  an inverse-image cardinality at all. It was a structural rule: *a consumer
  coordinate occurring in C is pinned by y and contributes a factor of 1; one
  that does not is free and contributes its whole range.* That is correct for
  an identity occurrence (`hh` maps 1:1 to a producer coordinate) and wrong
  for a floordiv occurrence, where many consumer coordinates map to one
  producer coordinate. §2.7 row 3 (`RoPE_k -> KVappend`) is exactly this
  case: KVappend tiles the cache row axis by 1 (one task per row) while
  `rope_k` produces Tm = 128-row blocks, so one producer block feeds 128
  consumer row-tasks. The true fanout is 128; the heuristic reported 1, and
  the skeleton's own table also said 1 (it was written against a coarser
  model where both sides are Tm-blocks and `m ↦ m` is a bijection). Two
  independent "1"s agreeing is why this survived several rounds of review.
  `wait` was unaffected throughout (a row still needs exactly one block).
- Evidence: `table27_test` now asserts 128 for both KV edges and still
  asserts every other row's tabulated `wait`/`fanout` unchanged (rows 1, 2,
  6, 7, 8, 9, 10, 11, 12, 13 all match exactly);
  `docs/experiments/P3_ISL/result.md` has the row-by-row table.
- Skeleton impact: §2.7 row 3's fanout corrected to Tm, and its cluster
  candidacy flipped to ✗ (128 exceeds cluster capacity, the same reason rows
  1 and 10 are ✗). The general lesson is stronger than the one row: a
  derived-quantity formula that is *structural* rather than *counted* will
  agree with a hand-written table exactly where both share the same
  simplifying assumption, which is precisely where neither is checking the
  other.
- Confidence: high. The corrected value is a direct barvinok count over the
  relation the same code derives, and the aligned/misaligned controls in
  `coupling_types_test` pin the boundary behaviour on both sides.

## F-29 — Coarsen was inexpressible before isl, and its algebra is what catches implementation bugs

- Finding: §2.3's `C_kappa = floor(./kappa) o C` could not be written against
  `AffineRelation` at all — that type had no image, preimage, or composition
  operator, only a printable structure. As an isl_map it is one
  `isl_map_apply_range` against a floor map. Measured behaviour: `wait`
  divides by kappa exactly on both producer axes of a 2-D producer
  coordinate (4096 -> 2048 -> 1024 -> 256 for kappa = 1, 2, 4, {4,4}), and
  saturates rather than going below 1 where wait is already minimal.
  Crucially, the *algebraic laws* are what found the bug in the first
  implementation: fresh output names (`q0, q1, ...`) collided with the range
  names of an already-coarsened relation, so a second Coarsen produced
  `q1 = floord(q1, 2)` — a constraint on a single variable whose only
  solution is 0, silently collapsing that coordinate to a point instead of
  halving it. Neither a single-Coarsen value check nor a type-level test
  would have caught it; `floor(floor(./2)/2) == floor(./4)` did.
- Evidence: `docs/experiments/P3_ISL/raw/coarsen.txt`; both laws asserted in
  `coupling_types_test`.
- Skeleton impact: P4.6's `[!] 待验证：ISL 对含参数化整除的映射是否表达式爆炸`
  is answered with measurements rather than left open — across the sweep the
  relation and the quasi-polynomial each stay at **one piece**, and isl text
  length is flat in kappa (leaving S symbolic costs a constant ~15
  characters). kappa therefore need not be restricted to powers of two on
  expression-size grounds. ⚠️ Measured for one decoder layer's edges at
  kappa <= 4, coarsening one axis at a time; deeper nesting untested.
- Confidence: high for the measured range; the "no explosion" claim is
  explicitly scoped to it.

## F-30 — wait and fanout need opposite bounding treatments, and isl says so only by failing

- Finding: the two Definition-4 counts want contradictory things from the
  same relation. `fanout` needs the producer (range) tuple bounded, or
  `isl_map_card`'s piecewise decomposition of the reversed map keeps a tail
  that is reachable only for other parameter values and evaluates to 0
  there — making a uniform fanout of 32 look position-dependent (max 32,
  min 0). But bounding the range *inside C* regresses `wait` instead: a
  relation carrying both a genuine isl parameter and an inequality-range-
  derived producer coordinate bounded on both sides drives barvinok into
  `unexpected missing (bounded) solution` (`basis_reduction_tab.c:210`) and
  an incomplete result. Applying the producer box **only to the reversed
  map**, inside the fanout computation, keeps each direction in the regime
  its own counting problem is tractable in.
- Evidence: both failure modes were observed directly while migrating and
  are reproduced by the reference models; the resolution is at
  `ProducerRangeBoxText` in `lib/Analysis/CouplingDerivation.cpp` with the
  reasoning recorded there.
- Skeleton impact: recorded in §1.5.1 as a worked-around limitation rather
  than a fix. It is a property of this isl/barvinok build; an upstream
  version change should re-test it.
- Confidence: high that the workaround is correct for the covered models
  (every §2.7 row's wait and fanout evaluates, and the aligned/misaligned
  controls behave); low confidence that the underlying barvinok behaviour is
  fully characterised — this is a boundary found empirically, not a root
  cause understood in barvinok's algorithm.

## F-31 — The verified coupling derivation does not yet drive the generated code

- Finding: `lib/Frontend/Frontend.cpp` — the path that turns a real
  `export_bridge.json` into a CG module — does not call `CouplingDerivation`
  at all, and did not before this migration either. It builds one
  `task_space` per ATen `call_function` (179 for the V-H model), one
  `coupling` per tensor use, and fills every coupling's `relation` from a
  placeholder, with `wait`/`fanout`/`volume`/`count` hardcoded to 1 and
  `tier` hardcoded to 0. The derivation that §2.7, the Tier classifier and
  event synthesis all validate runs only over the operator-level
  `OperatorGraph` in `ReferenceModels.cpp`, reached by `tilemega-derive` and
  the unit tests. The generated `.cu` is unaffected because Codegen uses only
  the *structural* fact of which stage pair a coupling connects, never the
  metric values — which is also why the placeholder went unnoticed.
- Evidence: `fixedRelation()` and the coupling-construction loop in
  `lib/Frontend/Frontend.cpp`; `(void)coupling.getWait()...Eval(known)` in
  `lib/Codegen/Codegen.cpp` discards the value it forces.
- Skeleton impact: this is the remaining gap between "L3b is implemented and
  verified" and "L3b drives the product". The two paths also differ in
  granularity — operator-level vs. one node per ATen call — so connecting
  them means having the frontend build `OperatorNode`s at operator
  granularity. `ModelPlan` already recognises exactly those operators
  structurally, so it is the natural attachment point. Out of scope this
  round (the task listed `Frontend.cpp` as untouched), recorded in §1.5.1.
- Confidence: high — established by reading both paths and by the fact that
  the E2E bit hashes are unchanged by a migration that rewrote every metric.

## F-32 — L2's slowness was the dependency-table scan, not the event design

- Finding: the first L2 was 1.182× (2-layer GQA) / 1.355× (4-layer MHA) the
  median of L1's global barrier. Three plausible causes were measured and
  ruled out before the real one was found: per-CTA event fan-out (no
  change), consumers polling `arrivals` instead of a published `epoch` (no
  change — read-write sharing of one line across the grid), and the spin
  backoff (64/256/1024 iterations all gave ~1.187). The decisive measurement
  was disabling the waits entirely: L2 fell to 0.99× L1 and produced 5376
  mismatches, which establishes that the compute paths are identical and the
  entire gap is synchronization. The cause was that `WaitDependencies`
  scanned all 55 `StageDependency` entries out of global memory, per stage,
  per CTA, in thread 0. Emitting the table sorted by consumer with a
  `kDependencyOffsets[stage_count+1]` index, so a consumer reads only its own
  slice, took the ratio from 1.19× to 1.021×.
- Evidence: `docs/experiments/E2E_L2/result.md` "How the 18–36% was
  attributed"; `lib/Codegen/Codegen.cpp` (`byConsumer` + `kDependencyOffsets`);
  `WaitDependencies` in `include/tilemega/Codegen/tasks/ModelHarness.cuh`,
  whose comment records the two rejected designs so they are not retried.
- Skeleton impact: §1.5.1's L2 entry previously attributed the slowdown to
  the `kAll` relaxation. That attribution was wrong: `kAll` costs one epoch
  poll per incoming edge, which is cheap; what was expensive was finding the
  edges. The correction matters because it moves the fix from "extend the
  TaskBody ABI" (a large change) to "index the table" (a small one).
- Confidence: high — each step is a separate measurement with the ratio
  reported, and the fix is bitwise-output-preserving (hashes unchanged,
  50/50 fresh processes).

## F-33 — Transitive reduction of the stage DAG is sound under monotonic epochs

- Finding: the generator emitted one `StageDependency` per distinct
  producer/consumer stage pair — 55 for the 2-layer GQA model — including
  pairs already implied by a longer path. Those are removable without
  weakening the ordering: a stage's `epoch` is published only after every
  *active* CTA of that stage has arrived, and each of those CTAs arrived only
  after clearing its own waits. So `p → q → c` implies that `epoch[q]`
  published ⟹ `epoch[p]` published, and `c`'s direct wait on `p` can never be
  the blocking one. The happens-before edge survives through `q`, so I2 is
  untouched. Measured: 55 → 31 edges (GQA), 63 (MHA), bitwise-identical
  output, worth roughly 0.5% of runtime.
- Evidence: `TransitiveReduction` in `lib/Codegen/Codegen.cpp`;
  `kDependencies` count in `docs/experiments/E2E_GEN/raw/generated_e2e.cu`
  (55 → 31); hashes unchanged in `E2E_GEN/raw/fresh_process_raw.txt`.
- Skeleton impact: this is the first §2.3 event-graph simplification the
  generator performs that is justified by the synchronization semantics
  rather than by the relation algebra. It is also what makes F-34's
  conclusion sharp: the one `kIdentity`-admissible edge is one that
  transitive reduction had already removed.
- Confidence: high for the two measured models; the soundness argument
  depends on "epoch is published only after all active CTAs arrive", which
  is `NotifyStage`'s invariant and would need rechecking if a stage could
  publish early.

## F-34 — `kIdentity` is worth zero waits on both accepted models

- Finding: `kAll` was assumed to be the thing standing between L2 and a
  performance win, with the TaskBody ABI's missing CTA→task ownership map as
  the blocker. With that ABI entry added (`TaskOwnership`, `TaskBase.h`) the
  assumption is now measurable, and it is false. An edge admits `kIdentity`
  only if `C ⊆ identity` *and* the two stages share a CTA→task map. Of the
  decoder layer's 21 derived edges: 10 have mismatched task-space ranks, 7
  satisfy `C ⊆ identity`, and of those 7 exactly **1** has both ends in the
  same ownership kind (`add1→add2`, `kElementChunk` both sides). The other
  six cross from a GEMM (`kTilePerBlock`: CTA `b` owns N-tile `b`) to RoPE or
  elementwise (`kElementChunk`: CTA `b` owns a grid-stride slice of a flat
  element range, a function of `gridDim` rather than of the task space) — an
  identity *task* coupling that is not an identity *CTA* coupling. And that
  single admissible edge is transitively implied by
  `add1→rmsnorm2→wgate/wup→silu→wdown→add2`, so F-33 already removed it.
  4-layer MHA: 20 identity candidates, 4 admissible, all four the same
  `add1→add2` shape, all four already implied.
- Evidence: `docs/experiments/E2E_L2/identity_probe.cpp` and
  `raw/identity.txt` (`SUMMARY … identity_candidates=7 …
  kidentity_admissible=1` / `… 20 … 4`). The containment test is
  `isl_map_is_subset`, an operator that did not exist before the ISL
  migration — this could not have been measured last round.
- Skeleton impact: §1.5.1's `kIdentity` debt item changes character. The ABI
  half is done; the remaining blocker is that the generator never sees a
  derived `C` (F-31), and even when it does, the payoff on these models is
  zero. Anyone reading "add the ABI and L2 gets faster" should read this
  instead.
- Confidence: high for these two models. The ownership classification in the
  probe mirrors `RunStage`'s TaskKind dispatch by operator name, which is
  exact for the reference models but is a proxy, not a link against the
  device code. ⚠️ **Superseded by F-58**: the probe now reads the derived `C`
  and `LiftedOp::ownership` cross-checked against the TaskBodies' own
  `OwnershipOf`, the criterion changed from "same ownership string" to "both
  `kTilePerBlock`", and the admissible counts became 7/42 and 15/86. The old
  numbers are not reproduced.

## F-35 — A per-edge event graph cannot beat a barrier under a sequential stage loop

- Finding: after F-32, F-33 and skipping waits for CTAs that own nothing in a
  stage, L2 sits at 1.0136× (2-layer GQA, 50 fresh processes) and 1.0155×
  (4-layer MHA, 25 fresh processes) of L1 — ⚠️ **those two ratios are
  superseded by F-61**: they are cross-session and the same binary re-measures
  at 1.0367×; the finding's conclusion (L2 is slower, structurally) is
  unchanged and is re-established with a same-session control in F-58 — — a large improvement from 1.182×
  and 1.355×, but still slower, and structurally so. The megakernel's stage
  loop is sequential per CTA: every CTA walks stages `0 … stage_count-1` in
  order. L2 can therefore only *remove waits*; it can never let a CTA execute
  a later stage first. On a stage DAG that is essentially a chain — what a
  transformer decoder layer lowers to — transitive reduction leaves almost
  nothing to remove, and what remains is one epoch poll per incoming edge
  against L1's one arrival counter per stage. L2 pays a slightly larger
  constant for the same ordering.
- Evidence: `docs/experiments/E2E_L2/result.md` performance table and "Why
  the residual 1.4% is structural"; the stage loops in `tilemega_l1_kernel`
  and `tilemega_l2_kernel` (`ModelHarness.cuh`).
- Skeleton impact: the honest reading of §3's L2 goal. Fine-grained events
  pay off when consumers can start out of order; realizing that needs a task
  queue rather than a stage loop, which is a Phase 4 scheduling question, not
  a synchronization-primitive question. Recording this prevents another round
  of tuning the primitive.
- Confidence: high for the claim as measured (two models, fresh-process
  medians, bitwise-identical output). Medium for the generalization: a model
  with genuinely wide, independent branches might show a different sign, and
  none of the accepted models has one.

## F-36 — `tasks ≤ grid` was an unstated precondition of every task body

- Finding: `GemmStageTaskBody` mapped one task to one CTA (`int task =
  blockIdx.x;`). At the hard-coded granularity no stage ever has more tasks
  than the resident grid, so the assumption held silently for the whole
  project so far. The oracle sweep is the first thing that varies the tile
  shape and the split factor, and it breaks it immediately: gqa2 at
  `16x32x16 stages=2 split_k=16` has a gate/up GEMM with 512 tasks against a
  resident grid of 384, and the last 128 tasks were never executed. The
  failure mode is a wrong answer, not a crash.
- Evidence: pre-fix `gqa2_16x32x16s2k16` reports `grid=384 ctas_per_sm=3
  smem=6400` and `RESULT status=MISMATCH l05_vs_l0_mismatch=4090`; the
  post-fix build of the identical config reports the identical resources and
  `PASS`, so the grid-stride loop costs nothing here. 141 of the 143 pre-fix
  RUNFAILs recovered (`docs/experiments/ORACLE/raw_gridbound/screen_gqa2.tsv`
  and `fail_grid.txt` against `raw/screen_gqa2.tsv`); the other 2 are F-37 and
  1 new one is F-38.
- Skeleton impact: §2.4's Split is not a free axis for the task bodies. Any
  body written against the ABI must loop `for (task = blockIdx.x; task <
  count; task += gridDim.x)`, and that requirement belongs in the ABI text
  rather than in the one body that happens to have been exercised.
- Confidence: high. The before/after is the same source at the same flags, and
  the mismatch count is deterministic.

## F-37 — Trait legality is not compilability, and the gap is in the mainloop

- Finding: 8 of the 224 tile shapes that pass CUTLASS's own `constexpr`
  legality *and* fit the 101376 B opt-in smem budget fail nvcc outright. All
  eight are 16×16 tiles (`16x16x{16,32} stages={2,3,4,5}`), 80 compiles across
  both models and all five split factors. The error is in the SM80 cp.async
  mainloop, not the epilogue: `cute/int_tuple.hpp(890): error: no operator "<"
  matches these operands / operand types are: const cute::C<0> < const
  cute::ArithmeticTuple<int, int>`, instantiated through `cute::copy_if` →
  `CollectiveMma<MainloopSm80CpAsync<…>>` → `GemmStageTaskBody::operator()`.
- Evidence: `docs/experiments/ORACLE/raw/tier3_summary.txt`
  (`tier3_compiled_ok 2160 / tier3_compile_failed 80`) and the failing
  `raw/log/*.ptxas`.
- Skeleton impact: this is the empirical case for Part 4's tier 3 existing at
  all. 3.6% of the trait-legal, smem-legal space does not compile, no
  host-side query predicts which 3.6%, and the only way to know is to run the
  compiler. A pruning design that treats tier 1 as a verdict rather than a
  filter is wrong by that margin.
- Confidence: high for this CUTLASS revision and nvcc 12.8. The specific 8
  shapes are a property of the pinned third-party source, not a law.

## F-38 — A persistent spin-wait kernel must be launched at its own resident grid

- Finding: three configurations per model (`128x16x32 stages=2 split_k={4,8,
  16}`) hung rather than failing. `tilemega_l2_kernel` costs 144 registers
  against `tilemega_l1_kernel`'s 128 at that granularity — 1 CTA/SM instead of
  2, a resident grid of 128 instead of 256. The harness derived one grid from
  L1 and launched both kernels at it, so L2's non-resident CTAs never ran, and
  the resident ones spun forever on arrivals only those CTAs could make. The
  general statement: when two persistent kernels share a launch grid, the grid
  must be the *minimum* over their resident bounds, because a spin-wait
  kernel's correctness — not merely its performance — depends on every
  launched CTA being resident.
- Evidence: gdb pinned the hang to `cudaEventSynchronize` inside `LaunchL2`;
  an instrumented build printed `DBG l05_done grid=256 / DBG l1_done / DBG
  l2_launch grid=256` and then nothing. The predicate `occ(l2) < occ(l1) ∧
  max_stage_tasks > resident_l2` is exact over all 1080 configurations on both
  models: `occ(l2) < occ(l1)` holds for exactly the five `128x16x32s2k*`
  configurations (`ORACLE/raw/occupancy_l1_l2_*.tsv`) and exactly the three
  whose largest stage exceeds 128 tasks hang (k=4 → 256, k=8 → 512, k=16 →
  1024; k=1 → 64 and k=2 → 128 pass). After taking the minimum over both
  kernels in `ModelHarness.cuh`, all ten affected configurations PASS at
  `grid=128` with the recorded per-split output hashes
  (`ORACLE/raw/f38_verify.tsv`).
- Skeleton impact: §3's L2 section. The resident-grid rule in §8.2 is stated
  per kernel; the harness contract needs it stated per *launch*. It also puts
  a real cost on L2 that F-35 did not measure: L2's extra registers can cost a
  whole CTA per SM, which halves the grid for L0.5 and L1 as well.
- Confidence: high. The predicate is exact on 2160 candidate/model pairs, and
  the fix is verified on all ten affected configurations against known hashes.

## F-39 — The hard-coded GEMM granularity was 6× off, and the optimum is a plateau

- Finding: sweeping `(tile_m, tile_n, tile_k, stages, split_k)` exhaustively
  over 1077 runnable, bitwise-correct configurations per model shows the
  hard-coded `128x128x16 stages=3 split_k=1` at rank 951/1077 (gqa2) and
  960/1077 (mha4). The best configuration is **6.11×** faster on gqa2
  (1.106944 → 0.181248 ms) and **6.75×** on mha4 (2.192384 → 0.324608 ms),
  median of 25 fresh processes. The entire difference is in the GEMM stages
  (93% of control time → 67% of optimum time; every one of the 14/28 GEMMs
  improves, 4.9×–12.1×); the non-GEMM stages are flat and on mha4 slightly
  worse (−7%). Tile shape alone is 2.94×/3.16×, split-K contributes a further
  2.00×/2.14×. The distribution is sharply peaked: only 10 of 1077 within 5%
  of best, median 2× best, worst 1489×/1572× best.
- Evidence: `docs/experiments/ORACLE/result.md` and `raw/`. 4846.56 s total
  wall for both models (19.61 s tier 1, 1893.85 s for 2240 compiles at 32-way,
  2739.73 s screening, 193.37 s finals).
- Skeleton impact: §6.4's threshold is >10% for "a full cost model + DP is
  worth building", and 611%/675% clears it. Two qualifications the data
  forces: (a) the top ~34 configurations sit inside a ±10% band that a
  25-process median cannot rank — two independent replicates disagree on the
  winner and the run-A winner moves 9.04% — so the model needs to land *in*
  the plateau, not on its argmin; (b) the split factor is 2× on its own and
  present in every top-10% configuration, yet it is invisible to Part 4's
  tier-1 traits query, which prunes tile shape only.
- Confidence: high for the 6× headline (stable control across replicates,
  25-process medians, verified output hashes). Low for any specific tuple
  being *the* optimum — see the plateau above. Untested: whether a per-operator
  `g` beats the best uniform `g`, which is the natural next question and is not
  what this sweep measured.

## F-40 — The occupancy closed form is exact on 1077 real configurations, and both terms bind

- Finding: `ctas_per_sm = min(⌊65536 / (8·⌈regs·32/256⌉·256)⌋, ⌊101376 /
  smem⌋, ⌊1536/256⌋)` reproduces the harness's measured occupancy on
  **1077/1077** configurations for both models, 0 misses. 605 are
  register-bound, 150 smem-bound, 322 tie; the thread term never binds at 256
  threads/CTA. Dropping the smem term alone is wrong for 605 candidates;
  dropping the register term alone is wrong for 150. The register and smem
  footprints are byte-identical between the 2-layer and 4-layer models for
  every shared configuration — the generated code is per-granularity, not
  per-model.
- Evidence: `docs/experiments/ORACLE/occupancy.sh` (pure re-analysis of
  `raw/`, no GPU and no compile) and `raw/occupancy_*_summary.txt`.
- Skeleton impact: Part 4's tier-1 query can answer occupancy from a `g` and a
  `TargetSpec` alone, with no model argument and no compile — which is what
  makes the 2.8 µs tier-1 cost achievable. It also settles §9.2's open
  question about the smem union's effect on occupancy: on this space the smem
  term is what binds for 150 candidates and ties for 322 more, so the union is
  not a bookkeeping detail.
- Confidence: high for sm_89 at 256 threads/CTA. The per-warp allocation
  granularity of 256 registers is read from `TargetSpec::Probe()`, so the form
  should carry to other architectures, but that is inferred, not measured.

## F-41 — The lit suite was pinned to a pre-isl IR syntax, and one of its three cases asserted a check the migration deliberately removed

- Finding: `test/Dialect/CouplingGraph/*.mlir` had not been touched since the
  Part 3 isl/barvinok migration and all three cases failed at *parse* time on
  `unknown attribute 'closed_form' in dialect 'tilemega'` — the metric payload
  is now `#tilemega.metric<"…">` holding isl_pw_qpolynomial text and the
  relation is `#tilemega.coupling_map<"…">` holding an isl_map, not the old
  schema-checked DictionaryAttr. Because the failure was at parse time, all
  three CHECK sets were vacuous: the suite had been asserting nothing since the
  migration, and nothing noticed because `check-cg-lit` was an
  `add_custom_target` and never an `add_test` (ctest reported 15 tests, none of
  them lit).
- The removed check: `event_shape_mismatch.mlir` asserted the diagnostic
  `event tensor has 2 elements but image(C_kappa) has 1`. That check no longer
  exists — `CouplingOp::verify` deliberately stopped re-deriving
  `image(C_kappa)` from the assembled relation, because the only isl query
  available for it (`isl_map_involves_dims`) is syntactic and overcounts
  (lib/Dialect/CouplingGraph/CGDialect.cpp, the comment above the
  `event.getExtent().getValue().Eval(known)` line). Re-checked here: a module
  declaring `tensor<2xi32>` against an image of cardinality 1 verifies
  cleanly today, exit 0. The expectation was **not** relaxed to match; the
  case was retired, and `wait_mismatch.mlir` put in its place to cover the
  check that took over the same job — `wait` recomputed as `card(C)` from the
  relation itself.
- A second, smaller gap found while writing that replacement: the specific
  diagnostic `wait … does not match the relation's fiber cardinality` is only
  reachable when the declared wait and `card(C)` inhabit the *same* isl space.
  A declared `{ 1 }` (0-dimensional) against a 1-dimensional `card(C)` makes
  `SemanticallyEqual` throw, so the error surfaces through the generic
  `cannot evaluate coupling metric after theta/g binding` catch, with isl's
  raw `spaces don't match` on stderr. Still a rejection, so no unsound IR
  gets through, but the message names the wrong cause. `wait_mismatch.mlir`
  therefore uses a same-space wrong value (`{ [i] -> 2 }` against
  `{ [i] -> (1 + i) }`) to pin the intended diagnostic.
- Evidence: `ninja -C build-portable check-cg-lit` — 3/3 fail before, 3/3 pass
  after; `ctest` 15 → 16 tests, all passing, `cg_lit` now among them.
- Skeleton impact: none on the design. It is a measurement-hygiene finding:
  a check target that is not an `add_test` is a check that does not run.

## F-42 — The SIMT f32 mainloop is LSU-issue-bound at every legal tile shape on sm_89, and the SMEM and L2 lanes are collinear across the whole calibration

- Finding: the SIMT f32 TN collective issues `(Tm+Tn)·Tk/2` scalar `ld.shared`
  warp-instructions per CTA per mainloop iteration. Fitting the calibrated
  per-iteration cost on `CTAs_per_SM × that count` alone reproduces all 12
  calibrated (shape, CTAs/SM) points to **3.93% rel rms** with a single
  constant, 0.8577 ns per instruction (≈2.16 SM cycles at 2.52 GHz), while the
  FFMA count varies 32× across those shapes and never enters the fit.
- Consequence, measured: deleting the tensor-core, CUDA-core, SFU, L2 and DRAM
  lanes from the cost model — the `full-lanes(smem only)` ablation rung — moves
  MAPE by 0.02 points and Spearman by 0.0006 on both 1077-point sets. On sm_89
  none of the other five lanes ever binds for this collective. The lane
  machinery is kept because it is what transfers to a tensor-core backend, not
  because it earns anything on this target.
- ⚠️ Unresolvable from these six shapes: the same collective moves
  `4·Tk·(Tm+Tn)` cp.async bytes per CTA per iteration, i.e. exactly
  `8 × LdsInstructions`. The SMEM lane and the L2 lane are **exactly
  collinear** over every shape in the calibration, so the fit cannot attribute
  the mainloop to one rather than the other. It is called the SMEM lane on the
  microarchitectural argument (LSU issue, 256 threads, 16×16 `UniversalFMA`),
  not on the fit. Separating them needs a shape family that breaks the ratio.
- Evidence: `bash docs/experiments/COST_MODEL/run.sh`;
  docs/experiments/COST_MODEL/result.md §3 and the ladder in §4.

## F-43 — §2.2(b)'s pipeline fill depth is not identifiable from a calibration that measures a line in `iters`

- Finding: the Stream-K calibration measures `T = a(o) + c(o)·iters` per
  (shape, CTAs/SM). An envelope `T_pro + max(N−d,0)·T_steady + T_epi` with fill
  depth `d` is the *same line reparameterised*: its intercept is
  `pro + epi − d·c`. Recovering a per-CTA setup constant therefore means adding
  `d·c` back — but `c` scales with occupancy while a per-CTA constant cannot,
  so the recovered "constant" inherits `c`'s occupancy scaling. Measured over
  the 12 calibrated points: `d = 0` gives setup 706 ns at **483 ns** absolute
  rms; `d = stages−1` gives 3562 ns at **1449 ns**; charging the fill at `c(1)`
  instead gives 1596 ns and a wider span still. `32×32×32 s3` alone yields
  setup 2590 / 3955 / 6361 ns at 1 / 2 / 3 CTAs/SM.
- Not a missing regressor: regressing the `d = 0` residual on `stages`, `o`,
  `Tm·Tn` or the LDS count improves 483 ns rms by at most 70 ns and does so
  with nonphysical signs (setup *decreasing* in `stages`). A scalar is the
  right form; the dispersion is genuine.
- End to end the term is a wash on error and a loss on ranking: MAPE
  26.24 → 24.94 (gqa2) and 25.15 → 23.90 (mha4), but Spearman 0.9450 → 0.9382
  and 0.9435 → 0.9361, and the measured optimum falls from rank 19 → 53 and
  12 → 17. The acceptance criterion is ranking, so `pipeline_envelope` defaults
  **off**; the prologue and epilogue terms are always charged, the switch stays,
  and the ablation ladder measures the depth term on every run.
- Skeleton impact: §4.4's cost-model form keeps the envelope's structure but
  must not claim the fill depth is calibrated on sm_89. Recorded as a negative
  result, not removed.

## F-44 — The cost model's error is a near-uniform per-stage shortfall, which is why it ranks far better than it predicts

- Finding: the model **underpredicts 99.4% / 99.1%** of the 2154 measured
  points. Normalised by post-split stage count the shortfall is a median
  **1.58 µs/stage** (gqa2) and **1.47 µs/stage** (mha4), p10 0.85 / 0.74, p90
  5.75 / 5.55. That residual is essentially all of the 25–26% MAPE, and being
  close to configuration-independent it barely perturbs the ordering: Spearman
  0.945 / 0.944 against the tier-2 baseline's 0.461 / 0.446, with the model's
  top-1 and all of its top-3 inside the measured top 3% on both models.
- ❌ The cold-barrier-line hypothesis is falsified. The grid-barrier
  calibration was rewritten to use a fresh 128-byte-aligned `BarrierEvent` per
  iteration — the layout the megakernel actually has — on the theory that the
  original single-hot-line kernel understated the cost. It measures 1045.9 ns
  at 128 CTAs against 1.03 µs for the reused hot line: no difference beyond
  run-to-run spread. The new layout is kept because it is faithful, not because
  it explained anything.
- The residual is recorded as unexplained and no free parameter is allowed to
  absorb it. Both of the model's fitted scalars come from the microbenchmark
  table only; fitting anything against the 2154 measured latencies would make
  the validation set and the model share a number.
- Evidence: docs/experiments/COST_MODEL/result.md §6;
  docs/experiments/CALIB/result.md §(b).

## F-45 — §0.3's "SplitKReduce is 24–26% of the best configuration" is a launch-overhead artifact; split-K matters for a different reason

- Finding: the profiled `GemmCombine` stages average 3.630 µs against a
  3.456 µs per-launch floor — a body of roughly **0.17 µs**. The reduction's
  arithmetic is not a quarter of the runtime.
- Split-K still has to be a first-class model variable, and the ablation shows
  the size of the effect: adding it moves MAPE 181 → 36 and Spearman 0.44 →
  0.91 on gqa2 (185 → 35, 0.43 → 0.90 on mha4). But the mechanism is that
  splitting **doubles the stage count** — gqa2 30 → 44 stages, mha4 60 → 88 —
  and therefore the number of grid barriers, while cutting `iters` per CTA and
  filling the machine. `+splitk` and `+sync` are the two halves of one trade
  and neither layer works without the other.
- Skeleton impact: §2.3 stays, but the justification in §0.3 should cite the
  stage-count/barrier trade, not the reduction's share of the profile.

## F-46 — A block-per-arm A/B protocol makes two identical binaries differ by 0.64% at p = 1.8e-4

- Finding: the first per-operator experiment measured each arm as a block of 25
  fresh processes. On mha4 the DP's split-only and per-operator plans are the
  **same plan** — `diff plan_mha4_split_only.h plan_mha4_per_op.h` differs only
  in a generated comment line — and the two blocks separated by **0.639%** with
  a Mann-Whitney p of **1.76e-4**. The same run had `uniform` and
  `best_uniform`, also the same configuration, land on identical medians.
- Cause: within-session drift, not sampling noise, is the dominant term at this
  effect size, and a block layout aliases drift onto arm identity. §0.2 already
  records the winner drifting 9.04% between replicate sessions.
- Fix: arms are compiled first and then measured **interleaved** — one fresh
  process per arm per round, the starting arm rotated each round — and reported
  **paired** on the within-round ratio, so drift shared by a round cancels.
  Under the paired protocol the same two identical-plan pairs give p = 0.42 and
  p = 0.80 with confidence intervals straddling zero. ✅ verified.
- `summarize_per_operator.py` now canonicalises each arm's compiled plan and
  flags identical-plan pairs automatically, so the null control is reported
  every run rather than noticed once.
- Evidence: docs/experiments/SOLVER/result.md §(b).

## F-47 — Shared memory in a multi-variant megakernel is a union, so a narrower tile on one operator buys that operator no occupancy

- Finding: `GemmStageTaskBody` now compiles up to four tile-shape variants into
  one megakernel, and their `Mainloop::SharedStorage` is spelled as a `union`.
  The gqa2 per-operator plan mixes `16x64x16s2` (10496 B) with `16x32x16s2`,
  and the kernel reports `smem=10496 ctas_per_sm=2` — variant 0's footprint,
  unchanged. ✅ verified (`E2E_RESOURCE` on the `per_op` arm).
- This is §4.3's globality made concrete and it is why per-operator tiling
  cannot buy occupancy: `resident_tiles_per_SM` is decided by the global
  maximum over the shared-memory union and the register maximum, so the only
  thing a narrower operator can win is its own tile-quantisation waste.
- Implementation note: `Mainloop::Params` is a member type of the
  tile-parameterised `CollectiveMma`, so it is a **distinct nominal type per
  variant** even though its four fields are identical, and
  `to_underlying_arguments` has no cross-variant overload.
  `GemmInvocation` therefore carries a variant-independent
  `GemmMainloopOperands` POD built on the host. The epilogue collective, by
  contrast, does not depend on the tile at all and is shared across variants.
  ✅ verified by `static_assert`.

## F-48 — Per-operator `g` is worth 1.2% / 0.8%, and on mha4 all of it is the split factor

- Finding: at 400 interleaved rounds per arm (3200 fresh processes), paired
  within round, the DP's per-operator plan beats its own best uniform plan by
  **−1.214%** [−1.371, −1.100] on gqa2 and **−0.769%** [−0.962, −0.645] on
  mha4, and beats the oracle's measured-best uniform by **−2.820%** on gqa2.
  ✅ verified.
- Decomposition: on gqa2 the per-GEMM split factor is worth −0.512% and the
  per-GEMM *tile shape* a further −1.128% on top of it. On mha4 the DP's
  per-operator plan **is** its split-only plan, so that contrast is a null
  control (+0.000%, p = 0.75) rather than a measured zero.
- The protocol's own floor is two identical-plan pairs at a median of exactly
  0.000% with CI half-widths of ~0.15%, so every number above is outside the
  floor.
- Against §0.3: the per-GEMM improvement range under a uniform optimum is
  4.9×–12.1×, but almost all of it is already collected by the uniform choice,
  because the operators with the most to gain are the ones that dominate the
  total and therefore drive that choice. What per-operator `g` collects is the
  small operators' residue. Recorded as measured; the expectation was not moved.
- Skeleton impact: §4.4's per-operator claim should be stated as ~1% on
  decode-shaped models, not as the per-GEMM spread.
- Evidence: docs/experiments/SOLVER/result.md §(b),
  docs/experiments/SOLVER/raw/per_operator_n400.tsv.

## F-49 — The interface term is separable on both reference models, so the chain DP degenerates

- Finding: `ChainDpStats::interface_spread_ns` is **0 ns** on gqa2 and mha4.
  `Interface(s', s)` does not depend on `s'`, so the O(L·|C|²) transition loop
  and the per-layer-minimum shortcut reach the same objective to 0 ns, at
  939 ms vs 16 ms (gqa2) and 2105 ms vs 53 ms (mha4). ✅ verified.
- Cause: the pair-dependent part is the L2 carry, and both models' live
  footprints sit below the calibrated L2 knee, so the miss probability — and
  the whole term — is zero. This is a property of these two models, not of the
  formulation.
- The general loop stays the default and the shortcut is checked against it,
  because a larger working set would make the term non-zero and the shortcut
  wrong. `interface_spread_ns` is the switch that would catch it.
- Also verified: `decomposition_error_ns` is 1.46e-11 ns / 5.82e-11 ns, so
  prefix + Σ Cost_i + Σ Interface + suffix reproduces the whole-model
  evaluation — no barrier double-charged, no stage charged to nobody.
- Evidence: docs/experiments/SOLVER/result.md §4.3, raw/summary.txt.

## F-50 — Tier-2 alignment propagation is derived correctly and prunes nothing, because tier-1 tiles are powers of two

- Finding: alignment propagation derives each GEMM's reader granularity from
  the generated stage table — RoPE and KVAppend expose `head_dim` in `width`,
  the elementwise tail its row length in `extent` — so §3.2's "QKV columns
  align to `d`" appears as a derived `Tr = 128` rather than as a hard-coded
  rule. ✅ verified (`raw/alignment_gqa2.txt`).
- It then prunes **0 of 1077** candidates per operator on both models, with 0
  operators unconstrained. Every tier-1 `tile_n ∈ {16,32,64,128,256}` and
  `tile_k ∈ {8,16,32}` is a power of two, every granularity in these models is
  a power of two, and a power-of-two tile never straddles a power-of-two reader
  boundary.
- Counterfactual, same derivation on a `tile_n` axis stepped by 16: 48 → 21
  candidates on the QKV operators, joint space 10^23.54 → 10^21.38 on gqa2
  (≈143×) and 10^47.07 → 10^42.77 on mha4 (≈2.0e4×). ✅ verified. The mechanism
  binds; the axis it is given does not need it.
- 1077 = 216 tile shapes × 5 split factors, and the split factor is not an
  alignment-constrained axis, so even a maximal tier 2 could only touch the 216.
- The DP solves the uniform problem with and without the mask and prints
  whether the answer moved; it does not. ✅ verified.
- Evidence: docs/experiments/SOLVER/alignment.md.

## F-51 — `CurrentArch` exists only in the device pass, so a capability assertion in a shared header breaks the host compile

- Finding: `arch::CurrentArch` is defined behind `#if defined(__CUDA_ARCH__)`.
  nvcc still parses every `__device__` body in the **host** pass, so any
  namespace-scope `static_assert` naming `Caps<CurrentArch>` in a header
  included by a full (host+device) TU fails with `namespace "tilemega::arch"
  has no member "CurrentArch"`. ✅ verified with a minimal `-arch=sm_89 -c`
  reproducer.
- Why it had never been hit: the `docs/experiments/V_*` probes that name
  `CurrentArch` are all compiled `-ptx`, which runs the device pass only.
- Fix, inside `ArchDispatch.h` — the one file the policy check lets name
  `__CUDA_ARCH__`: the host pass gets `using CurrentArch = void;` (the primary
  `Caps` template, every capability off) plus `inline constexpr bool
  kDevicePass`. A capability assertion is then written `!arch::kDevicePass ||
  <assertion>`, which is vacuous in the host pass and real in the device pass.
- Evidence: include/tilemega/Target/ArchDispatch.h;
  include/tilemega/Codegen/tasks/ModelHarness.cuh;
  docs/experiments/CLUSTER/result.md §7.5.

## F-52 — ptxas spells the cluster barrier `UCGABAR_ARV` / `UCGABAR_WAIT`, not `BAR.CLUSTER`

- Finding: a SASS self-check that greps for `BAR.CLUSTER` or `CLUSTERBAR`
  matches **nothing** even in a correct sm_90 cluster kernel. The real
  mnemonics are `UCGABAR_ARV` and `UCGABAR_WAIT` (CGA = cooperative grid
  array). Found by diffing sorted opcode histograms of a dim-1 against a dim-2
  object, not by reading documentation.
- Cross-compiled evidence, same source, both sm_90 and sm_120: `UCGABAR` count
  0 / 8 / 8 at cluster dim 1 / 2 / 8; PTX `barrier.cluster` count 0 / 4 / 4.
  ✅ verified.
- Why it matters beyond spelling: `run_on_cluster_gpu.sh` hard-fails when a dim>1 arm
  shows no cluster barrier. With the plausible-looking pattern that check would
  have exited non-zero on a machine where everything was correct — a self-check
  that fails closed on its own typo is worse than no self-check.
- Evidence: docs/experiments/CLUSTER/run_on_cluster_gpu.sh;
  docs/experiments/CLUSTER/result.md §7.5.

## F-53 — Cluster capture on inter-operator dataflow is 0.14–0.19, because the stage-serial megakernel cannot hold a tile across a stage boundary

- Finding: with §4.3's `w(A,B) = Volume × Frequency` evaluated from the
  derivation (gqa2 36 nodes / 44 edges, mha4 64 / 72, 0 unevaluable) and
  size-constrained heavy-edge agglomeration, the fraction of coupling traffic
  a cluster can keep on-GPC is **0.136 (gqa2) / 0.187 (mha4)** at temporal
  reach 1, rising to 0.985 / 0.971 at reach 4. ✅ verified (analytic).
- Reach 1 is the only reach the current TaskBody ABI can implement: shared
  memory is reused by the next stage, so a value produced in stage *i* is in
  global memory before stage *i+1* runs. The 0.97–0.99 rows require four to
  five operators' outputs resident across stage boundaries, which no code here
  does.
- This is the skeleton's §4.5 claim ("簇的粒度匹配「算子内跨 CTA 归约」，
  不匹配「算子间数据流」") as a number rather than an assertion — and it is a
  *confirmation*, arrived at from the opposite direction.
- What binds is reach, not size: at reach 1 the size cap is never the reported
  limiter on either model. Capture is an upper bound on traffic, not a speedup.
- Evidence: docs/experiments/CLUSTER/result.md §7.7,
  raw/labeling.txt; lib/Solver/ClusterLabeling.cpp.

## F-54 — Calibration falsified the skeleton's own independent-duration assumption

- Finding: §P4.4 wrote its own gate — *"干扰偏差 > 30% 则独立时长假设失效"*.
  Measured: a DRAM-bound neighbour at one CTA per SM slows the victim GEMM from
  18.3 µs to 27.6 µs, `interference_ratio = 1.518`, a **51.8%** deviation.
  ✅ verified (n = 41, rsd 0.82%, run-to-run 0.35%).
- So `T_steady` cannot be a sum of per-operator durations measured in
  isolation. The response was not the prescribed degradation to "coarse rank +
  measure the top 3"; it was to make the steady state a `max` over resource
  lanes and to make acceptance a *ranking* criterion, which does not depend on
  absolute predicted durations at all.
- The measurement itself needed two fixes before it meant anything: the victim
  GEMM at N=512,K=512 was 100% launch overhead (enlarged to N=1536,K=2048), and
  the SM clock falls back toward 210 MHz between short host-bound kernels, so
  the two halves of a pair landed at different points on the ramp (a second 1 s
  warm-up immediately before the stage). Before those, the ratio swung 0.90–2.50.
- Evidence: docs/experiments/CALIB/result.md §(d), §"Detours".

## F-55 — Event coarsening κ is monotonically harmful, so it is not a DP state variable

> ❌ Superseded by F-79/F-86. This curve belongs to the retired stage-loop
> executor and must not be used for the task queue.

- Finding: §P4.6 predicted the ablation would show a flat curve. It shows a
  steep one, pointing the wrong way. Against the per-stage event scheme,
  `l2_ms` at κ = 1 is **+237.693%** (gqa2) / **+276.413%** (mha4) and falls
  monotonically to +7.865% / +8.480% at κ = 256 — no κ reaches parity.
  ✅ verified: 60 interleaved rounds, one fresh process per arm per round,
  paired within round, all p = 1.7e−11; every κ arm PASS 60/60.
- The experiment carries its own noise floor: κ does not touch L1, so nine
  `l1_ms` arms are null controls. On gqa2 they span −0.115%..+0.192%, widest CI
  [−0.460,+0.572]. On mha4 all nine are *positive*, +0.069%..+0.305%, five with
  p < 0.05 — reported as measured; the likely cause is that every κ > 0 build
  allocates a grid-deep event array where `stage` allocates a stage-deep one,
  an offset shared by all nine and independent of κ.
  ❌ **That explanation is withdrawn** (F-61): on the 2026-09-04 re-run the
  same nine mha4 L1 controls are no longer all positive. The noise-floor bound
  survives; the mechanism does not.
- Both models put the knee at κ ≈ 32, exactly where the analytic probe puts it
  (`waits` flattens at 812 / 1596 while `overwait` keeps doubling). The two
  halves agree on the shape and disagree on nothing.
- κ therefore stays out of `ChainDpOptions` — and the reason is stronger than
  "the curve is flat": the argmin over κ is the same κ for every configuration,
  namely the curve's limit, which is what the generator already emits. A
  quantity whose optimum never depends on another decision is not a state
  variable.
- ⚠️ This measures κ's cost and not its benefit, by construction: the generated
  dependency table carries no coupling relation, so a consumer waits on every
  group of each producer. **Both sides have since been measured** — see F-59,
  which supersedes this finding's *reason* while leaving its conclusion (κ is
  not a DP state variable) standing. The benefit is bounded from the other side by the
  `nosync` arm — deleting L1's grid barrier (wrong output, 60/60 MISMATCH)
  saves **35.631% / 36.665%**. Synchronization is ~36% of L1 and ~42% of L2, so
  the headroom is real; but L2's event scheme at its cheapest granularity
  already costs +9.8% / +9.5% against the barrier it replaces, and κ only adds.
- Evidence: docs/experiments/COARSEN/result.md, raw/kappa_arms.tsv,
  raw/kappa_summary.txt.

## F-56 — The placement objective is blind, and the permutation it recommends is the worst arm

- Finding: §4.3's temporal-locality objective `|R(c₁) ∩ R(c₂)|` cannot rank CTA
  placements on the accepted fixture — with `seq = 4` and `tile_m = 16` every
  GEMM has one M tile, so `w` is constant and all five arms score identically.
  The hardware spans 24 percentage points on L1 anyway (`pair` +17.0 to
  `reverse` −7.1). ✅ verified: 60 interleaved rounds × 5 arms × 2 models,
  all 60/60 PASS, paired within round.
- `pair`, the only permutation the objective argues for, is the **worst** arm:
  L1 **+17.017% / +18.172%**, p = 1.7e−11. The cause is derivable from the map,
  not from the timing: with `grid=256, ctas_per_sm=2, num_sms=128` it puts
  logical index `L` on SM `⌊L/2⌋`, so a stage with `A` active tasks occupies
  `⌈A/2⌉` SMs instead of `min(A,128)` — half the machine on every stage that
  does not fill the grid. Making co-resident CTAs take consecutive task indices
  *is* packing the low indices onto few SMs; occupancy is not a term in the
  objective, so the objective cannot see the cost of its own recommendation.
- The locality component is worth ≤ ~1%: `scatter` destroys index adjacency
  entirely and costs +1.204% / −0.751% on L1 — one model slightly worse, one
  slightly better. Below §P4.8's own 2% simplify threshold, so list scheduling
  on this objective was not built.
- ⚠️ `reverse` (`g−1−b`) is **7.053% / 6.939% faster** on L1, p = 1.7e−11, and
  the mechanism is ❌ not established. Ruled out: locality (it preserves
  adjacency exactly, and scatter shows adjacency is worth ~1%), occupancy (its
  active set gives the same per-SM CTA multiset as the identity, on the other
  resident slot), and barrier structure (`GridBarrier` has no master CTA). The
  effect is 5–6× larger on L1 than on L05, pointing at repeated residency
  rather than launch scheduling. Not shipped: a permutation whose mechanism is
  unknown can invert on the next fixture or architecture.
- Evidence: docs/experiments/PLACE/result.md, raw/place_summary.txt,
  raw/affinity.txt; include/tilemega/Codegen/tasks/Placement.cuh.

## F-57 — The derived coupling relation now reaches the generator, and it is not the identity

- Finding: `lib/Frontend/Frontend.cpp` used to write a constant
  `{ [0] -> [0] }` into every `CouplingMapAttr` (`fixedRelation()`), because
  `CouplingDerivation`'s input is an `OperatorGraph` and the frontend emitted a
  per-`call_function` stage list instead. It now lifts `ModelPlan` to L-sem,
  instantiates at the launch granularity, and calls
  `CouplingDerivation{}.Derive(graph, known)`; `relation`, `wait`, `fanout`,
  `volume`, `count` and `tier` all come from the derivation.
- ✅ verified, 2-layer GQA at launch granularity: 34 task spaces, 42 couplings,
  **0** placeholder relations; codegen joins them into 38 stage pairs =
  **20 `kAll` / 3 `kIdentity` / 15 `kWindow`**, where before every pair was
  `kAll`. Five of §2.7's six map shapes are carried exactly by the affine
  interval encoding `[(task/div)*scale + offset, ... + count)`.
- The check is non-circular: `wiring_coupling_test` derives the same graph
  twice — once through the FX import, once from `analysis::LlamaStackSem` at
  the fixture's own dimensions — and compares 440 cells. ✅ **4 differences,
  all one naming fact** (`i` vs `m` for the second RMSNorm's parallel dim); the
  test asserts the difference list *equals* that set, so a real divergence
  cannot hide inside it.
- The wiring corrected §2.7 rather than being corrected by it: rows 4 and 5's
  fanout cells (`S`, `⌈L_s/Tkv⌉`) disagree with the table's **own `C` column**
  under a hand count at `S=7, Tm=3, Tkv=2, G=4` — 28 and 18, i.e. `G·S` and
  `Tm·⌈L_s/Tkv⌉`. Both values are kept visible; the expectation was not moved
  to fit the implementation.
- Evidence: docs/experiments/WIRING/result.md, raw/wiring-gqa2-launch.md,
  test/unit/wiring_coupling_test.cpp.

## F-58 — An exact wait set narrows almost nothing, because 89.9% of the poll mass is on element-chunk edges

> ❌ Superseded by F-79/F-86. The dependency analysis was real, but the
> measured executor waited at stage granularity and could not redeem it.

- Finding: with the derived table in place, L2 is still slower than L1 —
  **1.036× / 1.038×** (2-layer GQA / 4-layer MHA), ✅ 25 fresh processes per
  model, round-level pairing, bootstrap CI [1.035951,1.036719] and
  [1.036984,1.037936], Wilcoxon p = 1.30e−05, **0/25 rounds L2 faster**.
- ✅ It is not the dependency table. Four builds differing *only* in the
  contents of `kDependencies` — derived windows (38 edges), everything forced
  to `kAll` (38), `kAll` + full transitive reduction (35), and the pre-wiring
  binary (35) — have overlapping CIs on both models. At the default build
  `TILEMEGA_EVENT_KAPPA` is 0, one event per producer *stage*, and
  `WaitDependencies` then polls exactly one event per edge regardless of `map`:
  a window cannot narrow a wait set that is already one event wide.
- With κ > 0 the narrowing is real and small: total polls over all 38 edges are
  **1.00x–1.10x** fewer than the same edges as `kAll`, across seq ∈ {1,4,128,512}
  and κ ∈ {1,8,32}. The reason is where the mass sits, not how loose the
  windows are. At seq = 512, κ = 1: edges with a `kElementChunk` end carry
  **89.9%** of the polls (chunk→tile 67.6%, chunk→chunk 22.3%, tile→chunk
  4.2%), the 15 `kWindow` edges carry **5.9%**, the 3 `kIdentity` edges 0.03%.
  **5.9% is the ceiling on what an exact window can remove here even if it
  removed all of it.**
- Within that 5.9% the windows currently remove nothing: every fitted window on
  a tile-row edge has `count = scale = Tm = 128`, which on this GPU equals
  `gridDim.x`, so under grid-stride placement a contiguous task interval one
  stride wide covers every CTA and maps back to the whole launch axis. ❌ A
  blocked placement would preserve it — the same counting model gives 1.047× at
  κ = 1 and 1.082× at κ = 32 — and is still bounded by the 5.9%.
- The actionable statement is therefore not "the derivation is too weak" but:
  this model's wait set is dominated by element-chunk placement, a §2.3
  **Place** decision, and no amount of exactness in `C` moves it.
- This refines F-34 (`kIdentity` saves zero waits). With the true `C` and the
  true `TaskOwnership`, `kIdentity` is admissible on **7/42** gqa2 edges and
  **15/86** mha4 edges (both ends `kTilePerBlock`), 3 of which survive into the
  emitted table — and they carry 0.03% of the polls. Not zero; not useful.
  ✅ 0 disagreements between `LiftedOp::ownership` and the TaskBodies' own
  `OwnershipOf`, which replaces the old name-prefix proxy table.
- Evidence: docs/experiments/E2E_L2/result.md, waitset.md,
  raw/waitset_profile.txt, raw/part3/paired_default_kappa.txt.

## F-59 — κ is still not a DP state variable, and the old reason is no longer the reason

> ❌ Superseded by F-79/F-86. The implementation still used a stage outer
> loop, so its benefit/cost balance does not describe the task queue.

- Finding: F-55 rejected κ while measuring only its cost. Both sides are now
  measured against the wired-in table, and the conclusion survives with a
  different argument.
- **Benefit** (device-side poll counting, the fixture's own seq): κ = 1 saves
  **1.0244×** (gqa2) / **1.0222×** (mha4) of polls, κ = 2 saves 1.0161× /
  1.0147×, and κ ≥ 4 saves **exactly 1.0000×**. The benefit is bounded at 2.4%
  and identically zero from κ = 4, because every fitted window on a tile-row
  edge has `count = scale = Tm = 128 = gridDim.x` — a §2.3 **Place** property,
  not a weakness of `C` (F-58).
- **Cost**: ✅ 24 arms, 60 interleaved rounds, paired within round. The window
  arithmetic is **0.48–0.57 points slower** than its own `kAll` control at every
  κ on both models, with disjoint CIs at every κ. The κ sweep itself is
  unchanged in direction: k1 +242.478% / +280.789% against the per-stage scheme,
  falling monotonically to +9.4% / (mha4 comparable) at κ = 256, parity never
  reached.
- So κ stays out of `ChainDpOptions`, and the recorded reason is now: the thing
  a finer event granularity could win is capped at 2.4% and vanishes at κ = 4,
  while the arithmetic that exploits it costs more than the cap. The old reason
  ("the argmin is the same κ for every configuration") is not reused.
- Evidence: docs/experiments/COARSEN/result.md §7, raw/waittable/.

## F-60 — A shape-dependent dependency table has a compile-time granularity precondition, and violating it under-waits silently

- Finding: `div`, `scale`, `offset` and `count` are integers over a *task
  decomposition*, and the GEMM's decomposition is a compile-time knob
  (`TILEMEGA_GEMM_TILE_M`, …, `TILEMEGA_GEMM_SPLIT_K`) that the generator does
  not control. Rebuilding the same generated `.cu` with
  `-include plan_gqa2_uniform.h` (`tile_m 16`, `split_k 16`) makes every fitted
  constant name the wrong producer tasks.
- ✅ verified, and the failure mode is wrong output rather than slowness:
  **0 / 50** fresh processes pass on that build, `l2_vs_l1_mismatch=4096` in
  every one (and `l2_iter1_vs_iter0_mismatch=2319`). Under-waiting is silent —
  no timeout, no assertion.
- Fix, not a workaround: the generator emits the granularity it fitted against
  (`TILEMEGA_GENERATED_WINDOW_TILE_M/_TILE_N/_SPLIT_K`) and the harness admits
  the narrow path only when the compiled granularity agrees, degrading to
  `kAll` — always a superset — otherwise. Every run reports which it got, in
  `E2E_KAPPA … wait_table=exact|degraded`. ✅ **50 / 50** on both branches after
  the fix, hash `8a737188b958a2ae` (degraded) and `5245714bc5d3ab4d` (exact).
- It also invalidates a measurement discipline: every COARSEN arm compiles with
  `-include plan_<model>_uniform.h`, so every arm reports `degraded`. The κ
  cost sweep therefore still measures κ's cost against a `kAll` table — the
  old caveat survived in a new form and had to be re-stated rather than dropped.
- Generalization: any future optimization that bakes a shape constant into
  generated code needs the same self-check. Its failure mode is a wrong answer.
- Evidence: docs/experiments/COARSEN/result.md §8, raw/waittable/fresh_processes.txt.

## F-61 — Absolute latencies do not survive a session boundary; only within-session pairing does

- Finding: the byte-identical pre-wiring binary `E2E_GEN/generated_e2e` was
  recorded at `l2/l1 = 1.0136×` on 2026-09-03 and measures **1.036713×** on
  2026-09-04, 25 fresh processes, CI [1.035648,1.036862]. The absolute L1
  median moved within a single day too, 0.998400 ms → 1.082560 ms, and
  `nvidia-smi` reports the SM clock idling at 210 MHz against a 3105 MHz
  maximum. Nothing in the code changed.
- Second instance: the COARSEN κ sweep re-run reproduces every paired ordering
  conclusion while its medians drift by up to **1.8%** against the previous
  session's.
- Consequence, applied retroactively: the `1.014×`/`1.016×` L2-vs-L1 headline
  in earlier rounds is **superseded, not reproduced**, and every cross-round
  absolute number in this repository is a historical record rather than a
  comparison. Any comparison must be re-measured in one session with a control
  in that same session — which is why the attribution in F-58 is four builds
  measured together rather than one build measured against a memory.
- Evidence: docs/experiments/E2E_L2/result.md, raw/part3/paired_prev_binary.txt;
  docs/experiments/COARSEN/result.md §3.5.

## F-62 — The resource vector is nine lanes; on sm_89 one lane decides the ranking, and it is not identifiable

- Finding: `ResourceVector` had been trimmed to the six lanes sm_89 exercises.
  It is restored to §2.2(a)'s nine — ⟨tc, cuda, sfu, tmem, smem, l1_5, l2, ddr,
  net⟩ — with `TargetSpec::Caps` deciding which participate. A zeroed lane is
  never bare: each carries a `LaneStatus` ∈ {kLive, kCapabilityAbsent,
  kNotCalibrated}, so a max that skips a lane can still say why.
- On sm_89 the audit reports live = [tc cuda sfu smem l2 ddr],
  capability_absent = [tmem l1_5 net], not_calibrated = []. ✅ Keeping only the
  SMEM lane changes MAPE by **0.02 points** and ρ by **0.0006**: one lane
  decides the ranking on this target.
- ⚠️ But which lane it is, is not a fit result. The SMEM and L2 lanes are
  **collinear over every calibrated shape** — the mainloop's shared-memory
  traffic and its L2 traffic are the same tile loads counted twice — so
  attributing the ranking to SMEM is a micro-architectural argument, not
  something the data separates. It must be re-measured on sm_90+, where TMA
  moves the L2 side without moving the SMEM side.
- Related, and recorded so it is not quietly counted as gain: in the layered
  ablation the SDCM cache model (`+cache`) changes MAPE and ρ by **exactly 0**.
  The whole gain over the analytic ranking is split-K (ρ 0.4435/0.4303 →
  0.9071/0.9043) and the wave tail (0.9095/0.9070 → 0.9450/0.9435).
- Evidence: docs/experiments/COST_MODEL/result.md, raw/summary.tsv;
  include/tilemega/Solver/CostModel.h; TileMega_skeleton.md §4.4.

## F-63 — A target audit finds what per-target tests do not: the missing field with no reason

- Finding: `tools/tilemega-target-audit` checks, for each of sm_80/89/90/100/120,
  the JSON field set against `TargetSpec`'s full field set, the status of each
  of the nine resource lanes, whether each cost-model term is evaluable or
  explicitly zero, and that the target cross-compiles. Anything missing **with
  no reason attached** fails.
- ✅ It found 12 real failures on first run — `calibration/device`,
  `measured_at` and `wall_seconds` absent from four config files. Fixed by
  adding the fields, not by exempting the check.
- It also had to be taught the difference between "absent" and "absent for a
  reason": constructing a `CostModel` on an uncalibrated target throws, and the
  audit now reports `UNAVAILABLE reason=not_calibrated` rather than crashing —
  without weakening `CostModel`'s own guard, which is what stops a fabricated
  constant from being quoted.
- Current state: `SUMMARY targets=5 failures=0`, including the optional
  `calibration_by_dtype.bf16` schema, wired into ctest as `target_audit`
  (ctest is this repository's CI). ✅ 24/24 tests pass.
- Evidence: tools/tilemega-target-audit.cpp, configs/targets/*.json.

## F-64 — The migration check is built and baselined, and refuses to run on the calibration GPU

- Finding: a 4090 and a 5090 have comparable shared memory per SM (~100 KB) but
  128 → 170 SMs, which hits wave quantization directly — the one term the cost
  model gets its largest single gain from (F-62). `tools/tilemega-migrate` plus
  `docs/experiments/MIGRATION/run_on_sm120.sh` re-run an ORACLE subset (top-50
  by measured latency + 50 random, fixed seed) on a different GPU using the
  4090-calibrated model, and report rank correlation and top-hit-rate loss.
- ✅ Baseline on the calibration target itself, so the transfer arm has
  something to lose against: gqa2 n=100 MAPE 27.48, ρ **0.9144**, optimum_rank
  13; mha4 MAPE 25.75, ρ **0.9095**, optimum_rank 8. The subset is harder than
  the full sweep **by construction** — it is 50 near-optimal configurations plus
  50 random ones — so top-3% hit rate is 0 by construction and Spearman,
  optimum_rank and top-10 are the metrics that carry meaning here.
- ❌ The transfer arm has **not run**: no sm_120 is present. `--probe`
  hard-fails with exit code 3 on the calibration GPU rather than degrading to a
  same-GPU comparison that would look like a pass.
- Evidence: docs/experiments/MIGRATION/result.md, raw/summary_sm89_baseline.txt.

## F-65 — Runtime variants must own both the implementation and its dependency table

- Finding: a GEMM tile is not merely a backend tuning flag.  The tile changes
  the producer task space, so every affine `StageDependency` constant derived
  from that space is part of the same variant.  Keeping the dependency table
  fixed while changing `TILEMEGA_GEMM_*` produced a silent under-wait and 0/50
  correct processes in the earlier COARSEN regression.
- The generator now independently instantiates and derives each runtime
  variant and emits `{GemmRuntimeDesc, StageDependency[]}` together in
  `ModelSpec`.  The host selects a `seq` interval by one indexed lookup.  The
  old exact/degraded check is gone because no cross-granularity table exists.
- ✅ Two intervals of one BF16 executable select different tiles and pass
  50/50 at both `seq=4` and `seq=2048` on both models.  The old FP32
  `16x64x16s2k16` failure is now 50/50 on both models with `variant_exact`.
- Evidence: `docs/experiments/VARIANT/`.

## F-66 — Variant capacity is register-bound before the shared-storage union grows

- Finding: `sizeof(TaskBodyUnion)` is not a sufficient Phase-5 capacity
  metric.  For 1/2/4 variants the union remains 16 KiB, but ptxas registers
  rise 80/85/114 and occupancy falls from 5 CTA/SM to 4 at four variants.
  At 8 and 16 variants the measurements are 218/255 registers, 18/96 KiB,
  and 2/1 CTA/SM.
- ✅ The no-occupancy-loss budget on this sm_89 implementation is therefore
  **two variants per binary**.  Same-operator and multi-operator mixtures have
  the same maxima for the measured ladder: C++ union storage takes a maximum,
  while compiled dispatch paths still raise register pressure.
- Consequence: Phase 5 must merge near-equivalent adjacent intervals or split
  them across binaries; allowing 16 template parameters does not make 16 a
  performance-safe interval count.
- Evidence: `docs/experiments/VARIANT/curve.tsv`.

## F-67 — BF16 is a semantic dtype and uses a distinct calibrated resource regime

- Finding: dtype is now extracted from ExportedProgram FakeTensor metadata and
  carried by L-sem, the implementation registry, generated `ModelSpec` and all
  TaskBodies.  BF16 stores two-byte values while GEMM, norm and attention
  reductions accumulate FP32.  This is not a storage-only path: cuobjdump finds
  96 `HMMA.16816.F32.BF16` instructions in each accepted model executable.
- ✅ Both models pass L0/L0.5/L1/L2 in 50/50 fresh processes.  The accepted
  BF16 profile records 179.997 TFLOP/s Tensor Core throughput, 97.37% of the
  DRAM pin and 0.00073% start/end drift.  It is stored beside—not over—the
  existing FP32 calibration.
- ✅ FP32's ρ cannot be transferred.  The dtype-aware model populates `tc`,
  but its contribution requires the final BF16 oracle sweep.  The requested
  identifiability retest already shows SMEM/L2 remain structurally collinear:
  their common feature yields a constant 3.4715200776 ratio and Pearson 1 over
  the 20 calibrated occupancy points, so the current fit still cannot assign
  their effects independently.
- Evidence: `docs/experiments/BF16/`, `configs/targets/sm_89.json`.

## F-68 — Ownership Place, unlike cache-locality Place, unlocks the event graph

> ⚠️ The TaskBody ownership correction remains valid, but all latency and poll
> values below were measured before the task queue and are superseded by
> F-79/F-86.

- Finding: changing only RoPE to tile ownership made adjacent inverse images
  narrowable and improved seq=128 L2 by 3.23%/6.40% without a material body
  regression, satisfying the predeclared promotion rule.  Extending the same
  contract to KVAppend, activation and the split-K combiner changes GQA's
  all/identity/window mix from 20/3/15 to 10/7/21.
- ✅ At seq=128, exact polls fall 482316 → 282636 (−41.40%) and the
  exact/relaxed ratio falls 0.999925 → 0.520240.  In 25 fresh-process paired
  rounds L2 improves 13.57%/16.32%; body-only L0.5 costs 1.00%/1.25%.
- This is a Place decision about the mapping from logical tasks to CTAs.  It is
  independent of the previously rejected objective that co-locates adjacent
  tasks for cache reuse.  Calling both “Place” without this distinction hid
  the largest available synchronization lever.
- Evidence: `docs/experiments/OWNERSHIP/`.

## F-69 — A seq matrix is useful only if an obsolete implementation fails it

> ⚠️ The TaskBody fixes remain valid. The old clamp negative is unreachable
> after queue conversion and now passes 50/50; F-85 replaces it with a live
> `TaskWait` deletion control that fails 50/50.

- Finding: expanding the runtime dimension uncovered two bugs unrelated to
  numeric tolerance: Attention and RMSNorm declared grid-stride ownership but
  executed only their first placed task; Attention's shared score row was also
  sized by CTA width rather than runtime total length.  Both are fixed, with a
  hard supported-total bound of 4096.
- ✅ Two models × five seq values × three past values × 50 fresh processes pass
  **1500/1500**, comparing L0, L0.5, L1 and L2 in every cell.
- ✅ The deliberately obsolete `min(count, grid)` clamp fails 50/50 at
  `seq=2048,past=0`.  This negative is the evidence that the expanded matrix
  actually observes the silent under-wait class; a large green matrix alone
  would not establish that.
- Evidence: `docs/experiments/SEQSCAN/`.

## F-70 — BF16 breaks the cost model, and the culprit is a constant that FP32 hid

- Finding: re-measured on its own validation set — 1540 generated, compiled and
  measured points under BF16 with the structured ownership — the calibrated
  cost model reaches ρ **0.5605** (gqa2, n=770) and **0.6239** (mha4, n=462),
  against 0.9450 / 0.9435 in FP32. `top1 = top3 = top10 = 0` on both models and
  the true optimum ranks 104th / 51st. ✅ Part 2.4's acceptance ("no worse than
  FP32") is **not met**, and no threshold was moved to meet it.
- The uncalibrated analytic ranking (`tier2-baseline`) reaches **0.8778 /
  0.8738** on the same points and ranks the optimum 25th / 19th. In BF16 the
  calibrated model is *worse than the baseline it was built to replace*, having
  beaten it 2:1 in FP32. That is what makes this a finding rather than drift.
- ✅ Attribution, and it is not §2.2's structure. The ladder localizes it:
  `+splitk` is the one layer that *lowers* ρ (gqa2 0.5211 → 0.4926) where in
  FP32 it produced the entire gain (0.4435 → 0.9071). The model's eight best
  configurations are all split-K 16, predicted at 0.099–0.107 ms and measuring
  0.207–0.291 ms. Behind that: `combine_fixed_ns` is **0** in the BF16 profile
  and **108.1 ns** in FP32. The calibrator fits the reduction stage's
  width-independent term and stores `max(0, fit)` because the intercept is
  honestly unresolved (`|value| < 300 ns`, 150% spread, either sign,
  `GemmCalibration.cu:479`); in FP32 the fit landed positive and the clamp never
  bound. The model therefore charges ~0.12 µs for every reduction in the whole
  graph and split-K is nearly free.
- Second-order: three of the six BF16 Stream-K points fit a **negative** per-CTA
  setup (`a_ns` −119.2 / −438.6 / −400.7) against `256x128x16s3`'s +10342.1,
  with `fit_r2` 0.922 versus FP32's 0.974. A setup time cannot be negative.
- The transferable lesson: **an absolute uncertainty that is harmless under a
  slow mainloop becomes decisive under a fast one.** Nothing about the constant
  changed; the denominator did. Every clamped or unresolved constant in
  `TargetSpec::Calib` should be re-examined whenever the pipeline it competes
  with gets faster.
- Evidence: docs/experiments/ORACLE/result.md §6.7, raw_bf16/cost/summary.tsv,
  raw_bf16/cost/predictions_gqa2.tsv, configs/targets/sm_89.json.

## F-71 — The `tc` lane, the reason nine lanes were restored, changes the ranking by 0.001

- Finding: Part 2.1 predicted that BF16 would make the Tensor Core lane the
  bottleneck for the first time and thereby justify the nine-lane resource
  vector (F-62). ✅ Measured on the BF16 oracle, one-variable lane ablations:
  removing `tc` moves ρ from 0.5605 to **0.5595** (gqa2) and 0.6239 to 0.6230
  (mha4), and MAPE by 0.01 points. Removing `cuda`, `sfu`, `tmem`, `l1_5`, `l2`,
  `ddr` or `net` changes **nothing at all**. Only `smem` moves MAPE (39.41 →
  53.78) — and removing it slightly *improves* ρ.
- So six of the nine lanes are exactly inert on sm_89 in BF16, and the premise
  is not confirmed: a faster mainloop moves the bottleneck *away* from the
  compute lanes rather than into them. The nine lanes are still the right
  carrier for a port, but they are not what ranks configurations here, and the
  earlier "this is why we keep nine dimensions" argument is withdrawn on this
  target.
- The SMEM/L2 identifiability retest Part 2.3 asked for is also negative and for
  a structural reason: both lanes are built from the same
  `occupancy · 2 · Tk · (Tm + Tn)` feature, so their ratio is constant at
  3.4715200776 with Pearson 1 over all 20 calibrated occupancy points. ❌ dtype
  alone cannot separate them; that needs a target where TMA moves one and not
  the other.
- Evidence: docs/experiments/BF16/result.md, identifiability.tsv;
  docs/experiments/ORACLE/result.md §6.7.

## F-72 — An elementwise BF16 bound fitted on one split factor censors the oracle along split-K

- Finding: the BF16 comparison bound `1.6e-2 + 1.6e-2·|e|` was selected on the
  SEQSCAN matrix, which holds the split factor fixed. On the oracle's split-K
  axis it does not hold: ✅ **154/154** mha4 configurations fail at split-K 2
  and **154/154** at split-K 16, while **154/154** pass at each of 1, 4 and 8 —
  perfectly determined by the split factor and independent of tile shape.
- It is the comparison, not the implementation. Every failing run has
  `l1_vs_l05_mismatch = 0` and `l2_vs_l1_mismatch = 0`: TileMega's three levels
  are bit-identical. At split-K 2 exactly **one** element of the whole model
  exceeds the bound — `actual −0.96875, expected −0.9375, delta 0.03125` against
  a tolerance of 0.031000, over by 0.8%. Configurations that *pass* carry larger
  deviations: split-K 8 reaches `max_abs = 0.0625` and clears the bound only
  because it lands on an element with a larger `|e|`.
- **The threshold was not widened.** The consequence is recorded instead:
  `screen_mha4.tsv` holds 462 of 770 points, censored along `split_k` — a
  decision variable — so mha4's ranking statistics are not comparable with
  gqa2's uncensored 770, and mha4's optimum is the optimum of {1,4,8} only.
  gqa2 (2 layers) is unaffected at 770/770; the effect appears with depth.
- `TILEMEGA_DIFF_DUMP=n` was added to print the first `n` offending elements
  with both values and the bound, because `max_abs` and `max_rel` are
  independent maxima and identify no element.
- Evidence: docs/experiments/ORACLE/result.md §6.7, raw_bf16/screen_mha4.tsv;
  include/tilemega/Codegen/tasks/ModelHarness.cuh.

## F-73 — Two silent breakages that only a full pipeline run could expose

- Finding: the runtime-variant work renamed the generated dependency table from
  `kDependencies[]` to per-variant `kDependencies0[]`, `kDependencies1[]`, …
  `ModelDescription::FromGeneratedCuda` still searched for the old name, so
  `tilemega-costmodel` threw on every generated source. Unit tests and ctest
  pass without touching that path; only running the oracle end to end reaches
  it. The parser now accepts either name and reads variant 0, which is correct
  because variants differ in the affine window constants, not in which stage
  feeds which.
- Second: the oracle's screening loop used `local tag=... bin="${raw}/bin/${tag}"`.
  Bash expands every word of a builtin before performing any assignment, so
  `bin` read an unset `tag` and `set -u` aborted the sweep on its first
  configuration — after all 1540 compiles had completed. Split into two `local`
  statements.
- Both are the same class: a change validated by unit tests, and a stage that
  only a full-pipeline run executes. The ORACLE sweep is that run, and it is the
  only thing in the repository that exercises generator → parser → cost model in
  one pass.
- Evidence: lib/Solver/ModelDescription.cpp, docs/experiments/ORACLE/run_bf16.sh.

## F-74 — The BF16 ranking collapse was not the clamped constant; it was one scalar and one mis-priced lane

- Finding: F-70 attributed the BF16 cost model's ρ 0.5605 / 0.6239 to
  `combine_fixed_ns` being clamped to zero. ✅ That attribution is **falsified**.
  Both defects it names were real and both were repaired — the launch baseline
  is now the same kernel doing nothing rather than an empty one, `a` is measured
  at zero mainloop iterations instead of extrapolated, and the clamp is replaced
  by a `combine_fixed_resolved` flag — and the ranking moved only 0.5605 → 0.5560
  and 0.6239 → 0.6235.
- The evidence that the old baseline was wrong is arithmetic, not statistical:
  the launch-subtracted duration of the reduction at **one** output measured
  **−928 ns**. An empty `__global__ void f(){}` has neither the shared-memory
  footprint nor the launch bounds of the kernel it stands in for. After the fix
  no BF16 Stream-K shape fits a negative per-CTA setup and `ac_r2` rises from
  0.922 to 0.984–0.9997; FP32's rises to 0.9992–1.0.
- ✅ The first real cause: the cost model prices all 154 shapes through **one**
  fitted scalar `setup_ns`. In FP32 the calibrated `a` spans 993–4501 ns and one
  scalar is fair; in BF16 it spans 0–10112 ns and the fit reported
  `setup = 1187.94 ns, rms 3247 ns` — a residual 2.7× its own value. Fitting
  `setup = α + β·tile_m·tile_n` over the same points (two numbers, nothing newly
  measured) moves ρ to **0.8246 / 0.8247**.
- ✅ The second: the **SMEM lane prices a path BF16 does not use**. Its rate is a
  scalar `ld.shared` throughput and its work term scales with mainloop
  iterations, so splitting K divided a cost the Tensor Core kernel never pays.
  The signature was a clean monotone bias — median predicted/measured 0.93 at
  split-K 1 falling to 0.55 at split-K 16. Marking the lane `kNotCalibrated`
  for BF16 flattens it to **0.46–0.48 at every split factor** and lifts ρ to
  **0.8942 / 0.8834**.
- Part 2.4's minimum ("no worse than the uncalibrated analytic baseline",
  0.8778 / 0.8738) is now **met**; its target (FP32's 0.9450 / 0.9435 with top-3
  inside the measured top 3%) is **not**, and no threshold was moved. What
  remains is a near-constant 2.1× under-prediction — a scale error, not a shape
  error — because nothing replaces the removed lane's contribution to absolute
  time.
- ⚠️ Recorded as it happened: the SMEM lane was found harmful by ablation and
  explained afterwards. The explanation is a micro-architectural argument about
  `cp.async` → `ldmatrix` → MMA, not a fit, and needs a target where the operand
  feed can be measured apart from byte traffic.
- FP32 is the control throughout and is unaffected: ρ 0.9450 → 0.9432 and
  0.9435 → 0.9421 across all three changes, optimum rank 19 → 23 and 12 → 14.
- Evidence: docs/experiments/BF16/result.md, docs/experiments/ORACLE/raw_bf16/cost/,
  docs/experiments/COST_MODEL/raw/summary.tsv; lib/Target/GemmCalibration.cu,
  lib/Solver/CostModel.cpp.

## F-75 — Metrics derived at the bottom of the parameter range were wrong by up to five orders of magnitude

- Finding: the frontend pinned every symbolic dimension to its range minimum
  before deriving, so `wait`, `fanout`, `volume` and `count` reached the IR as
  integers measured at `S_min`. Only tile sizes and the GQA group factor
  actually have to be literal (`isl_aff_div` rejects a parametric divisor); the
  workload dimensions do not.
- ✅ With them free, the metrics come out as quasi-polynomials, and evaluating
  them equals re-deriving the whole coupling at that point: **420 / 420**
  agreements over 21 edges × 4 metrics × `S ∈ {1, 4, 128, 512, 2048}`, compared
  as functions rather than collapsed to scalars.
- ✅ What the old constant claimed: **27 of 63** metrics were wrong. The one
  §2.7 names — `attn_chunk -> attn_combine`, `wait = ⌈L_s/Tkv⌉`, which is
  exactly what `T_sync = |image(C_κ)| × latency` reads — was **32× too small**
  at a 4096-token context, and the error grows linearly with KV length. Fanout
  on `rope_q -> attn_chunk` was 4096× low and its `count` 131 000× low.
- The generated kernel is unchanged: both reference models still pass 50/50
  fresh processes with bit-identical output hashes and the same
  10 `kAll` / 7 `kIdentity` / 21 `kWindow` dependency mix. What changed is what
  the IR can be asked, which is the precondition for P5.1's piecewise solution.
- ✅ The follow-up removed the three-point re-fit. `div/scale/offset/count`
  remain per-variant integers, but they are now discovered with symbolic isl
  lexicographic endpoints and admitted only after two-way set equality over
  the full parameter domain. Both references and the 16/32-layer graphs use
  zero per-edge fallbacks.
- Evidence: docs/experiments/SYMBOLIC/, tools/tilemega-symbolic-probe.cpp.

## F-76 — L2's excess time is the poll, not persistent dispatch

> ❌ Superseded by F-79. This measurement correctly decomposed the program that
> existed, but that program was a stage-loop executor rather than the task
> queue required by §5.4. Its κ and component conclusions do not describe the
> replacement executor.

- Finding: a four-arm, within-round decomposition isolates L2's stage loop,
  event notify, event wait, and L1's grid barrier. At kappa=1 and seq=128 the
  wait accounts for **104.1% / 103.9%** of the L2-minus-L1 gap on gqa2/mha4;
  notify is 7.4%, L1's barrier offsets 8.5%, and L2's bare stage loop is
  12--28 us faster than L1's. Task-table reads, dispatch, grid width, and
  occupancy are therefore not hidden positive costs in this harness.
- The 2.13x ratio that prompted the attribution is a kappa=1 build. At the
  shipped kappa=0, seq=128 is **1.055x / 1.061x**, and the ratio falls rather
  than rises with sequence length.
- The implementation serialized every independent dependency poll on thread
  zero. Distributing them over the CTA improves the kappa=1 L2 path by
  **15.05% / 14.78%** in 25 paired rounds (bootstrap intervals exclude zero,
  Wilcoxon p=1.29e-05), without changing either model's output.
- Evidence: docs/experiments/L2_ATTRIB/, commit fdcbbe6.

## F-77 — A host launch sweep cannot test the epoch ABA argument

- Finding: 32 launches over event memory that is never cleared pass in 50/50
  fresh processes for both models, but the intended negative control -- clear
  the counters and always announce iteration zero -- also passes 50/50.
- This is structural, not a weak stress test. `iteration` is a launch argument,
  launches use one CUDA stream, and each launch is synchronized before the
  next begins. No CTA from iteration i can overlap iteration i+1, so the ABA
  interleaving is unreachable regardless of the process count.
- The monotone target remains a valid defensive rule, but its necessity can
  only be tested once the Phase-6 loop runs iterations inside a persistent
  kernel (or otherwise overlaps them). The green host-launch matrix is not
  recorded as coverage of that property.
- Evidence: docs/experiments/AUTOREGRESSIVE/.

## F-78 — Real width makes correctness a first-class solver constraint

- ✅ A production-shaped 16x2048 decoder (973,144,576 parameters) generates,
  compiles, and runs, but fails the unchanged BF16 PyTorch bound in 0/50 fresh
  processes. The boundary is depth: 4x2048 passes, 8x2048 first fails by three
  elements, and the 16-layer final hidden owns 190 of 198 mismatches. L0.5,
  L1, and L2 are bit-identical, so this is numerical accumulation rather than
  synchronization.
- ✅ The permitted real-width control, 4x4096 / intermediate=14336, has
  872,448,256 parameters and passes 50/50. Its existing declarative pattern
  needs no new slot or target-name branch.
- ❌ The BF16 cost model's two leading uniform configurations at real width
  use split-K 16 and 8. Both are fast and both violate the fixed comparison
  bound (24 and 26 mismatches). The best split=1 candidate passes 50/50 and is
  28.05% faster than the control on L1 in 25 paired rounds, but it is not the
  unconstrained model winner.
- Therefore Phase 5 needs a correctness-feasibility contract before its cost
  objective. This is not a request to loosen the BF16 bound: configurations
  outside it are inadmissible regardless of predicted latency.
- ✅ Ownership windows are now symbolic: the one-layer case falls from 63 s to
  0.861 s and the 16-layer graph to 362.035 s, with zero fallback. ⚠️ A measured
  32-layer graph still takes 10,872.781 s despite all 702 windows being
  symbolic, exposing a separate whole-graph coupling/proof scaling problem.
- Evidence: docs/experiments/REALMODEL/.

## F-79 — A green reference implementation can validate the wrong executor

- The generated and handwritten L0.5/L1 implementations agreed because both
  iterated stages. That acceptance checked equivalence to the reference, but
  the reference itself omitted §5.4's worker schedule and per-task wait.
- ✅ L2 now consumes a variant-exact worker queue. The generator rejects cycles
  and backward schedule edges; the host rechecks the expanded queue before
  launch. Two references pass 50/50 and the seq×past matrix passes 1500/1500.
- ✅ Fifty fresh-process traces, which the old acceptance lacked, observe the
  property being purchased: mean early starts are 68.72/200 and 205.70/512
  (34.36% / 40.18%; ranges 68–70 and 197–217).
- Consequence: all old κ, Place and L2 performance conclusions are invalidated,
  even when their statistical method was sound. Acceptance must observe the
  semantic distinction a feature is supposed to introduce.
- Evidence: docs/experiments/TASKQUEUE/, docs/experiments/OVERLAP/.

## F-80 — Normalize full fan-in, retain fine events for windows

- Each queue task receives a deduplicated `(producer, event-group)` list;
  monotonic epochs then lift waits already satisfied earlier in the same worker
  queue, while the remaining polls stay CTA-parallel.
- A full-fan-in CG edge is represented by one producer-stage aggregate event;
  otherwise κ=1 would expand a single task edge into O(producer tasks) polls.
  Narrow windows retain logical-task events, so aggregation does not erase the
  overlap benefit that κ is meant to expose.
- ✅ On the reference fixtures deduplication plus monotone lifting reduces
  580 raw descriptors to 500 and 1,284 to 1,076. With the logical-task event
  prefix table, the static wait
  control path is 50 `sm_89` instructions, including a 23-instruction retry
  loop.
- ❌ General MPK fan-in-one dummy expansion is still not competitive. After
  the aggregate handles kAll, even the lower bound of `wait_count-1` dummies
  expands 200 tasks to at least 544 (2.72x) and 512 to at least 1,208 (2.36x),
  before encoding dummy fan-out.
- Evidence: docs/experiments/TASKQUEUE/.

## F-81 — Symbolic windows remove the local constant and expose the global one

- `FitWaitWindowSymbolic` takes lexicographic endpoints of the existing isl
  coupling, constructs the row-major interval, and proves both subset
  directions over all free workload parameters. Unsupported edges alone use
  the concrete fallback.
- ✅ The two reference sources are byte-identical to their pre-change outputs;
  all 128 edges use the symbolic path. Real one-layer codegen falls from 63 s
  to 0.861 s (73.2x), and 16 layers take 362.035 s with 350/350 symbolic.
- ⚠️ The measured 32-layer run takes 10,872.781 s with 702/702 symbolic. Thus
  the three-point window fit was the one-layer constant, but full-graph isl
  construction/proof has a separate superlinear scaling debt.
- Evidence: docs/experiments/REALMODEL/symbolic_codegen.tsv.

## F-82 — Place became measurable, but critical-path order is not yet valuable

- ✅ The solver's order now changes the actual worker queues. A five-node
  exhaustive oracle has 3 feasible permutations out of 120; the scheduler
  matches the optimum maximum dependency span of 2.
- ✅ Both full models pass 25/25 under critical-path and numeric topological
  orders. The paired ratios are 1.000000 and 0.998552. gqa2's interval crosses
  one; mha4's interval barely favors round-robin but Wilcoxon p=0.063. Both
  effects are far below the predeclared 2% threshold.
- Place remains part of the execution contract because omitting it recreates
  the architectural defect. It is not claimed as a latency win; span alone
  does not price task duration or ready-queue slack.
- Evidence: docs/experiments/PLACE/.

⚠️ Read with F-126: measured under L1-identical ownership, stage-major queues and FIFO execution; not evidence about the value of placement, ordering or windows in general.

## F-83 — BF16's top ten identify a feature interaction, not a missing scale

- After correcting occupancy construction, BF16 reaches ρ 0.8984 / 0.8871 but
  still places none of its top-1/top-3/top-10 inside measured top 3%; the true
  optima are ranked 48/46. No threshold is moved.
- Predicted leaders split into narrow-N split=1 and aggressive split=8/16
  families, at only 0.36–0.47 predicted/measured. Yet both measured top tens
  favor narrow N with split=8, showing an interaction rather than a global
  scale correction.
- SMEM and L2 cannot be identified by any GEMM tile in the present model: both
  use exactly `occupancy·2·Tk·(Tm+Tn)`. BF16 therefore retains only the
  calibrated L2 byte lane; an independent shared-reuse/global-working-set
  microbenchmark is required to restore two coefficients.
- Evidence: docs/experiments/BF16/topk_diagnosis.tsv.

## F-84 — Task-queue overlap and iteration overlap are separate properties

- ✅ Queue-era tracing observes cross-stage overlap inside one launch, yet the
  reset-events negative still passes 50/50 for both models over 32 repeats.
  Therefore failure of that negative cannot by itself diagnose whether the
  stage barrier was removed.
- The missing edge is between launches: `LaunchL2` synchronizes the default
  stream and `iteration` is a launch argument. There can be no producer from
  iteration `i` still running when a consumer from `i+1` reads its epoch.
- ❌ The requested growing-past greedy decode is not implemented. Making ABA
  reachable requires a device-side iteration loop plus versioned activation
  buffers, KV-cache rotation and per-step reference tokens; a loop without
  versioning tests unrelated WAR/WAW races, while an iteration barrier hides
  ABA again.
- Repeated-launch L2/L1 remains in the single-forward regime (1.08–1.10), as
  expected when there is no cross-iteration persistence.
- Evidence: docs/experiments/AUTOREGRESSIVE/raw_taskqueue/.

## F-85 — A queue indexed by tasks still needs task-indexed events

- The first queue executor removed the outer stage loop, but grouped producer
  events by the CTA that owned a task and published only after that CTA's last
  task in a stage. This is correct but conservative when a stage has more
  logical tasks than resident workers: a consumer can still wait for unrelated
  work owned by the same CTA.
- ✅ Each stage now owns one aggregate row plus prefix-indexed
  `(stage, logical_task / kappa)` fine rows, and every completed `TaskRef`
  contributes its own arrivals. `TaskWait` ranges are constructed from
  producer logical-task coordinates, not producer owners; kAll uses the
  aggregate, and kappa=0 remains the explicit aggregate-only control.
- The initial queue-era latency, kappa, window, attribution and Place numbers
  are consequently treated as invalid and rerun. This is the same acceptance
  lesson as F-79 one level deeper: naming a table `schedule` or iterating a
  `TaskRef` does not establish task-granular readiness; the event key and
  publication point must be inspected too.
- ✅ A second negative control deletes the materialized `TaskWait` rows and
  fails 50/50 at seq=2048, while corrupting the retired stage-wait path still
  passes 50/50. The negative therefore distinguishes the live executor.
- Evidence: docs/experiments/TASKQUEUE/, docs/experiments/SEQSCAN/.

## F-86 — Fine readiness is measurable, but its value changes sign

- ✅ After kAll aggregation and selective publication, kappa=1 becomes the
  measured winner over aggregate-only by 0.285% / 0.165% in the two
  25-round sweeps. The old claim that kappa=0 is structurally optimal is false.
- ✅ Forcing every dependency to kAll separates from exact windows in all six
  seq cells. Exact is 0.694% / 0.322% faster at seq=4, but 1.171% / 1.163%
  slower at seq=512: growing fine-event publication outweighs earlier
  readiness on these DAGs.
- ❌ Observable overlap is therefore not sufficient for an end-to-end win.
  Final kappa=1 L2/L1 is 1.081–1.113 across the six primary cells. Four-arm
  accounting closes exactly; notify is the largest positive component, while
  the bare queue loop is faster than barrier-free L1.
- Evidence: docs/experiments/COARSEN/, docs/experiments/E2E_L2/,
  docs/experiments/L2_ATTRIB/.

⚠️ Read with F-126: measured under L1-identical ownership, stage-major queues and FIFO execution; not evidence about the value of placement, ordering or windows in general.

## F-87 — BF16 TC already wins 37.66% of the pre-change resource vectors

- ✅ T2.1 on baseline c4f4123 calls `CostModel::Steady` and
  `ResourceVector::BottleneckName` for every accepted dtype-specific ORACLE
  configuration. BF16 TC wins 290/770 GQA configurations and 174/462 MHA
  configurations; L2 wins the remainder. FP32 has 1047 SMEM and 30 CUDA
  winners out of 1077 on each model.
- ❌ The T1/T2/T3 task's proposed attribution “register-resident peak TC never
  wins, so BF16 degenerates to one feature” is falsified. The TC and L2 work
  terms already have different geometry (tile area versus perimeter). This
  neither validates the peak calibration nor resolves the poor ranking.
  Per the explicit stop rule, no achieved-rate substitution is attempted.
  The user clarified that this gate pauses only T2; independent T1/T3 work
  continues.
- ✅ The first proposed correction, notify rather than wait dominates the
  positive L2 overhead, agrees with F-86 and the existing four-arm evidence:
  mha4/128 has notify 171.008 us versus wait 53.312 us. ⚠️ The second proposed
  correction, single-word atomic fan-in amplified by RMW polls, remains an
  untested causal hypothesis; no T1 intervention has yet isolated it.
- ⚠️ The current BF16 MHA screen has 308 RUNFAIL rows and 462 PASS rows; the
  FP32 1077 denominator cannot be reused for BF16. The diagnostic preserves
  all per-configuration vectors and hashes its input files. It performs no
  GPU work and cannot attribute those existing failures.
- Code: `tools/tilemega-costmodel.cpp:219`, `lib/Solver/CostModel.cpp:118`,
  `lib/Solver/CostModel.cpp:297`, `lib/Solver/CostModel.cpp:301`.
- Evidence: `docs/experiments/COST_MODEL/result.md` T2.1 and `t2_before/`;
  reproduce with `docs/experiments/COST_MODEL/run_bottlenecks.sh`.

## F-88 — isl can schedule finite task C; proximity does not guarantee smaller worker-slot spans

- ✅ The offline probe derives C from one reference decoder variant, sets
  both isl validity and proximity to C, and checks every edge against the
  returned lexicographic schedule. Six cases (seq 4/128/512, workers 16/256)
  are legal; C plus the materialized worker queue edges is acyclic.
- ✅ At seq=512, workers=16, slot p95/max increase from 677/772 in the
  stage-major control to 797/896 after lexicographic flattening and cyclic
  worker assignment. A legal isl schedule is not itself a locality win.
- ⚠️ This enumerates a fixed finite 18-operator analysis graph, not the
  production 30-stage graph or a symbolic worker mapping. All grids fit
  resident_limit=256. No parametric bound or overresident I3 proof is claimed.
- The initial mapping enumeration was unbounded outside the actual task
  domain; intersecting the returned map with that domain fixes enumeration
  without weakening validity. The tool checks task count and uniqueness.
- Code: `tools/tilemega-affine-probe.cpp:107`, `:136`, `:150`, `:169`.
  Evidence: `docs/experiments/AFFINE_PROBE/result.md` and `raw/`.

## F-89 — T1 sharded notification fails its performance gate

- ✅ Frozen-source base versus load+split+S128, four arms and 25 paired rounds
  per model/seq: notify increases by 15.552 / 14.112 / 27.648 / 25.568 us for
  gqa2/4, gqa2/128, mha4/4, mha4/128. All paired 95% intervals are positive;
  all two-sided signed-rank p values are approximately 1.31e-5.
- ✅ Notify remains the largest positive component; loop is negative and
  the four-arm accounting identity closes with zero numerical error. Load
  alone, split lines alone and their combination previously showed no
  significant notify reduction. In the four-arm definition, poll/notify
  contention interactions belong to full-minus-nowait, not nowait-minus-neither.
- ❌ The second proposed attribution's intervention prediction is not met.
  Extra first-level atomics, plan reads and compiler resource costs are
  plausible explanations, not isolated findings. No claim that atomic
  contention is absent follows from this negative intervention.
- ✅ Final full arms plus extra checks give 50/50 fresh processes for each
  model/seq/state (400/400 total). Earlier independent load/line checks are
  800/800, shard exploration 1600/1600, and kappa=4 checks 200/200. None is
  mislabeled as the requested complete 1500 seq/past matrix.
- Per the user's stop condition, T1 is paused before the full matrix,
  seq=512 E2E and full kappa retuning. All switches remain available and off.
  T1.4 is intentionally unimplemented pending its release proof. Cluster B
  only cross-compiles; the sm_120 manual script is not a hardware result.
  T2 is independently paused by F-87; T3 is complete within F-88's scope.
- Code: `EventSync.cuh:13`, `ModelRuntime.h:234`,
  `ModelHarness.cuh:425`, `ModelHarness.cuh:580` (all in
  `include/tilemega/Codegen/tasks/`); `lib/Codegen/Codegen.cpp:364`.
  Evidence: `docs/experiments/L2_ATTRIB/t1_result.md`, `t1_final/`,
  `raw_t1_final/`; updated E2E_L2 and COARSEN status sections.

## F-90 — Warmup measurements expose an instance mismatch in the occupancy premise

✅ T0, baseline e305a9f. Shared benchmarking now uses 5 untimed warmups and
11 timed samples, resetting inputs/events outside timing. Historical cold
single-launch timings remain available behind an independent macro and are
not mixed into the new comparison. Per-arm ptxas and runtime resource fields
are recorded, including actual occupancy-query shared memory.

The tested BF16 instances have 128 threads, 212 L2 registers, 24576 shared
bytes and two CTAs/SM under both min-blocks=1 and =2. F-40 predicts 2/4/12
for register/shared/thread limits. No spills occur. Thus this experiment did
not exercise a one-to-two-CTA tradeoff; the supplied 256-thread example was
not this instance. No thread count or TaskSmem semantics were changed to
manufacture the desired transition. Correctness is 400/400; four-arm 25-round
paired measurements do not establish an occupancy gain. This changes the
current fusion shared budget to 26624 B, not 1152 B.

Code: `include/tilemega/Codegen/tasks/Benchmark.cuh:1`,
`docs/experiments/L2_ATTRIB/run_t1.py:1`. Full data and ptxas logs:
`docs/experiments/OCCUPANCY/result.md`, `raw/`, `report/`.

## F-91 — BF16 split partial storage is a numerical defect, independent of ranking attribution

✅ FP32 partial epilogues and FP32 combination preserve the exported
Linear→BF16→residual boundary. At seq=128/past=3, both reference models and
split=1/2/4/8/16 pass 50 fresh processes each: 500/500. The paired baseline
fails many split settings; error is not monotonic in split. No tolerance
changed. Storage doubles for split>1; the cost model charges the additional
epilogue and combine traffic, with an explicit missing-calibration error.
The FP32 CPU cost regression is bit-identical on 2154/2154 configurations.
This does not prove the unrerun 973M or 4×4096 configurations acceptable.

✅ Original-binary replay classifies all 308 mha4 RUNFAIL entries as numerical
criterion failures (154 split=2, 154 split=16). The historical shell driver
classified every nonzero exit as RUNFAIL before examining mismatch output.
The replay logs are new evidence, not recovered historical stdout.

Code: `include/tilemega/Codegen/tasks/ModelRuntime.h:22`,
`GemmStageTaskBody.h:391`, `GemmCombineTaskBody.h:22` in the same directory;
`lib/Solver/CostModel.cpp:372`. Evidence: `docs/experiments/BF16/result.md`,
`raw_splitk/`, `runfail_audit/`. F-87's rejected TC attribution remains rejected.

## F-92 — A concrete-input equality gate is necessary but insufficient for symbolic consumption

✅ Direct verified-CG model input and parameter binding agree with the
archived generated-input path on all five CostBreakdown double bit patterns
at 2154 FP32 points. A finite integer-domain DP on S=1..16 produces one piece
per model, with all 32 concrete choices and cost bits matching independently
bound historical input. ⚠️ The coupling metrics are not yet fully priced,
and this enumerator is not a general symbolic-intersection DP. The input-only gate must not be called
T1 acceptance. Initially S/past binding missed CG's s11/s14 names and aliases;
the input gate still passed, illustrating exactly why carry-only metric
parsing cannot prove consumption. Frontend now records semantic roles.

The existing cache model contains sqrt/exp, and repeated-wave IEEE addition
is not equivalent bitwise to multiplication by a wave count. Design (b) is
documented explicitly, without claiming generic quasi-polynomial max roots.
Code: `lib/Solver/ModelDescription.cpp:136`, `tools/tilemega-parametric.cpp:1`;
design and gate: `docs/experiments/PARAMETRIC/`.

## F-93 — Band-aware placement improves finite-graph locality but can worsen balance

✅ Six offline instances show more same-worker edges for band tiling and
wavefront than stage-major. Cross-worker proportion is the complementary
statistic, not independent evidence. For seq=128/workers=256, same-worker
fraction rises .004301→.026898 while the longest queue rises 22→425.
No latency improvement follows from this count alone. All finite C + queue
graphs are acyclic and wholly resident; this is not an overresident I3 proof.
⚠️ T4.2 production integration and GPU acceptance remain undone.
Code: `tools/tilemega-affine-probe.cpp:1`; evidence:
`docs/experiments/AFFINE_PROBE/raw_mappings/`, `result.md`.

## F-94 — The isl exit warning was a point-space leak, not just static teardown

✅ A scoped reference audit caught `Points before=0 after=4`. CollectPoint
called the owning `isl_point_get_space` twice without freeing either result.
It now owns one space and point with RAII. Tool entrypoints explicitly own
IslContext; SharedIslContext only borrows the live owner. Teardown checks
the pinned isl implementation's actual reference count and aborts on leaks.
The parametric tool and all six affine cases explicitly print remaining=0;
24/24 tests pass. This is lifetime evidence, not a GPU race claim.
Code: `lib/Analysis/CouplingRelation.cpp:184`, `lib/Analysis/ISLContext.cpp:14`;
before/after logs: `docs/experiments/ISL_LIFETIME/`.

## F-95 — Task-count times a constant poorly predicts steady-state event arms

✅ Reanalysis of 25 paired rounds per cell on sm_89 and supplied sm_120
logs uses only warmup=5/repeat=11 data. Three-cell fits predict short-seq
notify/wait 94–98% below observation. Effective rates are 14.0171/1.43874
ns (sm_89 notify/poll) and 11.7653/1.54380 ns (sm_120), but the residuals
prevent describing them as a validated price model. Fence cannot be
separated from these four arms and remains not_calibrated. Rates retain
explicit ns/runtime_task_ref and ns/runtime_wait_entry units in TargetSpec.
Code/data: `docs/experiments/EVENT_COST/calibrate.py:28`, `calibration/`.
❌ Parallelism/fixed per-worker work is a possible explanation, not isolated
by these data. The rejected atomic fan-in attribution is not revived.

## F-96 — Stored wait and fanout need not count the same physical task graph

✅ The real BF16 CG input at seq=4/past=3 gives wait_sum=512 and fanout_sum=16
on the first gqa2 edge. The producer side of wait spans nominal 128 rows;
fanout counts only four physical rows. CouplingDerivation intentionally
restricts only fanout because of an earlier barvinok symbolic counting
failure (`lib/Analysis/CouplingDerivation.cpp:490`). Existing QP point checks
do not prove physical-incidence equality. Runtime window clipping means
this finding alone is not a runtime correctness defect.

The event-price prototype was removed before commit: besides that domain
mismatch, multiplying an ns/runtime-task fit by an event-image cardinality
has the wrong meaning. No κ price delta or T1.5 success is claimed. The
retained audit rejects instead of treating carrying metrics as consumption.
Reproducer: `tools/tilemega-event-cost.cpp:1`;
`docs/experiments/EVENT_COST/metric_audit.txt`, `result.md`.

## F-97 — Queue caps recover a real offline locality/balance Pareto improvement

✅ All six finite instances have a balanced map with more same-worker edges
and a longest queue no longer than stage-major. At seq128/workers256 the
same-worker fraction goes .004301→.014090 with longest queue fixed at22;
at seq512/workers16 it goes .062981→.106826 with queue913 unchanged.
All 78 maps pass C+queue acyclicity checks. Looser caps are not uniformly
better. No GPU timing or overresident I3 proof is implied; exact production
task/stage/split projection and valid event pricing remain open.
Code: `tools/tilemega-affine-probe.cpp:233`;
data/plot: `docs/experiments/AFFINE_PROBE/balanced/`.

## F-98 — BF16 input equality includes all classified failures, but is still not a price gate

✅ The actual BF16 archived universe is 770 configurations per model,
1540 total, not the FP32 2154. All 308 mha4 numerical failures are included,
recovering occupancy/shared fields from classified replay logs rather than
discarding rows or inventing timings. Five cost-double bit patterns match
the independent generated-input path at 1540/1540 points with FP32 partials.
All three concrete DP modes and 32 finite-domain points match; finite BF16
enumeration changes configuration at seq8 in each model. FP32 retains its
2154/2154 bit gate and byte-identical prediction/finite-DP TSVs.
This does not retire (b) as an implementation, prove (a), rerun GPU numerical
acceptance, or correct BF16 rho/top-k. F-96 blocks the event functionality
gate independently. Code: `tools/tilemega-parametric.cpp:1`;
evidence: `docs/experiments/PARAMETRIC/input_gate_bf16.tsv` and logs.

## F-99 — Exact finite component enumeration avoids one pathological codegen

✅ Wide4×4096 split8 compilation was interrupted after1089s, sampled in
isl disjointization from Points(). Enumerating each basic component then
deduplicating the finite union completes the same input in17.1849s.
Independent overlapping-set tests pass; reference gqa2/mha4 and16-layer
generated CUDA are byte-identical. The16-layer compiler control is
321.8086→322.9925s: **no observed speed benefit** there. These are individual
CPU compiler timings, not GPU performance statistics or a universal scaling
claim. Macro TILEMEGA_ISL_COMPONENT_ENUMERATION retains the old control.
Code: `lib/Analysis/CouplingRelation.cpp:222`; evidence:
`docs/experiments/ISL_LIFETIME/enumeration_result.md`.

## F-100 — Fixed-prefix depth growth is on the observed BF16 noise scale

✅ Six depths×50=300fresh processes with common input/weight-prefix hashes.
Depth2/4/6 each pass50/50;8/12/16 each fail0/50 under the unchanged criterion.
max_abs=.015625/.03125/.03125/.046875/.0625/.078125, with4 rather than
hundreds of mismatches at depth8. All300 inter-level hashes and existing
second-iteration comparisons agree. Final-hidden depth16 errors against
common FP32 are1.26496978(PyTorch BF16) and1.25400052(TileMega BF16),
k_L2=.9913284393. This supports arithmetic accumulation/noise-floor
attribution in this fixture; it does not prove every shared defect absent
or authorize a replacement tolerance. Condition7's fixed criterion fails.

✅ CPU golden threads alone explain a historical discrepancy:8threads give
162mismatches,56threads give198, max_abs=.078125 in both, against the same
historical TileMega hash621738651f623f5b. Thus162 is not an accuracy gain.
The controlled thread pin predates the sweep; both deep runs still fail.
Code: `docs/experiments/REALMODEL/export_real.py:62`, `run_depth.py:79`,
`compare_golden_threads.py:10`; raw data and limits: `depth_result.md`.

## F-101 — FP32 partials pass FP32 regression but do not close the wide scene

✅ FP32 dtype partial-state regression passes400/400fresh processes:
two models×split1/16×partials0/1×50; paired output hashes agree.
`docs/experiments/BF16/run_splitk.py` and `fp32_regression.md` carry the
implementation and complete evidence. Actual FP32-partial combine-rate
calibration remains unmeasured; traffic accounting is not its substitute.

✅ The original4×4096 BF16 cost-leading split8, now using FP32 partials,
still fails with1mismatched element, max_abs=.046875. The runner stops at
0/1; split16 is compiled but unrun, and there is no50-process claim.
L0.5/L1/L2 hashes agree in that one run. Condition9 **remains open**;
the two-reference-model500/500 results on each architecture do not imply
this scene passes. No split ban or criterion change follows. Remaining
cause is unresolved; a data-domain-aware pre-cost feasibility design,
not an implemented legality proof, is in `REALMODEL/condition9_result.md`.
Code: `docs/experiments/REALMODEL/run_condition9.py:17`;
evidence: `condition9/k8_r0.txt` and `correctness.tsv` in that directory.

## F-102 — Round5 A0 closes condition9 as a criterion artifact

✅ The original failed wide split8 has k_L2=.9961308506568204 against common
FP32, inside the user-specified attribution interval. CPU threads56 reproduce
all original BF16 golden tensors bitwise. The missing tensor dump required
one original-binary GPU capture, explicitly approved by the user; its hash
and diff exactly match the old failed process. Tolerance was not changed.
Per the new task, conditions7/9 are closed as attributed criterion artifacts
and T2.d is cancelled, not implemented. This supersedes F-101's pending
feasibility proposal without rewriting the historical failure as PASS.
Code/data: `REALMODEL/condition9_noise.py:16`, `condition9_noise/result.json`.

## F-103 — Physical wait correction also requires physical C in verified CG

✅ A1's initial wait-only patch was rejected by CG verification: the stored
nominal relation still counted nonexistent producer tasks. Derivation now
persists and counts the same physically bounded relation. The verifier was
not weakened. BF16 production gates are1920/1920, versus372 unequal sums
in the explicit old-path control; first gqa2 seq4 edge changes512→16.
Eight reference graphs pass2505/2505; all26CTest pass. Real ComputeMetrics
error-exit audit reports before0/after0, and both gate tools end remaining0.

Boundary waits become genuinely piecewise; tests retain original full-tile
numeric anchors and add short-seq checks. Redundant printed inequalities
are checked by mutual containment, not updated output snapshots. A11's
44-edge/440-cell wiring comparison still has4 naming differences. The
OWNERSHIP poll statistic is runtime-side; labeling uses unchanged volume
and count. No old solver rank or GPU timing is said to be contaminated by
the previously unconsumed wait metric. Evidence: `INCIDENCE/result.md`;
code: `lib/Analysis/CouplingDerivation.cpp:546`, `test/unit/incidence_test.cpp:14`.

## F-104 — A2 exact runtime counts do not prove the runtime projection is sound

✅ Round5 A2 adds a shared verified-CG runtime plan and symbolic task/wait/
longest-worker counts (`lib/Solver/RuntimeProjection.cpp:35`). Split1 matches
3000/3000 archived task_refs/waits values across 1500 archived processes;
the codegen refactor is byte-identical in4/4 controls. CTest27/27, policy and
five-target audit pass. BF16 input1540/1540 and FP32 input2154/2154 still pass;
these are not A6 per-stage cost gates. No event price is connected yet.

✅ New one-process-per-cell capture with unchanged archived BF16 binaries
stopped at95/150 cells:94 passed, gqa2/seq512/past0/split16 failed with223287
L2-vs-L1 mismatches, max_abs1.1054688. L0.5/L1 hashes agree; iteration1 L2
also differs from iteration0. This is not the closed common-FP32 criterion
artifact and is not a new50-process correctness claim. No later cell ran.

✅ Static witness: `TaskInstantiation.cpp:170` appends chunk as the last
coordinate, while `GemmStageTaskBody.h:483` decodes chunk-major. The archived
first edge uses div64/scale128/count128, applied directly to runtime task id
at `ModelHarness.cuh:1263`. Task4 needs producer rows128–255 but waits for
rows0–127;192/256 tasks on this incoming edge are undercovered. This explains
why matching existing runtime counts cannot by itself establish semantic
validity, and why the old seq128 split sweep missed a multi-M-tile defect.
❌ The undercoverage is a plausible dynamic cause, not yet an exclusive
attribution established by repaired-code controls. A2 remains stopped,
without a kAll fallback, tolerance change or premature B implementation.
Evidence/code: `EVENT_COST/runtime_projection/result.md`,
`EVENT_COST/explain_split_projection.py:13`,
`EVENT_COST/runtime_projection/capture/gqa2_s512_p0_k16.txt`.

## F-105 — Kappa0 is not the start of a monotonic coarsening sequence

✅ The current policy makes kappa0 a whole-stage aggregate special case and
kappa1 a singleton-event policy. Exact gqa2 seq4/128 waits are244/4520 for
kappa0 and500/16292 for kappa1; only the latter has been matched to runtime
archives in A2. The user approved judging A9.3 by nonzero price difference
with direction consistent with actual counts, separately reporting positive
kappa coarsening. Kappa1's same-worker elision also precludes assuming a
monotonic positive-kappa curve without checking it. No runtime semantics or
price coefficients were altered to force the original stated direction.
This resolves a gate-definition conflict; A9 itself remains unimplemented.

## F-106 — Physical tail reads and nominal collective work are different domains

✅ Round5 A3's access-derived BF16 GEMM probe gives 4224 B/iteration at
seq4, tile128×128×16, N=K=512, versus historical MainloopBytes=8192.
Physical reads contain four actual activation rows; the collective work
contains a nominal 128-row tile. At seq128/512 both are8192. The user approved
separating these domains: physical R/W for actual access/fusion, nominal
work for the unchanged A6 historical GEMM bit gate. No tolerance changed.
The full analysis and explicit limitations are in COST_MODEL/round5_work.md.

Production and reference semantics omitted GEMM weights from R. Adding the
actual external (n,k) weight operand creates no CG producer edge, but makes
access counting complete for these GEMMs. Production two-model checks pass
210/210 count/nominal cells; repeated reads are unioned by tensor identity.
During implementation scalar `{0}` seeded QP sums failed on `[m,n]` spaces;
the first operand now seeds the sum, with permanent primitive regressions.
Both this probe and executed error paths leave zero ISL references. These
checks do not imply complete non-GEMM R/W, CG work consumption or A6 prices.

## F-107 — Split task enumeration repair resolves the focused L2 counterexample

✅ A2's concrete repair preserves storage layout but decodes logical task ids
in CG's `(m,n,chunk)` order, and builds partial→combine windows in that order.
Commit77c942e; independent `TILEMEGA_CG_SPLIT_TASK_ORDER` control.
At gqa2/seq512/past0/split16, interleaved fresh processes give old0/50 versus
new50/50; every L0.5/L1 hash stays8b8a3de9e7f35f9d and repaired L2 matches.
No fence/barrier widening or numerical-tolerance change was used. This
supports the coordinate-order attribution of F-104 in the measured case.
The full150-cell×50-process matrix subsequently passed7500/7500, with every
L0.5/L1/L2 hash matching and independent log/manifest verification. The
symbolic counter re-audit is still running; other ownership/variant shapes
remain untested by this matrix, so it is not full-domain synchronization proof.
Raw provenance and all100 independently audited logs are in
EVENT_COST/split_order_repair and split_order_matrix. Ten final-header builds
match the frozen device instructions and resource records10/10; this is
build-equivalence evidence, not another50-process claim.

## F-108 — Exact runtime work now changes the event price

✅ Round5 A9 imports BF16 CG, attaches A2's single-authority runtime QPs to
coupling_metrics, and CostModel consumes them with structured measured rates
(`CostModel.cpp:475`). Twelve cells give600/600 correctness processes and
1200 four-arm processes, warmup5/repeat11,25 paired rounds. κ0→1 event price
atseq128 rises6176.488129ns(gqa2) /12552.353211ns(mha4), matching the actual
wait-count increase. κ0 is aggregate, not positive coarsening's first point;
the user approved this semantic correction. κ1→2 decreases waits and price.
36/36 substitution prices are bit-identical; runtime fields1200/1200 match.
This is functional consumption, not carrier-only acceptance.

The longest-worker feature is present but both NNLS coefficients are0.
Poll leave-one-cell-out error still reaches95.06%; no positive epsilon or
hidden rescaling was applied. fence remainsnot_calibrated. L1 is preserved,
L2 DP transitions explicitly reject until implemented, and A6/B remain open.
EVENT_COST/round5_structured.md preserves values, uncertainty and limitations.

## F-109 — Repaired queue counts close the full sampled process matrix

✅ Ten exact symbolic S/P queries cover two models×five splits; their150
cells match15000/15000 counters from all7500 repaired GPU processes.
Interruption after query8 was recovered by adding only the two missing CPU
queries; neither raw batch was overwritten. The frozen tool, manifests,
zero-reference logs and independent process audit are underEVENT_COST.
This closes this sampled tile-ownership matrix, not every variant/domain.

## F-110 — Typed partial bandwidth resolves, fixed cost does not

✅ A12.2's float-partial/BF16-output width/peer microbenchmark ran after A9,
with no concurrent GPU timing. Small-width launch-subtracted measurements
were all−128ns; the existing fixed-cost resolution test failed. The new
profile was correctly rejected asnot_calibrated instead of silently using
the old BF16 coefficient or clamped zero. Positive L2/DRAM peer slopes are
diagnostic only; COST_MODEL/partial_combine.md records them and the raw log.
This locally stops A12.2 pending a resolvable measurement method, not A3/A6/A9.

## F-111 — Parameter binding is not task-coordinate binding

✅ Causal attention work exposed a real API documentation error:
QuasiPolynomial::Eval fixes isl parameters, not the remaining task tuple.
Position-independent GEMM work had hidden the distinction. Explicit
BindCoordinates now restricts a task point without changing historical Eval
semantics. Exact RoPE partner/frequency and causal K reads pass 648 production
BF16 cells plus 48 synthetic cells; set equivalence checks grouped-head
addresses as well as counts. Five error branches retain zero isl references.
Full 30/30 CTest and 4/4 generated-CUDA byte comparisons pass. See
COST_MODEL/element_work.md for source locations and preserved detours.
This does not close A6 or extend the evidence to split attention.

## F-112 — Paired graph timing resolves the partial-combine fixed term

✅ F-110's local stop was repaired without changing the measured kernel.
`lib/Target/GemmCalibration.cu:472` times rotated work/control graphs with
64 identical launches per timed interval. The resolution test applies to
the whole interval, not the divided per-launch number. The first graph
attempt incorrectly tested the latter and is retained as a measurement-unit
detour. Fifty fresh processes resolve the fixed term at median 61.750004535ns;
each also compares all 1048576 outputs bitwise against a sequential FP32 CPU
combine. Four coefficients are published only for the measured sm89 BF16
float-partial path; missing targets still report not_calibrated. No unrelated
FP32 calibration numbers changed. Full FP32 predictions/ranks remain byte
identical; the conditional historical BF16 rank comparison slightly worsens,
as recorded without threshold changes in COST_MODEL/partial_combine.md.

## F-113 — Physical interface pricing exposes non-adjacent DP factors

✅ `CostModel.cpp:655` consumes physical wait/volume instead of discarding
both endpoint shapes. At seq128, GEMM3 M32→GEMM6 M128 costs
19.230769396298506ns while the other three {32,128} endpoint combinations
cost zero. These residual edges are not adjacent in the GEMM list, so the
old chain/separable recurrence is not exact for the new price.
`CouplingInterfaceDP.cpp:14` retains live endpoints until their last factor,
with Residency still outside; reference frontier width is one. Two models'
16-assignment non-adjacent oracles match selected full-Evaluate price bits,
also with unified scalar task pricing enabled. Eight-candidate interface
spread is 615.3908806976317ns; per-op predicted percentage benefit decreases,
not increases, relative to the historical-price control. These are CPU
prices, not measured GPU gains. See COST_MODEL/interface_work/result.md.

## F-114 — A collective regression intercept cannot initialize SIMT work

✅ A6's scalar prototype initially reused the calibrated GEMM setup
intercept (-344.946ns for this BF16 profile). Small scalar task prices became
negative and a zero-initialized wave max masked them as zero. The bad output
is retained. `ScalarDataflow.h:47` now exposes actual TaskBody phases;
`CostModel.cpp:408` prices their memory depth, reduction/publication barriers
and schema arithmetic directly. There is no collective accumulator setup
phase to charge in these bodies. Negative/nonfinite scalar task prices are
rejected, not clamped. GEMM fitted expressions and arithmetic order are
unchanged. Fourteen legal CPU model/domain cases in both ownership modes
check independent index sets/counts; 13 real rejection paths have zero isl
reference deltas. Two out-of-export-domain attempts remain recorded instead
of widening the domains. See COST_MODEL/scalar_work/result.md.

## F-115 — Unified task pricing passes both GEMM entry gates

✅ A6 verifies 4308 configuration/model/dtype groups: direct `TaskCostNs`
and cached production `TaskStageNs` each pass 904680 bit comparisons.
FP32's two complete 1077-configuration rankings do not regress; all four
dtype/model solver plans are byte-identical to their controls. BF16's
historical measured subsets still have top-k failures: unchanged ranking is
not a new accuracy achievement. The unified path is now the default, with
the archived path explicitly selectable. A6/A9 release the B entry gate;
neither fusion nor symbolic DP is thereby complete. See
`COST_MODEL/stage_price_gate/result.md` and `unified_solver/result.md`.

## F-116 — Measured service interpolation slightly worsens BF16 ranking

✅ B3.1's affine service-time curve changes BF16 rho from .9038734673 to
.9036449015 (gqa2) and .8910704759 to .8910404180 (mha4), on the same
historical 770/462 measured subsets. FP32 predictions are byte-identical.
The negative gate is retained despite its small magnitude: CDF stays the
default; the curve remains an explicit experimental option, not a silent
substitution to enable design (a). `PARAMETRIC/cache_curve/result.md` records
every paired prediction and the interpolation/physical-clamp distinction.

## F-117 — More local edges need not mean more removable fences

✅ Counting complete producer fanout changes B2's interpretation: at seq=4
neither worker count improves the 47 fence-free producers without queue
growth, despite improved same-worker edge counts. At seq=128/workers=256,
affine_balanced_100 increases 35 to 541 with queue length still 22. Six
instances and 78 mappings pass the offline DAG check with explicit zero isl
references. This is not a GPU fence-elision claim; the uncalibrated fence
rebate remains zero. See `AFFINE_PROBE/fence_producers/result.md`.

## F-118 — Count-balanced production placement can still be much slower

✅ Two models x two sequence lengths x two states x 50 fresh BF16 processes:
400/400 correct, with state-rotated steady timing. Runtime mapping 4 has the
same maximum queue lengths and 212 registers / 2 CTAs per SM as mapping 0.
Its reduced waits match the independent CG projection in 200/200 records.
Yet the primary 25-pair L2 ratios are 1.639908, 3.087011, 1.396662, 4.002253.
The event model predicts small improvements, so the performance-sign gate
fails. Keep this negative result and the default mapping unchanged.

⚠️ Task-count balance is not work/critical-path balance; poll count does not
measure spin duration. These are candidate explanations, not a measured
decomposition of the slowdown. Code and full paired evidence:
`docs/experiments/PLACE/round5_balanced_result.md`. This does not block
independent Fusion or symbolic-pricing implementation.

⚠️ Read with F-126: measured under L1-identical ownership, stage-major queues and FIFO execution; not evidence about the value of placement, ordering or windows in general.

## F-119 — Chunk prices need sequential runtime phases and compiled resources

✅ Attention chunk plans now price scores, normalization, partial PV and
combination separately, with exact CG physical work and workspace. Sixteen
BF16 candidates agree bitwise with direct evaluation; four supplied-plan DP
minima all select chunk=1. Archived 800-process GPU receipts yield 3200 exact
resource/task/wait matches. The 25-pair rankings are not universally correct:
for gqa2 seq128/chunk8, predicted L2 is .545651 ms versus .938992 ms measured.
Normalization's tid0 serialization remains outside this throughput estimate.
Do not equate exact work counts with validated latency or arbitrary per-layer
chunk optimization. `COST_MODEL/attention_prices_batch/result.md` and
`attention_dp/result.md` record scope and all paired controls.

## F-120 — Mixed phase pricing is not an extra charge per logical edge

✅ `CostModel::TaskInstanceNs` now accepts physical memory-space traffic,
and `PriceFusionTasks` retains each phase's MMA/SIMT route. A counterfactual
4-producer/16-consumer partition really executes twelve extra producer
instances, reports 219645.236054 ns recompute work and increases global
traffic to 2170880 bytes. That work is not charged a second time after
assembling fused waves. All 4308 historical GEMM price groups and fourteen
scalar tables retain their bit patterns/bytes. Existing GEMM residual
epilogues are labeled existing fusion, not new runtime event savings.
See `FUSION/task_prices_streamed/result.md`; no new fused GPU claim.

## F-121 — A measured cache curve can make a CG interface price cubic

✅ Both BF16 models hit the exact degree guard on KVAppend->attention:
`repeated=512*S^2-4*S`, volume=1, multiplied by a nonconstant affine cache
service coefficient produces a nonzero cubic term. This is derived from
the real CG, not a synthetic high-degree rejection. The 5120 collective and
768 scalar task-price controls pass, but complete symbolic DP stops with
exit=2 and zero ISL references. Design (b) is not retired. Removing the CDF
eliminates transcendental expressions but does not bound polynomial degree
by two. Full coefficients and failed DP receipts:
`PARAMETRIC/task_prices/result.md`. No numerical search or sampling fallback
was used; B1 remains independent.

## F-122 — Fusion verification must check the replacement, not the old plan

✅ `FusionPass.cpp` applies an explicit logical pair transactionally and
creates a new consumer-indexed fused task space. The two-model rewrite
removes the internal edge and passes 1890/1890 A1 incidence matrix checks;
changed edges also pass symbolic conservation. Six rejection branches retain
zero ISL references. Codegen and ModelDescription reject the replacement
until mixed-body runtime projection exists, so an old model_plan cannot
silently masquerade as executed fusion. Interval selection and GPU bodies
are still unfinished. `FUSION/rewrite_complete/result.md` records the failed
proof attempt on untouched c3 (floor expressions not structurally zero),
the export-parameter alias correction and the final checks.

✅ Strengthening `CouplingOp::verify` also exposed an invalid old test fixture:
for `0<=j<=i<=3`, fanout(j) is `4-j`, not scalar 1. The corrected fixture
follows independent inverse-fiber counting; a dedicated false-fanout fixture
must now fail. This is not a change to numerical acceptance tolerance.

## F-123 — Logical fusion fanout is not runtime fusion fanout

✅ A2 ownership projection makes the tested RoPE/KV logical unit-fanout edge
runtime fanout 2: one RoPE CTA writes twice the elements of a KV CTA.
Fused runtime pricing must include that producer recomputation. Exact phase
composition and worker/event-image counting give a nonzero event-price delta
(-2873.8036648599809 ns for one gqa2/seq4 pair at kappa1). Four production CPU
cells enumerate 4/16 fusion patterns with bit-identical unfused baselines.
This is not a measured fused-kernel speedup.

✅ Genuine shared GEMM/add and GEMM/RMS bodies pass BF16 50/50 and FP32
50/50 fresh processes (24 cases each), with no global intermediate writes.
RMS consumes one row while recomputing the full producer tile, preserving
the fanout cost instead of retaining the unfused producer parallelism.
⚠️ Full-model lowering and GPU price/sign acceptance remain unfinished.
Details, resources, corrections and explicit scope are in
`FUSION/interval_runtime_progress.md`.

## F-124 — Physical tail traffic is not allocated fusion storage

✅ The genuine GEMM/add chain exposed a resource-model undercount: tile32x128,
seq4 uses only 4 live rows, but its compiled intermediate reserves all 32.
The original physical-R/W calculation predicted 16384 B shared; the body
allocates 23552 B. `DeriveFusionResources` now takes explicit allocation bytes
from the implementation while preserving predicated traffic accounting.
The full-pattern register API likewise accepts ptxas evidence: add separate
and fused L2 use 112 and 148 registers, not their old parent model's 212.
Tests reject an allocation smaller than the live intermediate.

✅ Production fusion lowering consumes replacement CG dependencies and verifies
them against exact runtime ownership, rather than reusing stage windows as
proof. The CG verifier runs before the fused lowering branch. Two GEMM chains
now pass this same path. GPU process matrices and prediction/sign acceptance
are separately recorded in `FUSION/interval_runtime_progress.md`.

## F-125 — Fusion correctness does not establish a useful fusion optimizer

✅ Actual generated RoPE/KV decoder fusion passes400/400 BF16 fresh processes.
Two generated GEMM calibration chains also pass400/400, with original numeric
tolerance and identical output bits across states and levels. CPU/GPU exact
task/wait/resource comparisons pass. GEMM/add improves at both seq4/128.

✅ GEMM/RMSNorm violates the price/sign gate: paired L2 ratios1.0625 and1.3521,
while the model predicts improvement. Full N, task counts and residency match.
At seq128,4 original producers become128 fused executions, but both fit one
modeled128-worker wave; predicted task time scarcely changes while global
traffic rises2.75→71.57MB. This exposes the recomputation/wave service envelope,
not an omitted fanout counter. The precise hardware bottleneck remains
unprofiled. No fitted coefficient or acceptance threshold was changed.

⚠️ Joint fusion search remains unaccepted and depends on this failed price
gate. Runtime/lowering are no longer missing. All comparisons and rejected
explanations are retained in `FUSION/runtime_result.md`.

## F-126 — Default L2 ownership is L1's grid-stride ownership, so L2 can at most recover the barrier

✅ Code inspection: `harness::Create()` sets `task_owner[stage][task] =
physical_worker[task % grid]`, and `HostPlacedBlock` is the identity for
`TILEMEGA_PLACEMENT` 0 and 4 (before the balanced override). L1 task bodies own
tasks through `for (task = PlacedBlock(); task < count; task += gridDim.x)`.
Every CTA therefore executes the same tasks in L1 and L2. Queues are built
worker → stage_order → owned tasks, so the first tasks of every small stage
land on the same low-index workers, whose queues serialize the stage chain.

⚠️ Inferred bound from the four-arm medians in
`docs/experiments/L2_ATTRIB/result.md` (unsafe probes, not valid kernels):
`neither − l1nosync` is the loop term only. With free synchronization, L2's
gain over L1 is at most (barrier + |loop|) / L1 = 13.6%, 9.4%, 12.1%, 9.1% for gqa2/4, gqa2/128, mha4/4 and mha4/128, while measured wait+notify is 2.2×, 2.5×, 2.1×, 2.9× the
barrier it replaces. The 1.081–1.113 L2/L1 ratio is structural before it is a
primitive-cost problem.

Consequence: F-82, F-86, F-118 and the OVERLAP early-start fraction measure this
executor structure; they are not general statements about ordering, windows,
κ or placement. Skeleton impact: §5.7.5. Code: `ModelHarness.cuh`
(`harness::Create`), `Placement.cuh`, `GemmStageTaskBody.h`.

## F-127 — The solver-to-codegen contract carries no task-level placement

✅ The frontend emits one `tilemega.placement` per task space with `map = [0]`
and `cluster = 1` (plus optional `resident_only` and `mapping_mode =
"balanced"`); no solver pass writes it. `BuildVariantSchedule` in
`lib/Codegen/Codegen.cpp` computes a stage permutation with `ListScheduler`
during code generation and emits `ScheduleStageDesc {stage, dependency_begin,
dependency_count}`. `BalanceTaskPlacement` returns `slot`, which no caller
reads; the host consumes only `worker` and rebuilds stage-major queues. The
skeleton's §2.6/§5.6 division — L2 decides placement, L1 lowers it — is
therefore not implemented. Skeleton impact: §2.3, §5.7.4, §8.11.

## F-128 — The chain DP optimizes the L1 objective; its L2 optimality is untested

✅ `ChainDP::Solve` prices each operator as stage time plus one barrier (plus
combine and another barrier when split) and throws when `l2_events` is
enabled. `CostModel::Evaluate` replaces barriers by `EventNs`, a sum of
calibrated rates times runtime counts, with no execution timing. Every solver
configuration, including `16x64x16s2k16`, is thus optimal for L1 execution.
⚠️ Hypothesis, not a finding: aggressive split-K buys per-stage parallelism for
the barrier model but multiplies task refs, events and combine hops under L2.
EX-S3 tests it. Skeleton impact: §4.4.2.

## F-129 — Balanced placement is affinity-first and can collapse a stage onto its producers' workers

✅ `BalanceTaskPlacement` picks the worker holding the most producers of a task;
a shorter queue only breaks ties, and the cap is the baseline maximum queue
length. When all tasks of a stage depend on the same few producers (at seq=4,
every QKV GEMM task depends on the four RMSNorm tasks), the stage fills those
workers up to the cap before any other worker is used. ⚠️ Attributing F-118's
1.40–4.00× slowdown to this mechanism is inferred; EX-D1 traces must confirm
it. A capped task count is neither work balance nor critical-path balance.

## F-130 — Queue-order wait lifting and owner elision are sound only for FIFO execution

✅ The host drops a task's wait when an earlier task in the same worker queue
already waited for the same event (`seen[worker]`), and drops the poll when a
κ=1 producer is on the same worker. Both are valid because each worker executes
strictly in slot order. ⚠️ Inferred: an executor that may run a later ready slot
first would skip required waits. With a window W, lifting or elision is sound
only from slots i ≤ j − W; nearer producers must become local dependencies.
EX-E2's negative control must demonstrate the failure. Skeleton impact: §5.7.3
L-d, §8.10.

## F-131 — `%globaltimer` is a device-wide 1024 ns ruler; `clock64` is fine but per-SM

✅ Measured before any trace v2 number was interpreted
(`docs/experiments/TRACE_V2/resolution.md`, RTX 4090, driver 610.43.02).
`%globaltimer`'s adjacent-delta minimum, median, p99 and maximum are all
**1024 ns**, 98.2949% of back-to-back reads return an unchanged value, and no
read ever went backwards. It carries no per-SM offset: 128 CTAs on 128 distinct
SMs, released from one software barrier, read a bit-identical value (spread
0 ns). `clock64` is the mirror image — 40 cycles (≈ 16 ns) effective resolution,
but at one instant its readings differ across SMs by 3.54 × 10⁹ cycles (≈ 1.41 s)
and its rate follows DVFS. Fitting each SM against the shared globaltimer ruler
over 256 anchors recovers rates agreeing to 0.006% and leaves a per-SM offset
uncertainty of 18.99 ns (1σ); the worst-SM fit residual, 303.8 ns RMS, matches
the 295.6 ns predicted for uniform quantization over a 1024 ns tick, so the
residual *is* the quantization and no drift term is detectable on top of it.
Consequence: §3.5's conditional branch applies, `TaskTraceV2` carries
`run_begin_clk`/`run_end_clk`, and a per-hop figure of one or two ticks reports
the clock, not the hop. Quantization by a floor is monotone, so a negative hop
would still be a real ordering violation rather than a clock artifact.

## F-132 — Trace v2 perturbs L2 by 1.7% and reconstructs it to 2.1–3.4%, but only once the publish term is charged to the producer

✅ Paired 25 rounds in one session with arm rotation, one fresh process per
round (`TRACE_V2/raw/perturbation.txt`): `l2_ms` median ratio on/off is 1.0157,
1.0167, 1.0163, 1.0149 for gqa2/4, gqa2/128, mha4/4, mha4/128 — worst 1.0167
against the 1.02 gate. Correctness with the instrumented build is 50/50 fresh
processes in all four cells, and with `TILEMEGA_TRACE_V2` at its default 0 the
SASS of both reference kernels is byte-identical to the baseline
(`TRACE_V2/sass_identity/`, both diffs empty, 6802420 bytes each).

✅ Gate D1-d **fails as defined**: a critical path whose node weight is the
measured `run_end − run_begin` and whose cross-worker edge weight is the
measured hop p50 reconstructs only 87.6%, 85.5%, 87.9% and 86.7% of measured
`l2_ms` (errors 12.42%, 14.51%, 12.10%, 13.27% against a 5% threshold). The
definition was not relaxed; the residual was measured instead. Decomposing the
reconstructed chain into its measured parts (task, wait, pre-run barrier,
publish, gap) sums exactly to the chain span, and the single term the §3.6
node weight omits is the producer's own notify cost `publish_end − run_end`,
which is 43–106 µs per chain. Restoring only that term moves the error to
3.10%, 3.17%, 2.08% and 2.14%. ⚠️ Inferred: §3.6's node definition, not the
reconstruction method, is what misses 5%; a chain-level critical path must
charge publish to the producer because the consumer cannot start until the
publish lands.

## F-133 — Per-hop latency is at or below the clock tick; synchronization is not what L2 is spending its time on

✅ Over every traced slot with a non-empty wait set, `hop(j) = ready(j) −
max publish(producers)` has p50 = 1024 ns in all four cells — exactly one
globaltimer tick, i.e. at or under the measurement floor (F-131). p90 is 1024,
2048, 1024 and 7168 ns; the maximum is 46–54 µs. The count of `hop(j) < 0` is
**0** in all four cells (gate D1-e). The corresponding bounds agree:
`cp_lb_sync − cp_lb_nosync`, the entire cost of charging every cross-worker edge
its measured hop p50 along the critical path, is 10.24 µs in every cell — 0.8%
to 2.3% of measured `l2_ms`. Consequence: F-126's structural bound is not
merely an upper bound on what removing synchronization could buy, it is close
to tight on the critical path itself. G4's per-task publish protocol is
expensive per *task* (F-132: 43–106 µs of publish along one chain) without the
*hop* it produces being long.

## F-134 — L2's time is ownership concentration, not synchronization: the busiest worker's own queue is 81–85% of the kernel

✅ From the trace v2 dumps (`TRACE_V2/analysis.md`): `queue_lb`, the sum of
measured run durations on the single busiest worker, is 0.3666, 0.5304, 0.7485
and 1.0752 ms against measured `l2_ms` of 0.4516, 0.6267, 0.9020 and 1.2780 ms
— 81.2%, 84.6%, 83.0% and 84.1%. The work bound `Σ run / grid` is 1.5–17.8% of
measured, and the critical-path bounds are 22–32%. So no bound except the
per-worker queue is anywhere near the measurement, and the one that is comes
from how tasks were handed out.

✅ The mechanism is visible directly. At seq=4 only **16 of 256** workers
receive any task at all, because `task_owner[stage][task] = physical_worker[task
% grid]` restarts at worker 0 in every stage and the reference stages have 2–8
active tasks. Per-worker idle fraction is 74.9% and 72.2% at seq=4 and 82.5%
and 82.1% at seq=128. Head-of-line blocking that is provably reclaimable — the
worker is stalled on its queue head while a later task in its own queue already
has all producers published — totals 18.85%, 52.99%, 56.80% and 63.98% of all
stall time, and at gqa2/128 and mha4/128 every one of the 256 workers
contributes some.

✅ The real-width arm makes it worse, not better: a 4-layer, hidden 4096,
intermediate 14336, 32/8-head model traced at seq ∈ {4, 128}
(`PLACE_ROTATE/raw/realwidth/`) has `queue_lb` at 6.0436 of 6.2312 ms and
7.9534 of 8.5492 ms — **97.0% and 93.0%** of measured. So the concentration is
not an artifact of a 2-layer reference model, and G12's extrapolation risk does
not apply to this particular finding. This confirms the ⚠️ attribution F-129
left open and gives G2 and G10 measured numbers. Skeleton impact: §5.7.4,
§5.7.5.

## F-135 — Cross-stage continuous round robin removes the concentration and is 25–35% faster on the correct kernel

✅ `TILEMEGA_PLACEMENT=5` starts stage *s*'s round-robin where the previous
stage in execution order stopped: `base[s] = (Σ_{i < pos(s)} active_tasks(
stage_order[i])) mod grid`, leaving the CTA-to-SM map, the stage-major queue
order, the window and the whole synchronization protocol untouched. Host-side
statistics over the exact runtime task DAG (`PLACE_ROTATE/raw/place_stats.txt`):
maximum queue length falls 30 → 1, 34 → 18, 60 → 2 and 80 → 47, while
cross-worker edges rise by 0.005% to 17% (1108 → 1292, 545140 → 545252,
3604 → 4108, 2149540 → 2149788). Locality was never there to lose: the
cross-worker edge fraction was already 95.1–99.6% under mode 0.

✅ Correctness is 50/50 fresh processes for both modes, both models, seq ∈ {4,
128}. Online, 25 paired rounds per cell with the arm × placement combination
rotated by `(round + slot) % 8`, one fresh process per round: the `full` arm —
the real kernel, `RESULT status=PASS` — gives mode-5/mode-0 `l2_ms` median
ratios of 0.6582, 0.7421, 0.6521 and 0.7461, pooled **0.6705**, bootstrap 95% CI
[0.6598, 0.7392], Wilcoxon p = 4.0e-18 over 100 pairs. With synchronization
removed (`neither`, an unsafe probe and not a valid kernel) the ratio is 0.2240
[0.1891, 0.2791]. ⚠️ Inferred from the gap between those two: rotation exposes
parallelism that the current per-task publish protocol then partly re-serializes,
which is the EX-E3 question, not a placement question.

## F-136 — Rescheduling headroom is 2.3–5.8×, and it is reachable only through a Plan the solver owns

✅ An earliest-finish-time list schedule over the exact runtime task DAG, using
measured task durations, the measured hop p50 on every cross-worker edge, zero
on same-worker edges, and free placement over the resident workers
(`PLACE_ROTATE/headroom.md`) reaches 0.1526, 0.2673, 0.1546 and 0.3533 ms
against measured 0.4516, 0.6267, 0.9020 and 1.2780 ms — 0.34, 0.43, 0.17 and
0.28 of measured. On gqa2/4 it uses 118 workers where the shipped placement
uses 16. The mandatory real-width cells agree: 1.5985 ms against 6.2312 ms
(0.26) at seq=4 and 3.3065 ms against 8.5492 ms (0.39) at seq=128. This is an
optimistic bound: it says what rescheduling alone could buy if the executor
could run any ready task, and it says nothing about legality under today's
strictly FIFO queue (G3, F-130).

stated, worth recording because it cost a run: the real-width cells first came
back FAIL because `build-phase12/tools/tilemega-compile` predates the schedule
table in `RuntimeVariantDesc` and emits an initializer that no longer compiles.
The generator, not the model, was stale; `realwidth.sh` now defaults to the
current-tree build.

✅ The fork rule fixed before measurement (`PLACE_ROTATE/fork.py`, §4.4)
adjudicates from the pooled online medians:

```
FORK rule=2 r_neither=0.2240 ci=[0.1891,0.2791] r_full=0.6705 ci=[0.6598,0.7392] cells=4
```

Rule 2 — placement gain is large and real, and it does *not* survive intact
into the correct kernel — routes the next round to **EX-E1** (a Plan contract
the solver writes and the host materializes) and **EX-S2** (EFT placement and
ordering). Consequence for G1: mode 5 is a hard-coded host heuristic behind a
compile macro, exactly the shape F-127 says the contract cannot express; the
measured 0.67 is therefore a lower bound on what an expressible plan is worth,
not an implementation to keep.

## F-137 — On sm_120 `%globaltimer` is a 32 ns ruler with a 160 ns cross-SM spread; F-131's tick is an sm_89 number

Provenance for F-137 to F-142: the Blackwell run was executed on the sm_120
machine on 2026-09-12 (RTX 5090, compute capability 12.0, driver 580.105.08,
CUDA 12.8, `build-cluster`) from source commit `886f0dc9`; the numbers below are
read here from its committed artifacts, not re-executed on this machine. The
run's own report is `docs/experiments/sm120_round_one_20260912.md`. It wrote
into the sm_89 `raw/` trees in place, so its artifacts now live under
`<experiment>/raw_sm120/` and `<experiment>/raw_sm120/insitu/` and the sm_89
files were restored from `886f0dc9`; `verify.py` re-run afterwards reproduces
the committed 24-of-25 output byte for byte apart from its temp directory name.
Absolute latency is not comparable across the two machines — different device,
different session — so every cross-architecture statement below is a comparison
of ratios or of fractions of the same kernel.

✅ The resolution probe (`TRACE_V2/raw_sm120/insitu/resolution.tsv`) measures
`%globaltimer`'s adjacent-delta minimum, median, p99 and maximum all at
**32 ns**, a 32× finer ruler than sm_89's 1024 ns. Only 53.4327% of
back-to-back reads return an unchanged value against 98.2949% on sm_89, and no
read goes backwards. `clock64` is 42 cycles minimum and p50, 64 at p99.
`needs_clock64_columns` is 0, so §3.5's conditional branch resolves the same way
as on sm_89 and `%globaltimer` alone carries the trace.

⚠️ The device-wide broadcast is *not* exact here. 128 CTAs on 128 distinct SMs,
released from one software barrier, read values spanning **160 ns** where sm_89
spanned 0 ns. A cross-SM hop on sm_120 therefore carries up to ~160 ns of offset
error; that is 31% of the measured 512 ns hop (F-138) and is why a single-tick
hop figure would still not be trustworthy even at this resolution.

stated, recorded because it would otherwise look like a gap in the evidence: the
trace metadata's `globaltimer_resolution_ns` field came back empty in this run.
The 32 ns tick rests on `resolution.tsv` and the probe log, not on that field.

## F-138 — On sm_120 the 512 ns hop is a real measurement, not a tick floor, and the whole synchronization term is ≈5 µs

✅ At a 32 ns ruler the per-hop p50 is **512, 512, 512 and 480 ns** across
gqa2/128, gqa2/4, mha4/128 and mha4/4 — 15 to 16 ticks, not one — over 10,528
hop samples with **zero negative hops**
(`TRACE_V2/raw_sm120/analysis.tsv`). F-133's sm_89 hop sat at or below its
1024 ns tick and was therefore a floor; the same conclusion now rests on a
measurement that the clock can actually resolve.

✅ The conclusion strengthens rather than changes. Charging every cross-worker
edge on the critical path at the measured hop instead of zero moves the
reconstruction lower bound by `cp_lb_sync − cp_lb_nosync` = **5.12, 5.12, 5.12
and 4.80 µs**, against measured `l2_ms` of 0.5675, 0.3700, 1.1621 and 0.7477 ms
— under 1% of the kernel in every cell. On sm_89 the same term was 10.24 µs.
Synchronization is not what L2 spends its time on, on either architecture.

## F-139 — Ownership concentration reproduces on Blackwell: the busiest worker's own queue is 77–84% of the kernel

✅ `queue_lb_ms / measured_l2_ms` is **0.7746, 0.8386, 0.7741 and 0.8236**
(gqa2/4, gqa2/128, mha4/4, mha4/128) against F-134's 81–85% on sm_89. The
shape is identical: the seq=4 cells occupy **16 of 340** resident workers, the
seq=128 cells occupy all 340, 95.07–99.72% of DAG edges cross workers, and
head-of-line blocking accounts for 19.13%, 40.28%, 56.04% and 67.47% of the
kernel span. Round one's diagnosis is not an Ada artifact.

## F-140 — Mode 5 and the fork rule land in the same place on Blackwell

✅ Cross-stage continuous round robin on the correct (full) arm gives
placement-5/placement-0 ratios **0.6514, 0.7403, 0.6313 and 0.7491**
(`PLACE_ROTATE/raw_sm120/summary.tsv`, 25 rounds per cell, Wilcoxon
p = 1.307e-05 in all four), against 0.63–0.75 on sm_89. Correctness is
400/400 across eight cells of 50 fresh processes each, zero failures or hangs.
The mechanism is visible in the placement statistics
(`PLACE_ROTATE/raw_sm120/insitu/raw/place_stats.txt`): max queue depth falls
from 30/34/60/80 to **1/13/2/36** while cross-worker edges grow only slightly
(1108→1292, 545476→547472, 3604→4108, 2150884→2155964) — the trade F-135
describes, on a second architecture.

✅ The fork rule fixed before measurement adjudicates from this run's own
pooled medians (`PLACE_ROTATE/raw_sm120/fork.txt`):

```
FORK rule=2 r_neither=0.1687 ci=[0.1302,0.2039] r_full=0.6985 ci=[0.6527,0.7401] cells=4
```

Rule 2 again, as on sm_89 (0.2240 / 0.6705). The routing to **EX-E1** and
**EX-S2** is therefore confirmed on both architectures rather than resting on
one machine.

## F-141 — The critical-path reconstruction degrades on sm_120, and charging publish to the producer no longer closes it

✅ `cp_error_vs_l2_ms` is **19.59%, 14.33%, 20.07% and 15.51%** (gqa2/4,
gqa2/128, mha4/4, mha4/128) against 12.10–14.51% on sm_89. D1-d's 5% limit,
already failed on sm_89 and recorded rather than repaired (H7), fails wider
here.

⚠️ F-132's repair does not carry over. `cp_with_publish_error` is **6.51%,
4.74%, 6.23% and 4.73%** — two of four cells still above 5%, where the same
secondary account closed sm_89 to 2.1–3.4%. The publish term is not an
architecture-independent explanation of the residual.

inferred, and left open rather than concluded: `cp_split_gap_ns` is negative in
every cell (−1152, −222944, −82528, −487968 ns), so the split terms sum past
the chain span they decompose. That is consistent with double counting between
the wait and publish terms, but nothing in this run isolates it. Whatever the
reconstruction is missing, F-138 and F-139 do not depend on it: both are
measured directly rather than through the reconstruction.

## F-142 — The sm_120 real-width arm covers seq=4 only, and its headroom table is not a clean sm_120 result

✅ The 4-layer, hidden-4096 real-width seq=4 cell completed with status PASS.
The seq=128 cell is FAIL: it died during PyTorch export with
`RuntimeError: basic_ios::clear: iostream error`, and the filesystem was 100%
full with zero bytes available after the run. Disk exhaustion is the inferred
cause; the run did not isolate it further. The PLACE_ROTATE wrapper tolerates
real-width failure, so its overall PASS must not be read as real-width
acceptance on Blackwell.

⚠️ F-136's headroom numbers are **not** reproduced on sm_120. The run's
`headroom.tsv` carries older reference and real-width rows plus append
duplicates, and its `HEADROOM cell=real_s128` line was computed from a dump
whose metadata predates the session (19:37:42 against a 20:07:52 start). It is
retained as raw provenance only. The 2.3–5.8× rescheduling headroom stands on
the sm_89 evidence alone until a Blackwell run has the disk to finish.

## F-143 — The Plan contract reproduces the pre-plan host byte for byte on modes 0, 4 and 5

✅ The solver now emits `(π, σ)` and the host materializes each worker's queue
by σ instead of walking stages; the generated source carries it as
`RuntimePlanDesc{mode, params, param_count, window, policy}` on
`RuntimeVariantDesc`, and `BuildVariantStageSchedule` moved out of
`lib/Codegen/` into `lib/Solver/`. With a `legacy_grid_stride` plan at `W = 1`
the two reference models' generated `.cu` are byte-identical to the round-2
baseline `6c359e2b`: sha256 `017a39b9…` for gqa2 and `1be74406…` for mha4, on
both sides (`PLAN_CONTRACT/legacy_identity/`). The emitter is not merely silent
— the same source generated with `balanced_placement` on changes exactly one
line per model, the variant initializer, which is the positive control that an
empty diff is evidence.

✅ Host materialization is identical too. 12 cells (2 models × placements
{0, 4, 5} × seq {4, 128}) dump `schedule.tsv`, `waits.tsv` and `events.tsv`
before any device work; 36/36 files are byte-identical between a baseline host
rebuilt from `6c359e2b:ModelHarness.cuh` and the working tree, and 24/24 runs
report `RESULT status=PASS` (`PLAN_CONTRACT/mode_identity/`). The dump block is
the same source text in both binaries, so the identity is about the placement
and not about the dumper. `E2E_PLACE_STATS` gains one field, `plan=`, and no
existing field changes; the refreshed `.out` files differ from the previous
commit only in `E2E_TIME` and `E2E_ITER`'s `l2_iter1_ms`, with every correctness
field and `sha256.txt` unchanged.

✅ H4 holds: `grep -n "ListScheduler\|BalanceTaskPlacement\|BuildVariantSchedule\|Schedule("
lib/Codegen/Codegen.cpp` leaves one hit, and it is a consumption call —
`solver::BuildVariantStageSchedule(variant.dependencies, stages.size())`
(`PLAN_CONTRACT/h4_grep.txt`; the recorded line number moved 384 → 403 when
EX-S2 grew the file above it, and `PLACE_EFT/verify.py` recomputes the grep live
rather than trusting the number).

## F-144 — An arbitrary σ is obeyed, and the poll-elision condition had to be stated as a hard check

✅ σ is the only thing that orders a worker's queue: `BuildPlanQueues` fills
`plan.queue` from `owner` and `slot` alone and rejects any σ that is not a dense
permutation of `[0, n)` on a worker, because anything else is not a total order
and has no executable queue (§5.7.2). Four unit tests pin this
(`test/unit/plan_contract_test.cpp`, `plan_table_test.cpp`,
`eft_placement_test.cpp`, `execution_simulator_test.cpp`): a σ that interleaves
stages on one worker is materialized in exactly that order; a σ containing a
cycle over the union of task edges and same-worker window edges is rejected; a grid
larger than the resident limit is rejected (`ResidentScheduleLegal`, L-b) as is
an owner outside the grid; and every rejection is a hard failure with a message,
never a warning (H5).

✅ The negative control is the one that changed the implementation. Before a
consumer polls a producer's event, the host elides the poll when the producer
runs earlier on the same worker. Stated that way the condition is wrong: a
same-worker producer with `σ(producer) > σ(consumer)` would have its poll
silently elided and the consumer would read an unpublished row. `CheckPlanLegality`
therefore checks L-c *first* and hard-fails such a Plan
(`lib/Solver/PlanMaterialize.cpp`), and the test constructs exactly that Plan and
asserts the rejection rather than asserting the elision.

✅ Correctness under the new materialization path: the SEQSCAN subset
seq ∈ {4, 128, 2048} × past ∈ {0, 512} on both reference models is 50/50 fresh
processes in all 12 cells, recounted from the per-process logs, and the full
CTest suite passes (`PLAN_CONTRACT/seqscan/`, `PLAN_CONTRACT/ctest.txt`).

## F-145 — Polling contention is not the missing factor: the hop is 1235 ns and flat over 256× line sharing

✅ `hop_ns(N, R) = c0 + c1·log2(1 + N/R) + c2·log2(R)`, fitted by weighted least
squares over 24 cells (`N ∈ {1…256}` × `R ∈ {1, 4, 16, 64}`, `N ≥ R`), 4096
rounds each, on the primitives the runtime executes — `EventPoll` and `atomicExch`
on a 128-byte-padded `EventCounter` row (`SIMULATOR/contention.cu`). In the arm
the generated kernel actually runs (RMW poll with `__nanosleep(64)` backoff):

```
c0 = 1235.4 ± 17.2 ns   c1 = −0.40 ± 2.41   c2 = −2.67 ± 2.36
```

Both contention coefficients are **zero inside one standard error**, and across
a 256-fold range of line sharing and a 64-fold range of live rows the hop stays
between 1211 and 1249 ns — a 3.1 % spread. R2 §0 item four hypothesised that
round one's 1.7–3.7× underestimate of mode 5 was polling contention; this
measurement does **not support** it. Recorded as a negative result; no gate was
moved.

✅ 910 ns of the 1235 ns hop is the `__nanosleep(64)` backoff granularity
(1235.4 − 325.0 with the backoff removed), not coherence traffic. With the
backoff out, contention becomes measurable and stays small: 4.04 ± 2.62 ns per
doubling of sharing for one consumer, 12.2 ± 4.6 ns for the slowest of the
consumers on one row — about 100 ns over the whole 256× range. That is a
synchronization-protocol observation and R2 §1 excludes protocol changes this
round, so it is carried to the next-round priority discussion rather than acted
on.

✅ `inversions = 0` in all 96 cells across both arms and both poll modes: no
observe timestamp precedes its publish. Two measurement artifacts were found by
measurement and are recorded rather than papered over — a fixed guard spin made
the sweep return exactly 1024.0 ns with `se = 0.00` in all 24 cells (the round
loop was periodic; fixed with a per-round phase dither), and one process of the
load-poll arm recorded a single ~2.3 ms device stall per cell, which is why the
fit uses a 0.1 %-trimmed mean with nothing dropped from the table
(`SIMULATOR/hop_fit.txt`).

✅ **The contention half of this reproduces on sm_120; the constant does not.**
The same sweep on an RTX 5090 (2026-09-13) fits `448.277066 + 1.134462·log2(1 +
N/R) − 0.436421·log2(R)`. In the column this finding draws on (`hop_trim_mean`,
`__nanosleep(64)` backoff) `c1 = +1.13 ± 1.96` is zero inside one standard error
just as sm_89's `−0.40 ± 2.41` was, and with the backoff removed the slowest
consumer on a row gives `+8.55 ± 1.85` against sm_89's `12.2 ± 4.6` — the same
order. R2 §0 item four is a clean negative on both architectures.

⚠️ What does differ is the hop's size and its composition: 448 ns rather than
1235 ns, of which the `__nanosleep(64)` backoff is about 32 ns (448.3 − 416.2,
both from `SIMULATOR/raw_sm120/hop_fit.txt`) rather than 910 ns. The backoff granularity that this finding identifies as the
optimizable part is therefore an sm_89-sized lever, not a universal one — on
sm_120 it is a twentieth of the size. Recorded as measured on each part; the
sm_89 numbers above are unchanged. One column cannot be compared: the backoff-64
`last_*` slope is `+18.08 ± 2.02` on sm_120 and sm_89's equivalent was never
quoted here. See the sm_120 follow-up section below and
[the session report](experiments/sm120_simulator_place_eft_20260913.md).

## F-146 — The execution simulator ranks placements well and predicts runtimes badly, in one direction

✅ `lib/Solver/ExecutionSimulator.cpp` replays §5.7.2: each worker walks
`plan.queue[w]` in σ order one task at a time, and a task starts at
`max(worker free, every predecessor's end + hop)` with the hop zero on a
same-worker edge. That last clause is the whole point — round one priced a
schedule as `max(work_lb, queue_lb, critical_path)`, and none of those three
bounds can express a worker idling at a not-yet-ready queue head while a task it
could have run waits behind it (F-139).

✅ S1-a, reported without a gate: per-task start-time error over 18 cells
(2 models × seq {4, 128, 512} × placements {0, 4, 5}), as a fraction of each
cell's own span, is |p50| 11.6–40.5 %, |p90| 19.7–65.2 %, |max| 23.0–81.5 %.
**The sign is negative in 18/18 cells**: the simulator predicts every task
starting earlier than it measurably did. A one-sided bias of that shape cancels
in an ordering and does not cancel in an absolute makespan, so the simulator's
output must be read as a ranking and never as a predicted runtime.

✅ S1-b, the hard gate, **PASS**: pooled Spearman **0.880** over the
placement × config scan, within-config 0.973, argmin correct in 6/6 configs, and
the predicted top-3 contains the measured top 3 %. The regime that mattered is
separated — `rotate` is predicted fastest in all six configs and measured
fastest in all six, and the predicted mode-5/mode-0 ratio (0.654–0.813) has the
same sign as the measured one (0.652–0.924). A model that could not tell modes 0
and 5 apart would have been unusable, and this one can.

⚠️ Two modelling choices are arms rather than assumptions, and the reported
numbers use the weaker one. `proportional_sharing` stretches a co-resident set by
its size; the nine-lane §4.4.1 resource model stretches it by aggregate demand.
**The whole-model numbers run the proportional arm** because only GEMM stages
expose a `ResourceVector`, so a mixed zero/non-zero lane set would make
`LaneStretch` return 1 and systematically under-estimate. The lane model is
unit-tested and carried; it is not what the numbers above use.

## F-147 — The simulator is two orders of magnitude over its evaluation budget, which is a §9 stop condition

✅ S1-c **FAILS**, and it fails by more than one order of magnitude, which is R2
§9's fourth stop condition ("the algorithm was chosen wrong — report first").
One `SimulateExecution` call, wall time measured around that call alone
(`predicted.tsv` column `eval_us`):

| cell | worst candidate | budget | over |
| --- | --- | --- | --- |
| gqa2 seq 4 | 61.4 µs | 1 ms | 0.06× |
| mha4 seq 4 | 128.0 µs | 1 ms | 0.13× |
| gqa2 seq 128 | 3.32 ms | 1 ms | 3.3× |
| mha4 seq 128 | 11.58 ms | 1 ms | 11.6× |
| gqa2 seq 512 | 38.58 ms | 1 ms | 38.6× |
| mha4 seq 512 | 276.79 ms | 1 ms | **276.8×** |
| real width seq 4 | 0.83 ms | 10 ms | 0.08× |
| real width seq 128 | 99.80 ms | 10 ms | **10.0×** |

Both budgets hold at seq 4 and break as the node count grows (200 nodes at gqa2
seq 4, 47 680 at mha4 seq 512), so the cost is the graph size and not a fixed
overhead. The gate was fixed before implementation and is **not moved** (H7);
the round continued to EX-S2 because §9 item four says "report first" where item
five says "stop immediately", and EX-S2 needs the simulator only to rank six
candidates per cell, which it does inside a second.

stated, and corrected here rather than deleted: an earlier draft of
`SIMULATOR/README.md` recorded S1-c as failing "at 2.9×, not 29×, so it is not
the §9 stop condition". That was written when the evaluation set stopped at
seq 128 for the reference models; adding seq 512 and real width made it false.
`eval_us` is host wall time and so is the one column not reproducible to the
digit — a previous run of the same driver gave 34.5 ms and 157.8 ms at seq 512
and 97.8 ms at real width seq 128, over budget by the same order of magnitude.

## F-148 — EFT and mode 5 trade the same quantity in opposite directions and cancel to within 3%

✅ The four-arm decomposition (25 paired rounds per combination, the `arm ×
probe` pair rotated by `(round + slot) % 8`, one fresh process per round) says
where each plan's L2 time goes. `neither` compiles out both the wait and the
notify — an unsafe probe, not a valid kernel — so it is the placement's
throughput bound; `full` is the real kernel:

| cell | mode 5 `neither` | mode 5 `full` | eft `neither` | eft `full` |
| --- | --- | --- | --- | --- |
| gqa2 s4 | 0.0584 ms | 0.2917 ms | 0.2376 ms | 0.2929 ms |
| mha4 s4 | 0.1355 ms | 0.5794 ms | 0.4700 ms | 0.5880 ms |
| gqa2 s128 | 0.1577 ms | 0.4557 ms | 0.3807 ms | 0.4567 ms |
| mha4 s128 | 0.2661 ms | 0.9114 ms | 0.6975 ms | 0.8787 ms |

Mode 5's placement is **4.1×, 3.5×, 2.4× and 2.6× better in the
no-synchronization limit**, and EFT's synchronization cost (`full − neither`) is
**0.055 / 0.118 / 0.076 / 0.181 ms against mode 5's 0.233 / 0.444 / 0.298 /
0.645 ms — 4.2×, 3.8×, 3.9× and 3.6× cheaper**. The two effects cancel to within
3%. That is the answer to "where is it stuck": under `W = 1` the total is a
throughput bound plus a synchronization term, EFT buys the second by spending
the first, and at this hop price the exchange rate is almost exactly one.

✅ The mechanism is visible in the host's own placement statistics
(`raw/place_stats.txt`), which EFT moves in the direction its cost model asks
for: same-worker edges 184 → 100, 504 → 280, 3736 → 3216 and 12664 → 11524
against `legacy_grid_stride`, longest queue 30 → 20, 60 → 40, 34 → 28, 80 → 74.
Mode 5 goes the other way — **zero** same-worker edges at seq 4 and a longest
queue of 1 and 2 — and wins the throughput bound by exactly that. The busiest
worker's own queue (`raw/predicted.tsv` `busiest_worker_ns`) as a fraction of the
measured `l2_ms` is **0.103 / 0.104 / 0.110 / 0.109 for mode 5 and 0.608 / 0.613 /
0.423 / 0.448 for EFT**: EFT drives the makespan down onto the busiest worker's
own work, which is what a good schedule is supposed to do, and it still does not
win.

⚠️ inferred: EFT priced a cross-worker hop at 1235 ns (F-145), 0.07–0.8 of a
single task's duration in these cells, so avoiding a hop looked worth
co-locating for. The measurement says the marginal cost of a hop inside the real
kernel is lower than that — the `neither`/`full` gap is dominated by per-task
publish, not by hop latency (F-133) — so the greedy over-bought locality. That is
a cost-model calibration question rather than a scheduling-algorithm one, and it
is the first thing to re-examine if EX-S2 is revisited after EX-E3.

## F-149 — The solver's plan loses to mode 5 by 0.2–4.9%: the research gate is a clean negative

✅ S2-b **FAILS**. 25 paired rounds per cell, six arms rotating by
`(round + slot) % 6`, one fresh process per round, ratios formed inside a round
so no median crosses a session boundary. EFT `l2_ms` against mode 5, with the
two other denominators from the same rounds:

| cell | eft / mode 5 | 95% CI | p | eft / mode 0 | eft / L1 |
| --- | --- | --- | --- | --- | --- |
| gqa2 s4 | **1.0096** | [1.0070, 1.0137] | 2.2e-05 | 0.6636 | 0.7320 |
| mha4 s4 | **1.0141** | [1.0106, 1.0212] | 1.3e-05 | 0.6624 | 0.7276 |
| gqa2 s128 | **1.0022** | [1.0017, 1.0026] | 8.4e-05 | 0.7451 | 0.8091 |
| mha4 s128 | **1.0273** | [1.0263, 1.0275] | 1.3e-05 | 0.7679 | 0.8537 |
| pooled (100 pairs) | **1.0111** | [1.0086, 1.0147] | 4.8e-17 | — | — |
| real s4 (S2-c) | **1.0273** | [1.0269, 1.0281] | 2.3e-04 | 0.7888 | 0.8419 |
| real s128 (S2-c) | **1.0487** | [1.0483, 1.0493] | 3.0e-05 | 0.8768 | 0.9711 |

0/4 reference cells pass, and neither real-width cell does: every CI lies
entirely **above** 1. The gate R2 §6.3 fixed before implementation was "faster
than mode 5"; it is **not** moved back to "faster than L1" (H7), even though EFT
passes that older gate at 0.73–0.85 on the reference cells. Correctness is not
the issue: S2-a is 50/50 fresh processes in all 34 arm-cells and the SEQSCAN
subset is 50/50 in all 12.

✅ The other candidates place the result. `wavefront` — a closed form that owns no
cost model at all — is statistically indistinguishable from EFT on the reference
cells (pooled 1.0105 [1.0050, 1.0149]) and is the one arm that touches mode 5 at
real width seq 128: median **0.9987**, CI [0.9982, 0.9994], but Wilcoxon
p = 0.063, so it is reported as a tie and not as a win, and it is in any case not
the plan the solver chose. `band` is byte-identical to `legacy_grid_stride` in
every cell (same `max_queue`, `same_worker_edges` and `cross_worker_edges`), so
its 1.49 is mode 0's number under another name and it is recorded as a degenerate
candidate rather than a result. `balanced` is the slowest arm at 2.21 pooled and
4.28 at mha4 s128, reproducing F-118's direction.

✅ S2-e, the simulator's predictive power on this round's real candidate set:
per-cell Spearman **+0.824 / +0.794 / +0.812 / +0.928**. The ordering is good and
not perfect — the simulator put `eft` first in three of the four cells `rotate`
actually won, and `wavefront` first in the fourth, predicting a near-tie (gqa2
s4: 200.8 µs for eft against 203.3 µs for rotate) where the measurement found a
0.96% loss. A model whose predicted gap is smaller
than its own S1-a bias (F-146) cannot call that ordering, and this one did not.

⚠️ Reproduced on a second architecture, 2026-09-13: on sm_120 the gate fails 0/4
again and by more — eft/mode 5 is 1.0409 / 1.0528 / 1.0542 / 1.0468 (pooled
1.0491 [1.0473, 1.0510]) against sm_89's 1.0096 / 1.0141 / 1.0022 / 1.0273, with
S2-e at +0.812 in every cell and the predicted improvement direction wrong 4/4.
The negative is not an sm_89 artifact. See the sm_120 follow-up section below.

## F-150 — What the negative result costs, and what it does not

⚠️ inferred, stated as this round's reading of F-148 and F-149 together: the
placement axis at `W = 1` is close to exhausted. Two plans built on opposite
principles — pure EFT with a measured cost model, and a two-line closed form
that owns no model — land within 3% of each other while both beat the legacy
placement by 1.14–1.53×. The remaining 0.2–4.9% is not where the next factor is.

The decomposition says where it is. Mode 5's `neither` runs at 0.20 / 0.23 /
0.35 / 0.29 of its `full`: **65–80% of the correct kernel's L2 time is the
per-task publish protocol**, and F-145 measured that 910 ns of the 1235 ns hop is
`__nanosleep(64)` backoff granularity rather than coherence traffic. That is
EX-E3, which R2 §1 excluded from this round by design, so it is recorded as the
next lever and not acted on. The second is EX-E2: F-134's reclaimable
head-of-line blocking is 18.9–64.0% of all stall time, and `W = 1` is what makes
it unreclaimable — with a window, EFT's concentrated queues stop being a
liability and its 3.6–4.2× cheaper synchronization becomes the whole of the
difference.

⚠️ sm_120 sharpens both halves of this, 2026-09-13. The publish protocol is
**68–91%** of mode 5's L2 time there rather than 65–80%, so the EX-E3 ordering
holds on a second architecture with a larger prize. The cancellation in F-148
also survives but stops balancing: mode 5's unsynchronized floor is 2.6–9.4×
better than EFT's (sm_89: 2.4–4.1×) while EFT's synchronization term is only
3.1–3.7× cheaper (sm_89: 3.6–4.2×), which is why the loss widens from 0.2–2.7% to
4.1–5.4%. The mechanism is the same; the coefficients are not.

## sm_120 scheduling follow-up, 2026-09-13

Verified from the RTX 5090 session: SIMULATOR completed both 48-cell sweeps
at 4096 rounds per cell, with zero recorded inversions and matching artifact
checksums. The RMW/backoff-64 fit has RMS residual 15.0 ns and reduced
chi-square 1936.5; sweep PASS does not establish fit adequacy.

Verified: PLACE_EFT stopped in the initial gqa2/seq=4 EFT invocation because
the copied sm_89 plan is pinned to grid 256 but this sm_120 run selected grid
340. All four legacy-source provenance comparisons passed and all 24
reference executables compiled, but the 50-fresh-process correctness gates,
paired timings, and decomposition were not reached. REALWIDTH was disabled.
This is a plan/grid transfer failure, not a successful sm_120 EFT experiment;
no guard or expected value was changed. Outputs remain under `raw_sm120/`,
without replacing sm_89 records. Full evidence and scope are recorded in
[the session report](experiments/sm120_simulator_place_eft_20260913.md).

### sm_120 EFT preparation repair

Verified from the failed invocation and runtime guard: materialized EFT
tables bind worker/slot assignments to a particular grid, so copying an
sm_89 table to sm_120 is not a valid cross-device preparation procedure.
The sm_120 wrapper now probes the frozen control sources on the selected
device, then solves using those measured counts/residency/grid, the sm_120
target, and the local sm_120 hop calibration. The runtime mismatch guard
remains unchanged. The experiment driver retains its original sm_89 defaults
for existing callers and accepts explicit target/hop inputs for the new path.
Without a trace, worker-SM assignment in prediction is explicitly modulo,
not borrowed sm_89 hardware placement. Output can be redirected with OUT_DIR;
the sm_89 raw tree is rejected as an output. Verified: driver rebuild,
two CPU preparation regression tests, shell syntax, and SELF_CHECK passed.
GPU retry results are recorded separately after execution, not inferred
from these preparation tests. The first on-device preparation attempt also
verified that the repository sm_120 BF16 cost profile was uncalibrated.
Preparation now measures a local BF16 target with tilemega-calibrate when
none is provided, preserves the repository target file, and refuses an
unaccepted or wrong-architecture profile. Verified: the on-device 41-repeat
calibration completed with calibrated=true, and a third CPU regression test
checks rejection of an unaccepted profile.

Verified: the repaired reference-only retry completed with grid 340 in all
four plans and unchanged frozen control-source provenance. Formal
correctness passed 1200/1200 (24 combinations, 50 fresh processes each);
paired timing passed 600/600. The 800 sync-decomposition probes completed:
safe full probes passed 200/200, while 600 intentionally unsafe probes
reported MISMATCH and remain timing-only. Script completion/PASS does not
mean the S2-b improvement gate passed: EFT was 4.1--5.4% slower than rotate
in all four cells, with each paired 95% confidence interval above one.
Predicted improvement direction disagreed with measurement 4/4. No guard,
expected value, tuning parameter or gate threshold was changed. REALWIDTH
was disabled. Per-process evidence, calibration limitations and selective
research checks are recorded in [the retry report](experiments/sm120_place_eft_retry_20260913.md).

## F-151 — The calibrated wait policy cuts the isolated hop by 86% and the kernel by 0.4%, and the sign reverses at seq 128

Verified: EX-E3 step 0 replaced the fixed `__nanosleep(64)` in the wait macro
and in the grid barrier with a graded policy read from `TargetSpec`
(`spin_iters`, `backoff_ns`, `backoff_grow`, `backoff_cap_ns`), and calibrated
it from a 168-point hop curve over eight arms. On sm_89 the smallest `c0` is
`spin_iters=64, backoff_ns=64, grow=1, cap_ns=64` at 165.3 ns against the status
quo's 1206.5 ns, a drop to 13.7%. Pure spin alone reaches 308.3 ns, so the
combination beats both extremes. Arm order was rotated three ways and the
spread is at most 7.2 ns, so the ranking is not an ordering artefact. sm_120's
target file carries `spin_iters=0` because its hop sits at a ~400 ns floor with
backoff worth only ~32 ns; that value is scripted, not measured here.

Verified: the same-SM spin interference question is answered and the answer is
architecture-relevant. A compute worker sharing an SM with a spinning worker is
unaffected when it is FMA-bound (ratio to idle 1.0000 for both spin and
backoff, 5120 paired samples) and slowed 0.34% by spin and 0.38% by backoff
when it is memory-bound, against a 3.03% p90/p50 tolerance. Spinning does not
cost issue bandwidth; it costs a little memory bandwidth, and backoff costs
marginally more of it than spin does.

Verified: the 86% cut in the isolated hop does not transfer to the kernel, and
at seq 128 it reverses. Paired ratios over 25 rotated rounds per cell, policy
on over off: gqa2 s4 0.995423 [0.993445, 0.997491], mha4 s4 0.995762 [0.993440,
0.997109], gqa2 s128 1.005000 [1.004643, 1.005269], mha4 s128 1.005281
[1.004224, 1.006614]. All four confidence intervals exclude one, so both the
0.4% gain at seq 4 and the 0.5% loss at seq 128 are real.

The cause is located in the four-arm decomposition, not inferred. The `wait`
term the policy targets is 0.023–0.041 ms of a 0.45–1.14 ms kernel, so removing
it entirely would cap the gain near 4%; the `notify` term beside it is
0.065–0.151 ms, between 2.6× and 3.5× larger. Waiting is not where the L2
protocol spends its time in these kernels — publishing is. At seq 128 the
`wait` term does not fall at all under the policy but rises, 0.023520 to
0.026368 at gqa2 and 0.037568 to 0.043040 at mha4, which is the same sign as
the memory-bandwidth interference measured above and is consistent with §8.3's
stated reason for backoff existing. Seq 128 tasks are longer and
memory-bound, so more workers spin concurrently against computing neighbours
than at seq 4.

Verified and worth separating: the policy helps L1 more than L2. The grid
barrier at `ClusterSync.cuh` uses the same wait macro, so the L1 baseline moved
too — gqa2 s4 0.404480 to 0.392384 ms (−3.0%) against L2's −0.44%, mha4 s128
1.029120 to 1.019904 ms against L2's +0.5%. Any `l2_over_l1` ratio therefore
moves against the policy even in the cells where L2 improved, and that is an
artefact of the shared macro rather than a regression in the megakernel.

The concrete next step is not a wider spin budget but a narrower one. The
calibration optimised `c0` for a single waiting worker; the in-kernel loss
appears where many workers wait at once, a count the Plan already knows as the
fan-in of each event. Sizing `spin_iters` per event from that fan-in, carried
in `TargetSpec` as two budgets rather than one, is the change that would let
seq 4's gain survive at seq 128.

## F-152 — Removing three of eleven per-task barriers is worth between nothing and 0.29%

Verified: EX-E3 step 1 cut `tilemega_l2_kernel` from 11 `BAR.SYNC` to 8 in the
plain build and to 10 under trace v2, leaving `membar` at 4 and `nanosleep` at
1 untouched, with the executor's per-task count at 2. Correctness is 200/200
across both models × seq ∈ {4, 128} and 1500/1500 over the 30-cell SEQSCAN
matrix, with both negative controls behaving (old clamp 50/50, task-wait clamp
0/50).

Verified: the measured gain is near zero. Paired ratios on over off, 25 rounds
per cell: gqa2 s4 0.999713 [0.997420, 1.000503], mha4 s4 1.000000 [0.997507,
1.001153], mha4 s128 0.998291 [0.996567, 1.000649], gqa2 s128 0.997087
[0.996724, 0.998648]. Only gqa2 s128's interval excludes one, at 0.29%.

The cause is visible in the same decomposition and is a matter of scale, not of
implementation. The `barrier` term is 0.037–0.089 ms of a 0.44–1.24 ms kernel,
and deleting three of eleven barriers moves it by about a twelfth of itself —
0.038720 to 0.037152 ms at gqa2 s4. Per-task CTA barriers are not a material
share of these kernels' time, which is consistent with 910 of the status quo's
1235 ns hop being backoff rather than barrier. This is recorded as a mechanism
that missed its expected gain, which §10 explicitly does not make a stop
condition; the lever it was competing with is the publish side measured in
F-153.

## F-153 — Publishing single-member events directly is the largest E3 step measured so far, at 0.5%

Verified: two thirds of all event arrivals in the reference models have a
single member — 148 of 196 calls at gqa2 s4 (75.5%), 3372 of 4412 at gqa2 s128
(76.4%), 332 of 508 at mha4 s4 (65.4%), 7772 of 11916 at mha4 s128 (65.2%) at
κ=1. EX-E3 step 2 skips `atomicAdd(&events[index].arrivals, 1)` for those and
publishes the epoch directly.

Verified: paired ratios on over off are 0.995110, 0.995017, 0.995412 and
0.995425 across the four cells, every confidence interval strictly below one,
p ≤ 3.1e-03, with correctness 200/200 and SEQSCAN 1500/1500. The notify term
moves in the direction the mechanism predicts — 0.147456 to 0.141312 ms at mha4
s128 — so the gain is attributable to the publish path rather than to noise
elsewhere in the decomposition.

Verified by comparison: this is a larger effect than either the calibrated wait
policy (F-151, +0.4%/−0.5% depending on sequence length) or the barrier
reduction (F-152, 0–0.29%), and it is the only E3 step so far that improves
every cell. It is also the cheapest of the three. That ordering is itself the
round's clearest signal about where L2 protocol cost lives: on the publishing
side, not the waiting side.

⚠️ 2026-09-14: superseded as the largest E3 step by F-156, which measures the
release-ordered publish at 2.5–3.4%, five to seven times this one. The title's
"largest so far" is left as it was measured. The ordering claim in the last
paragraph is not superseded — it is confirmed, and by a wider margin.

## F-154 — Critical-path chaining does not shorten the critical path at seq 128, and three candidate causes are excluded by measurement

Verified: EX-S2c extracts the longest path on the exact runtime task DAG and
places it whole on one worker, then clusters and fills per §6. At seq 4 it does
what §6.1 predicts: `critical_path_hops` 17 against rotate's 19 at gqa2 and 35
against 39 at mha4. At seq 128 it reverses — 18 against rotate's 17 at gqa2, 39
against 35 at mha4 — and the same reversal appears in the real-width cell.
`critical_path_same_worker_edges` rises exactly where the hop count rises, 0 to
20 at mha4 s128 and 0 to 88 at real s128: each hop chaining removes is replaced
by a queue edge, and the replacement is not free.

Verified: the statically longest path is not the path that ends up critical. At
mha4 s128 the spine is 40 nodes and 362813 ns while the scheduled critical path
is 87 steps, 28 of them queue steps. The post-placement critical path is a
different and longer path manufactured by queueing.

Three candidate causes were tested and excluded, each by its own sweep rather
than by argument.

Extraction coverage is not the cause. `chain_stop_ratio` swept 1.0, 0.5, 0.25
and 0.05 forces 4, 84 and 256 chains on mha4 s128 and moves `critical_path_hops`
only 39 to 38, while making gqa2 s128 strictly worse, 18 to 22
(`raw/stop_ratio_sweep.tsv`). Twenty-one times more chains buys five
critical-path nodes.

Fill pricing is not the cause. Uncapping the fill leaves both s4 cells
byte-identical, makes gqa2 s128 worse at 19 hops against the capped 18, and on
mha4 s128 gives 39 hops and 523779 ns against rotate's 35 and 437124
(`raw/fill_cap_sweep.tsv`).

Feeding the measured queue delay back into the extraction is not the cause
either, and its failure localises the problem precisely. Re-scoring every node
by the delay the previous pass measured it spending waiting for its worker, then
re-extracting and keeping the best pass, reaches a fixpoint after one round and
does not help: s4 and gqa2 s128 are unchanged, mha4 s128 moves from 39 hops to
41 (`raw/feedback_sweep.tsv`). The reason is that the feedback does improve the
objective it can see — the solver's own simulated makespan falls from 508348 to
507331 ns at mha4 s128 and from 4588802 to 4588788 at real s128 — while the
placement evaluator scores the same plans in the opposite direction, 530126 to
532494 ns. The two simulations do not model queueing identically.

⚠️ inferred from the above: §6.1's premise, that extracting the longest path and
placing it whole shortens the critical path, holds on these graphs at seq 4 and
does not hold at seq 128. The mechanism is default-off and the feedback rounds
are exposed only as an ablation knob.

The concrete next step is to make the two simulations agree before extracting
again. The ranking DP in `lib/Solver/ChainPlacement.cpp` scores a path as
`task_ns + hop_cost` with no queue term, and the pass's own simulation carries
worker occupancy that the evaluator models differently; closing that gap — one
simulation, used both to rank and to score — is a precondition for any further
chaining work, and is a smaller change than the chaining itself was.

## F-155 — Chaining loses at seq 128 on queue-edge task work, not on hops, and one simulation for ranking and scoring does not recover it

F-154 named the next step as making the extraction's two simulations agree: one
simulation, used both to rank and to score. That is now implemented.
`ChainRequest::evaluate` takes an arbiter from the caller, and
`docs/experiments/CHAIN/place_chain.cpp` supplies the same `SimulateExecution`
that scores the emitted plan, built from the same options, hop curve and
weights. Both the selection and the next round's ranking term come from it, so
the pass's own estimate is used for neither. The mechanism stays default-off:
`feedback_rounds` is zero, the arbiter is never constructed there, and the
committed `raw/predicted.tsv` reproduces byte-identically through column 20
(column 21 is wall-clock `eval_us`).

✅ Verified, `raw/feedback_sweep_arbiter.tsv`: the defect F-154 measured is
gone. Feedback is now monotone in the objective that reports the result — mha4
s4 403305 → 402832 ns, gqa2 s128 229487 → 228306 ns, every other cell flat —
where the previous loop moved mha4 s128 the wrong way, 530126 → 532494 ns. The
pre-arbiter sweep F-154 cites, `raw/feedback_sweep.tsv`, is kept as it was
measured rather than regenerated: the two files are one sweep over two code
states, not two runs of one, and overwriting it would have left F-154 stating
numbers its own evidence no longer contained.

✅ Verified: it does not fix the gate. S2c-b still fails two of four reference
cells at every feedback depth in {0, 1, 2, 4}: gqa2 s128 18 hops against
rotate's 17, mha4 s128 39 against 35. The two simulations disagreeing was real,
and it was not the cause.

Worth recording because it constrains the objective: at gqa2 s128 the arbiter
improves makespan from 229487 to 228306 ns while the hop count *rises* from 18
to 20. Makespan and `critical_path_hops` diverge on these graphs, so selecting
on hops would move the gate's own metric without moving the quantity the gate
exists to reduce.

### Where the loss actually is

✅ Verified, `raw/path_decompose.tsv`, which splits each realized path's task
work by the edge that carried it. At mha4 s128 chain loses 93003 ns to rotate:

| term | chain | rotate | delta |
| --- | --- | --- | --- |
| task work on queue edges | 78894 | 12576 | +66318 |
| task work on task edges (h + s) | 378535 | 360293 | +18242 |
| hop gap | 48078 | 43160 | +4918 |
| stretch excess | 23035 | 19511 | +3524 |

71% of the loss is task work the path spends behind tasks merely queued ahead of
it: 28 queue edges carrying 78894 ns against rotate's 32 carrying 12576. The
same shape holds at real width — queue share 0.1322 against 0.0516, 600221 ns
against 202583 — and does not hold at gqa2 s128, where chain's queue term is the
*better* of the two, 3902 against 4878, and the 14360 ns loss is 6916 ns of
heavier task-edge work. The two failing cells fail for different reasons.

✅ Verified: the internalisation mechanism itself works. `task_h_ns` is within
0.3% of rotate's in every seq-128 cell — 361229 against 360293 at mha4 s128 —
and chain carries 20 same-worker task edges worth 17306 ns that rotate pays as
hops. The 39-against-35 hop count follows from the path being 88 steps instead
of 68, and the extra steps are the queue detours. ⚠️ inferred: the hop excess
S2c-b measures is therefore downstream of the queue term, and the gate is not
out of reach on these graphs — it is blocked behind the fill, not behind the
extraction. This corrects a reading taken earlier in the same investigation,
that four mutually cross-linked equal-weight lanes make the hop count
structurally irreducible; the lane structure is real, but it is not what the
measurement attributes the loss to.

### A measurement that does not mean what it looks like

✅ Verified, recorded because it cost a working hypothesis: `block_ns` in the
path dumps is `start - free_at[w]` (`lib/Solver/ExecutionSimulator.cpp:208`),
the idle gap on one *worker* before a task. It does not telescope along a path
and does not stay inside the span — at gqa2 s128 rotate it sums to 394397 ns
inside a 215127 ns span — and its zero on every queue and same-worker step is a
tautology of the definition, not evidence that those edges are free. Only the
edge-type split of `task_ns` supports a claim about where a path's time goes.

### Next step

The fill chooses each node's worker by minimising that node's own finish time,
in topological order, with no notion of which nodes will end up critical
(`lib/Solver/ChainPlacement.cpp:285-340`). Ordering the fill by remaining
downstream path length is the untested lever, and it is distinct from the three
already measured: the stop ratio sets how many chains are extracted, `cap_fill`
sets the fill's bound, and the feedback objective sets the ranking term, and
none of them changes which worker a filled node lands on. The constraint on any
such change is that phase 3 prices `free_ns[w]` in the order it assigns while
phase 4 re-sorts every queue by topological index
(`lib/Solver/ChainPlacement.cpp:352-358`), so a priority-ordered assignment must
not desynchronise the two.

## F-156 — The release-ordered publish is the largest E3 step by a factor of five, and it pays for it on the polling side

EX-E3 step 3's publish side: the producer's `atomicAdd` on the aggregate row
becomes a return-value-free release reduction (`TILEMEGA_EVENT_RED_PUBLISH`,
default off). R3 §5 requires the publish side and the polling side be reported
separately and never merged; this entry is the publish side only. The polling
side is a different switch (`TILEMEGA_EVENT_LOAD_POLL`) and R3 §5 requires its
earlier negative result be re-measured rather than cited, so nothing here
claims anything about it.

✅ Verified, `docs/experiments/SYNC_V2/raw_red/`: correctness 200/200 over four
cells and SEQSCAN 30 cells 1500/1500, no short row. Paired on-over-off ratios on
the `full` arm, 25 rounds, rotating arm order:

| cell | ratio | CI95 | p | delta (ms) |
| --- | --- | --- | --- | --- |
| gqa2 s4 | 0.965596 | [0.963387, 0.969905] | 1.29e-05 | −0.015360 |
| mha4 s4 | 0.970885 | [0.969620, 0.972463] | 2.27e-04 | −0.023488 |
| gqa2 s128 | 0.974938 | [0.973631, 0.976667] | 1.30e-05 | −0.015424 |
| mha4 s128 | 0.973170 | [0.972380, 0.974087] | 2.53e-04 | −0.031392 |

✅ Verified, and this is the part that matters: the mechanism moves cost between
two terms rather than only removing it. `notify` falls 29.6%, 30.1%, 33.5% and
34.1% across the four cells, while `wait` rises 17.3%, 30.0%, 36.5% and 53.6%.
The publish side wins by more than the polling side loses in every cell, and the
two terms account for the end-to-end result: notify's fall minus wait's rise
reproduces the measured delta to within 5% in all four cells (gqa2 s4 −0.015296
against −0.015360 ms; mha4 s128 −0.031360 against −0.031392). The win is
therefore attributable to the protocol decomposition, not to drift elsewhere.

⚠️ inferred: the `wait` rise is the consumer observing a publish it no longer
shares a fence with, so the cost reappears where the consumer polls. This is
stated as the reading of the decomposition, not as a separately measured
mechanism.

✅ Verified at the instruction level, `raw_red/census.tsv`: in the plain build
`membar` falls 4 → 2 and counted `atomics` 9 → 7 with `bar_sync` unmoved at 11
and `nanosleep` at 1. The release reduction removes two fences as well as two
atomics rather than merely rewriting an `atomicAdd` in place, which is why the
notify term moves as far as it does. The trace build reads 11 atomics at v2=1
because trace stamps publish from every red arrival (commit `41d85013`), not
because the publish path differs there.

✅ Verified by comparison, `docs/experiments/SYNC_V2/e3_steps.tsv`: against the
other three E3 steps measured in isolation against their own baselines — the
calibrated wait policy (F-151), the barrier reduction (F-152) and the
single-member publish (F-153) — this step is five to seven times the largest of
them, and it is the only one that moves `notify` at all. The other three touch
the `wait` and `barrier` terms, which is why they are inert end to end: `notify`
is the largest positive term in the decomposition, 0.057–0.184 ms, and until
this step nothing in E3 had attacked it.

⚠️ The four ratios above cannot be multiplied to predict the all-on
configuration. Each switch was measured in its own fresh-process session per H6,
so their `l2` baselines differ for the same cell (gqa2 s4: 0.446368, 0.445440,
0.416768, 0.446368 ms). The cumulative staircase R3 §8 asks for is a separate
measurement, not a derivation from these.

## F-157 — The thread0 release fence is indistinguishable from the per-writer fence wherever the harness can tell them apart, and that is six cells of eighteen

✅ Verified, `raw_litmus/litmus.tsv`: EX-E3 step 4 ran four release shapes over
grid ∈ {64, 128, 256} × tile ∈ {1024, 4096, 16384} × acquire ∈ {1, 0}, 50 fresh
processes per cell — 18 cells and 900 runs per arm, 3600 runs in all. The rule
§8.5 states, `per_writer` (every writer fences, then the barrier, then thread 0
publishes), passed 900/900. The candidate `thread0_fence` (barrier first, then
one `__threadfence()` by thread 0, then publish) also passed 900/900. Per H3
this step changes nothing: §8.5 stands as written, and what follows is a
conclusion, not a licence.

⚠️ The number that bounds this conclusion is not 900 but 6. A cell is evidence
only where the harness can observe a missing release at all, which the
`no_fence` sensitivity arm measures directly: it mismatches only with the
consumer's acquire fence dropped and the tile at 1024 or 4096 elements — six
cells. At tile 16384 it passes 50/50 in every grid, and with acquire = 1 it
passes 50/50 everywhere, so twelve of the eighteen cells cannot distinguish any
release shape from any other. `thread0_fence` held in all six readable cells and
was never put under load in the other twelve. "Indistinguishable where the
harness can tell them apart" is the whole claim; F-3 and F-10 predicted exactly
this tile dependence.

⚠️ Verified and not smoothed: the negative control sleeps in one of those six
cells. `no_barrier` — the per-writer fence kept, the consumer-side barrier
removed — mismatched in 849 of 900 runs, but at (acquire = 0, grid 128, tile
4096) it passed 50/50, and at (acquire = 0, grid 128, tile 16384) it passed once
in 50. The hard gate asks the control to fire in a readable cell and it fires in
five of the six, so E3-4 passes; the sixth is recorded rather than rounded up.
⚠️ inferred as to cause: the race the control opens is the window between thread
0's publish and the other warps' stores retiring, and its width depends on
occupancy and on how long a tile's stores take. grid 128 at tile 4096 is the one
combination where that window closed on this device — grids 64 and 256 at the
same tile both mismatch 50/50 — so it is not a property of the tile size alone.

✅ Verified, `raw_litmus/per_writer_vs_thread0.diff` and `raw_litmus/census.tsv`:
the candidate is not cheaper in static instructions. Both shapes compile to the
same counts — `membar` 2, `atom_red` 11, 800 instructions — and the diff shows
why: the `MEMBAR.SC.GPU`, `ERRBAR` and `CCTL.IVALL` triple does not disappear, it
moves. In `per_writer` it sits ahead of `BAR.SYNC.DEFER_BLOCKING`, where every
thread runs it; in `thread0_fence` the barrier comes first and the triple lands
past the `@P1 BRA` that sends every non-zero thread away, so one thread runs it.
The saving is dynamic and its size is the CTA's warp count, which no instruction
census can show.

What this means for round four's E3-5: the correctness question this step was
built to answer is answered as far as this harness reaches, and the cost question
is not asked here at all — no timing arm was run, because §5 scoped this step to
a conclusion. E3-5's async publish rests on the same shape, so what it needs next
is a paired timing of the two release shapes inside the megakernel, not another
litmus. The detector gap is the thing to fix first: a litmus that cannot see a
missing release at tile 16384 cannot certify a publish shape for the tile sizes
the real models actually write.

## F-158 — No chain placement can close a cycle, so the split count is provably zero, and §6.3's hazard is not the condition that gates a Plan

✅ Verified, `raw/chain_weights.tsv`: `split_count` is 0 in all ten rows — both
weight sources, all six cells. R3 §6.3 asks for the number of chains the
implementation had to split to keep the Plan legal, and the honest answer is that
the number is zero for a structural reason rather than a lucky one.

The argument is a proof rather than an observation. A chain is a path in the task
DAG, so the queue edges between its consecutive members are task edges already;
and every queue is ordered by `topo_index` (`lib/Solver/ChainPlacement.cpp` phase
4), so the union of task edges and queue edges is a subset of one topological
order. L-a tests exactly that union for acyclicity. No chain placement can close
a cycle, and there is nothing for a split to repair.

⚠️ Recorded, not reconciled: §6.3 describes a hazard that is real but is not L-a.
Two chains with cross edges in both directions — a1→a2 and b1→b2, plus a1→b2 and
b1→a2 — close a cycle in the *contracted* graph, where each chain is one node.
That condition is strictly stronger than L-a and is not what gates a Plan.
`test/unit/chain_placement_test.cpp` constructs precisely that case: four nodes,
edges 0→1, 2→3, 0→3, 2→1, weights 100/100/90/90 and grid 2, chosen so the
extractor picks 0→1 and 2→3 and no mixed path. Both chains land whole,
`split_count` is 0, the two land on different workers, and `CheckPlanLegality` —
the check that actually gates a Plan, not the test's own idea of legality —
accepts the result. A cycle in the union would need a2→b1 together with b2→a1,
which is already a cycle in the task DAG and so can never reach a scheduler. The
same test hands `ScheduleByCriticalChain` a genuine two-node cycle (0→1, 1→0) and
requires it to be refused with "cycle" in the message. The binary prints
`chain_placement_test: ok`.

✅ Verified and worth keeping: enforcing the contracted form instead — this
round's first attempt — cost the mechanism everything it was for. At gqa2 s4 it
refused 770 chain extensions over 200 nodes and left a 19.9 µs spine against a
179.8 µs longest path, because a bypass around a path edge breaks convexity and
these graphs are full of bypasses. What is left worth protecting is contiguity,
which is a performance property, and it is counted as `chain_interleaves` rather
than enforced.

✅ Verified, and the answer to §6.2's instruction to try both weight sources and
report the difference: they are not interchangeable, and at seq 128 they are
barely related.

| cell | nodes | spine, cost model | spine, trace | solo work, cost model | solo work, trace | nodes moved |
|---|---|---|---|---|---|---|
| gqa2 s4 | 200 | 179817 ns | 223232 ns | 1.34 ms | 1.65 ms | 97 (48.5%) |
| mha4 s4 | 512 | 359634 ns | 448512 ns | 2.99 ms | 3.68 ms | 240 (46.9%) |
| gqa2 s128 | 4416 | 181407 ns | 314368 ns | 4.32 ms | 25.30 ms | 4359 (98.7%) |
| mha4 s128 | 11920 | 362813 ns | 628736 ns | 9.66 ms | 52.83 ms | 9378 (78.7%) |

⚠️ inferred, from the shape of that table: trace durations are measured under
co-residency and therefore carry the stretch the cost model prices at solo, which
is why the gap widens with sequence length — 1.2× the total work at seq 4, 5.9×
at gqa2 s128. The extraction is not robust to it. At gqa2 s128 the trace source
moves 98.7% of nodes to a different worker and collapses the chain count from 36
to 4, so the two sources do not produce variants of one schedule; they produce
different schedules. The two `makespan_ns` columns must not be compared across
sources, because the node weights themselves differ — what is comparable is the
placement each induces, which is what `worker_diff` counts.

The trace source is also not always available: it needs a round-one trace
directory, which the two `real` cells do not have, so those rows exist only in
the cost-model form. Any decision to prefer trace weights would therefore have to
carry a fallback for exactly the widths that matter most in serving.

## F-159 — The slot window adds head-of-line time instead of reclaiming it, and inflates its own ceiling while doing so

✅ Verified, `docs/experiments/WINDOW/raw/summary.tsv` (E2-d) and
`raw/analysis/analysis.tsv`: widening the execution window makes every reference
cell slower and leaves more head-of-line time behind, not less.

| cell | W | `measured_l2_ms` | `hol_reclaimable_ns` | `hol_workers_nonzero` |
|---|---|---|---|---|
| gqa2 s4 | 1 / 2 / 4 | 0.428032 / 0.470016 / 0.471904 | 1274880 / 1347584 / 1360896 | 8 / 8 / 8 |
| gqa2 s128 | 1 / 2 / 4 | 0.592896 / 0.646144 / 0.647296 | 79889408 / 86979584 / 86464512 | 256 / 256 / 256 |
| mha4 s4 | 1 / 2 / 4 | 0.849920 / 0.934912 / 0.939104 | 7692288 / 8433664 / 8445952 | 16 / 16 / 16 |
| mha4 s128 | 1 / 2 / 4 | 1.277952 / 1.324032 / 1.325056 | 207584256 / 217905152 / 216441856 | 256 / 256 / 256 |

`hol_delta_vs_w1_ns` is signed as reclamation — `summarize.py` computes W=1's HOL
minus this row's — so its negative value in all twelve rows means the window
added head-of-line time. `hol_workers_nonzero` never moves: the same 8, 16 and
256 workers block at W=4 as at W=1. HOL was therefore never the binding
constraint, and a wider window has nothing to spend it on. F-134 measured HOL as
18.85–63.98% of stall time; that remains true and is not the same claim as HOL
being *reclaimable* by reordering within a worker.

✅ Verified, and it is the reason the E2-d ratio column must not be read as a
result: `measured_over_ceiling` improves from 3.1429 to 1.9125 (gqa2 s4), 6.2406
to 1.9384 (mha4 s4) and 6.5000 to 1.9473 (mha4 s128) **while the measured kernel
gets slower in each**. The numerator rose and the ratio still fell, because the
denominator rose faster.

✅ Verified cause, by exact counting rather than inference. `cp_lb_nosync` is a
longest path over edges that `TRACE_V2/analyze.py` recovers from *recorded
waits*: its `preds` is built from the wait rows, and `longest` classifies an edge
as a queue edge when its two endpoints share a worker. H4 stops eliding the
same-worker producer poll for slots inside the window, so those polls now execute
and enter the trace, and each contributes one edge counted as cross-worker. At
seq 4, where the elision change is the only thing moving, the counts match
exactly: `dag_cross_worker_edges` 1028 → 1108 (+80) at gqa2 s4 and 3396 → 3604
(+208) at mha4 s4, with `dag_same_worker_edges` flat at 48 and 176. `cp_lb_nosync`
rises with them, 0.136192 → 0.245760 ms and 0.136192 → 0.482304 ms. That the W=1
value is byte-identical across two different models (0.136192 in both) is the
same effect at its maximum: maximal elision leaves both models the same sparse
skeleton.

⚠️ Recorded rather than generalized: that clean split holds at seq 4 only. At
seq 128 both columns roughly double (gqa2 s128 same-worker 1040 → 2072,
cross-worker 278932 → 545140), so "same-worker edges are unchanged under W" is
not a general claim.

✅ Verified, what the window does buy: `wait_total_ns` +12.4% and
`publish_total_ns` +13.5% at gqa2 s4 (3558400 → 4000768 and 281600 → 319488),
`hop_p90_ns` 2048 → 25600 there and 7168 → 27648 at mha4 s4, while
`idle_fraction_of_worker_time` barely moves (0.7505 / 0.7562 / 0.7587). More
polling, no more overlap.

Next step, per §0 item 2 and recorded in `PLACE_EFT2/summary.md` §11: the ceiling
must stop being a function of the executor's elision policy before the window can
be evaluated at all. `preds` should be built from the task DAG and the
materialized σ — which the CG skeleton has exactly — and an edge classified as a
queue edge by whether it *is* one, not by whether its endpoints share a worker.
Until then no ceiling is comparable across executor configurations. W stays at 1.

## F-160 — The W=1 lifting rules fail in every cell under a W=2 executor, 0 of 200 processes

✅ Verified, `docs/experiments/WINDOW/raw/negative.tsv`, H4's mandatory negative
control: built with `TILEMEGA_NEGATIVE_WINDOW_W1_RULES=1`
(`ModelHarness.cuh:96`), which keeps the W=1 wait-lifting and poll-elision rules
while running the W=2 executor, every one of the four reference cells fails.

| model | seq | passes | processes | failure rate |
|---|---|---|---|---|
| gqa2 | 4 | 0 | 50 | 1.0000 |
| gqa2 | 128 | 0 | 50 | 1.0000 |
| mha4 | 4 | 0 | 50 | 1.0000 |
| mha4 | 128 | 0 | 50 | 1.0000 |

200 fresh processes, 0 passes. The rules are not redundant bookkeeping: lifting a
wait out of slot `j` from a producer in slot `i` is only sound when `i ≤ j − W`,
and eliding a same-worker producer poll is only sound under the same condition.
Run the W=2 executor under the W=1 conditions and the results are wrong, every
time, in every cell.

⚠️ Worth stating because §10 requires it: the informative outcome here is the
one that occurred. Had the control *passed*, that would have been a stop
condition and specifically not evidence that the rule can be dropped — it would
have meant the reference models never reach the reordering the rule exists to
make safe, and H4 would then require a graph that does. As measured, the four
reference models do reach it.

## F-161 — Configuration B alone is the best cell in all four references; the three levers do not compound

✅ Verified, `docs/experiments/PLACE_EFT2/raw/final`, 25 paired rounds per
cell-candidate-configuration, configuration and candidate rotated together so
neither axis sits at a fixed point in a session's drift (H6). 96 cell-arm-config
cells, every one at n=25. Median L2 in microseconds; A is no flags, B the
calibrated wait policy plus E3 steps 1-3, D is B plus the window at W=2, W the
window alone.

| cell | best under A | best under B | best under D | best under W |
|---|---|---|---|---|
| gqa2 s4 | rotate 292.9 | **chain 280.6** | rotate 294.9 | rotate 304.0 |
| gqa2 s128 | rotate 456.7 | **eft 440.3** | wavefront 465.8 | rotate 477.2 |
| mha4 s4 | rotate 578.7 | **chain 557.8** | chain 587.8 | rotate 601.1 |
| mha4 s128 | rotate 860.2 | **rotate 833.6** | rotate 877.6 | rotate 901.1 |

✅ Verified, and it is a property of every candidate rather than of the winner:
the configuration ordering **B < A < D < W** holds in 23 of the 24 (cell,
candidate) pairs. The one exception is a tie, not a reversal — `chain` at gqa2
s128 gives A = D = 519.2 µs. Pairwise the pattern has no exception at all: B < A
in 24/24, B < D in 24/24, D < W in 24/24.

Holding the candidate at `rotate` to read the mechanism sizes directly, against
A: B is −2.80 / −2.73 / −2.70 / −3.09 percent across the four cells; D is +0.70 /
+2.22 / +1.56 / +2.02; W is +3.79 / +4.48 / +3.87 / +4.76.

✅ Verified consequence, and it is the round's central negative result: **the
three levers of R3 §1 do not compound.** D is the configuration that has all of
them on, and it is slower than A — the protocol's ~3% is smaller than the
window's ~4-5% regression, so turning both on is worse than turning neither on.
No cell in the round is won by C (protocol + chaining) or by D. The best
configuration in all four reference cells is B, the protocol alone.

⚠️ Recorded as a sign change rather than a ranking: chaining is the only
mechanism whose value depends on the cell. Under B it wins both seq 4 cells
(280.6 against rotate's 284.7, and 557.8 against 563.1) and loses both seq 128
cells, by 11.3% at gqa2 (494.6 against eft's 440.3) and by 36.0% at mha4 (1133.6
against rotate's 833.6). The discrete-event predictor called the direction in
advance — chain's predicted makespan at gqa2 s128 is 229487 ns against eft's
215120 ns — so this is the predictor agreeing with the measurement, not a
surprise. F-154, F-155 and `PLACE_EFT2/summary.md` §11 locate the cause in the
fill pass rather than in the capacity cap, by the sweep that tests the cap.

✅ Verified distance to the §7.3 ceiling, which is what the research gate S2r-b
is anchored to. `target = ceiling_A + 0.5 x (measured_A - ceiling_A)`; the gate
asks for the best configuration's median at or under target with the 95% CI
upper bound strictly below it, in at least 3 of 4 cells.

| cell | ceiling | target | best measured | 95% CI | gap closed |
|---|---|---|---|---|---|
| gqa2 s4 | 143 | 218 | chain_b 280.6 | [280.6, 281.6] | 8% |
| gqa2 s128 | 197 | 328 | eft_b 440.3 | [440.1, 441.2] | 7% |
| mha4 s4 | 144 | 362 | chain_b 557.8 | [556.2, 559.1] | 5% |
| mha4 s128 | 199 | 528 | rotate_b 833.6 | [831.6, 848.8] | 3% |

**0 of 4, and not narrowly.** The round closed 3-8% of the distance between the
round-two measurement and the ceiling where the gate asks for 50%. Per H7 the
gate is not redefined, widened, or re-scoped, and per R3 §0 item 2 this is not
written up as a finding that the direction is exhausted: the cause is located in
F-162 and in `summary.md` §11, and the next step named there.

## F-162 — The polling-side re-measurement timed one binary against itself: `TILEMEGA_EVENT_LOAD_POLL` has no reachable call site in the build it was measured in

R3 §5 requires the old "poll the count directly" negative result to be
re-measured rather than cited, and reported separately from the publish side.
The re-measurement ran and returned a clean null. It is void data, and is
recorded here as void rather than as a reproduced negative result.

✅ Verified what ran. `SYNC_V2/run_barrier.sh` with
`SWITCH=TILEMEGA_EVENT_LOAD_POLL TAG=poll OUT_DIR=.../raw_poll`, so the two arms
were `-DTILEMEGA_EVENT_LOAD_POLL=1` and `=0`: four attribution arms x switch
off/on, 25 paired rounds with the arm order rotated (H6). The phase passed every
gate it carries — correctness 4 cells 50/50, the SEQSCAN matrix 30 cells
1500/1500, and both negative controls behaving (`old_clamp` 50/50,
`task_wait_clamp` 0/50).

| cell | n | ratio on/off | 95% CI | delta ms | p |
|---|---|---|---|---|---|
| gqa2 s4 | 25 | 1.001584 | [0.999928, 1.002371] | +0.000704 | 1.577e-01 |
| gqa2 s128 | 25 | 0.999687 | [0.998331, 1.000000] | −0.000192 | 7.311e-02 |
| mha4 s4 | 25 | 1.000000 | [0.997699, 1.002093] | 0.000000 | 7.494e-01 |
| mha4 s128 | 25 | 0.999719 | [0.998332, 1.001406] | −0.000352 | 8.077e-01 |

⚠️ The table above says nothing about load polling. `mha4 s4` returns a ratio of
exactly 1.000000 on a delta of exactly 0.000000, and `gqa2 s128`'s CI upper bound
is exactly 1.000000. Paired medians that agree to the last digit are what timing
one device image against itself produces.

✅ Verified that both arms are the same device image, by two independent
measurements. The SASS census is identical across the switch for both models and
for both the plain and the trace build — `tilemega_l2_kernel` 11 `BAR.SYNC` / 4
`MEMBAR` / 1 `NANOSLEEP` / 9 counted atomics, `tilemega_l1_kernel` 11/3/1/4,
`tilemega_stage_kernel` 9/0/0/0. The dumps are byte-identical, and the first 16
hex digits of their sha256 are `e31a9311ca09a94c` for `raw_poll` v0, `raw_poll`
v1, `raw_barrier` v0 and `raw_solo` v0 alike: the "on" arm shares a device image
with two unrelated experiments' baselines.

✅ Verified that the switch was passed and is honored. The driver set the flag,
and a minimal translation unit shows the preprocessor selecting the
`cuda::atomic_ref::load(memory_order_relaxed)` branch of `EventPoll` under it.

✅ Verified that the switch does change code, in isolation. Compiling a probe
that calls `EventPoll` directly, `nvcc -arch=sm_89 -cubin` with and without the
macro, gives `2 ATOMG.E.ADD.64.STRONG.GPU` off against `2 LD.E.64.STRONG.GPU`
on, 140 differing lines, with the `VOTEU.ANY`/`FLO.U32` warp-aggregation preamble
present only in the off arm. The macro works; nothing reached it.

✅ Verified cause: at this round's default switch settings every one of the three
`EventPoll` call sites is dead at compile time, each for its own reason.

1. The executor's wait path never calls it. Each of the 134 generated sources
   defines `TILEMEGA_GENERATED_WAIT_global` itself, at its own line 9, with the
   poll spelled inline as `atomicAdd((ev), 0ull)` — before it includes the
   harness at line 18. The harness's `#ifndef` fallback, which is the definition
   that would route through `EventPoll`, therefore never fires for a real model;
   the header says so itself at `ModelHarness.cuh:65-68`. The harness's
   `#undef`/override, which redirects the macro to `GradedWait` and so to
   `EventPoll` at `EventSync.cuh:79-81`, is gated on `#if TILEMEGA_WAIT_POLICY`,
   which the poll arm never set.
2. `ProbeTaskDependencies` (`ModelHarness.cuh:604`) and its only call site
   (`:645`) both sit inside `#if TILEMEGA_SLOT_WINDOW > 1`, which opens at `:599`
   and closes at `:653`. The default is `W = 1`, so the whole window path is
   preprocessed away.
3. `ClusterSync::StageBarrier` (`ClusterSync.cuh:163`) is reached only from
   `GridBarrier`'s `#elif TILEMEGA_GENERATED_CLUSTER_DIM > 1` branch at
   `ModelHarness.cuh:822`. Neither generated source defines that macro, so the
   harness default of 1 at `:80-82` applies, the branch is preprocessed out, and
   a never-called member of a class template is never instantiated.

⚠️ So the finding is not "load polling costs nothing". It is that the switch has
no reachable call site in the configuration it was measured in, and the cost of
load polling remains unmeasured. F-156 deferred the polling side to this switch;
that deferral still stands.

**Next step**, concrete and cheap. Measure the switch on top of the calibrated
policy, which is the configuration in which `EventPoll` is live: reuse the
`ON_FLAGS`/`OFF_FLAGS` path of `run_barrier.sh` that E3-0 already uses, with
`OFF_FLAGS` the calibrated policy string and `ON_FLAGS` that string plus
`-DTILEMEGA_EVENT_LOAD_POLL=1`, so the override is installed in both arms and
only the poll spelling differs. Gate the GPU time on a SASS diff between the two
builds taken first: if they come out byte-identical again, the arm is still dead
and no timing should be spent on it. This matters to the round's own arithmetic —
E3-3 drove `notify` down 29.6-34.1% while `wait` rose 17.3-53.6% (F-156), `wait`
is the polling side, and the protocol lever is worth about 3% overall (F-161),
so the `wait` term is where any remainder would have to come from.

## F-163 — The cumulative E3 staircase: three of the four steps are inside ±1%, the release publish carries the protocol, and the whole is worth ~1 pp more than its parts

✅ Verified, `docs/experiments/SYNC_V2/raw_stair{1,2,3,4}`, 25 paired rounds per
cell per step, `full` arm, rotating arm order in fresh processes (H6). Each
directory's `v1` arm adds one switch to the arm before it and its `v0` arm is
the common all-off build, so the column below is cumulative against the round's
baseline rather than against the previous step:

| step | `ON_FLAGS` | adds |
|---|---|---|
| stair1 | calibrated wait policy | E3-0 |
| stair2 | `+ -DTILEMEGA_BARRIER_V2=1` | E3-1 |
| stair3 | `+ -DTILEMEGA_EVENT_SOLO=1` | E3-2 |
| stair4 | `+ -DTILEMEGA_EVENT_RED_PUBLISH=1` | E3-3 |

`OFF_FLAGS` is `-DTILEMEGA_BARRIER_V2=0` in all four, which is the all-off build
— an empty `OFF_FLAGS` would be treated as unset by `run_barrier.sh`'s
`${OFF_FLAGS:-...}` and would silently become `-DTILEMEGA_BARRIER_V2=0` anyway.
Median of per-round L2 ratios, bootstrap CI (seed 20260906, 20000 draws),
Wilcoxon signed-rank p — the estimator of `summarize_barrier.py`:

| cell | stair1 | stair2 | stair3 | stair4 |
|---|---|---|---|---|
| gqa2 s4 | −0.40% | −0.69% | −1.01% | **−5.16%** |
| gqa2 s128 | +0.35% | +0.18% | −0.40% | **−3.48%** |
| mha4 s4 | −0.22% | −0.35% | −0.70% | **−4.79%** |
| mha4 s128 | +0.52% | +0.10% | −0.43% | **−3.74%** |

Ratios and CIs for the last step, which is the one that moves: 0.948403
[0.945293, 0.954545], 0.965186 [0.962500, 0.967263], 0.952133 [0.948212,
0.957125], 0.962600 [0.961669, 0.964508], every p = 1.3e-05. ⚠️ One cell of the
sixteen is not significant: stair2 at mha4 s128, ratio 1.001019 with CI
[0.994791, 1.004476] and p = 0.648. The barrier cut is a null there, not a
regression.

✅ Verified shape: **three of the four steps are inside ±1% cumulative and the
fourth carries the protocol.** After E3-0, E3-1 and E3-2 together the staircase
stands at −1.01 / −0.40 / −0.70 / −0.43 percent; adding E3-3 alone takes it to
−5.16 / −3.48 / −4.79 / −3.74. This agrees in direction and magnitude with
F-156, which measured the release-ordered publish in isolation as the largest E3
step by a factor of five.

✅ Verified, and it is the one place the steps are not independent: the
cumulative result is **larger than the product of the isolated ratios** in all
four cells. F-156's caveat that the isolated arms cannot be multiplied into a
staircase — each ran in its own fresh-process session with its own `l2_off`
baseline — is why this is a measurement and not arithmetic:

| cell | product of isolated (`e3_steps.tsv`) | measured cumulative | difference |
|---|---|---|---|
| gqa2 s4 | −4.38% | −5.16% | −0.78 pp |
| gqa2 s128 | −2.79% | −3.48% | −0.69 pp |
| mha4 s4 | −3.77% | −4.79% | −1.02 pp |
| mha4 s128 | −2.78% | −3.74% | −0.96 pp |

⚠️ Inferred from the sign and the consistency across all four cells: the steps
are mildly super-additive rather than independent — each earlier step removes
work that would otherwise have hidden part of the next one's saving. The effect
is ~1 pp, so it changes no ranking in the round.

✅ Verified cross-check, and it corrects a discrepancy rather than confirming an
expectation. Stair 4 is configuration B's flag set, so it should equal
PLACE_EFT2's B-against-A — and at first reading it does not, −5.16% against
F-161's −2.80% at gqa2 s4. The cause is placement, not the protocol:
`run_barrier.sh` passes no `-DTILEMEGA_PLACEMENT` (`:59`), so every staircase
binary takes `Placement.cuh:23-25`'s default of 0, `legacy_grid_stride`, while
F-161 reads its mechanism sizes at `rotate`. Recomputed on PLACE_EFT2's own
`legacy_grid_stride` arm with the staircase's estimator, the two agree:

| cell | stair4 (SYNC_V2, legacy) | PLACE_EFT2 B/A at legacy | PLACE_EFT2 B/A at rotate |
|---|---|---|---|
| gqa2 s4 | −5.16% | −4.83% | −2.81% |
| gqa2 s128 | −3.48% | −3.14% | −2.87% |
| mha4 s4 | −4.79% | −4.81% | −2.67% |
| mha4 s128 | −3.74% | −3.93% | −3.10% |

Agreement within 0.35 pp in every cell, across two experiments that built their
own binaries and ran in separate sessions, which is also a session-to-session
reproducibility result for the protocol under H6.

✅ Verified consequence: **the protocol lever is worth roughly twice as much on
legacy placement as on rotate**, in absolute terms as well as in percent. At
gqa2 s4 configuration B saves 21.5 µs against a 445.4 µs legacy baseline but
8.2 µs against a 292.9 µs rotate baseline; across the four cells legacy saves
21.5 / 19.3 / 42.8 / 45.8 µs where rotate saves 8.2 / 13.1 / 15.4 / 26.6 µs.

✅ Verified, and it refutes the obvious explanation rather than confirming it.
The natural reading of the line above is that rotate, being the faster
placement, has already removed the stall the protocol collects. Recomputing
B-against-A on all six candidates from `raw/final` — no GPU time, those binaries
and logs already exist — says otherwise. Across the twenty cells that exclude
`balanced` the percentage saving is **uncorrelated** with the baseline, Pearson
r = −0.003 against median L2 under A. (The r = 0.964 that the *absolute* saving
shows across all 24 cells is carried by `balanced` alone, whose baselines run to
3845 µs; drop it and the absolute correlation falls to 0.919 while the
percentage one vanishes.) At gqa2 s4 `rotate`, `eft`, `wavefront` and `chain`
start within 4.2 µs of one another — 292.9, 294.7, 297.0, 295.9 — and the
protocol saves 8.2, 13.1, 13.3 and 15.2 µs respectively. Equal baselines,
savings differing by 1.9x.

✅ Verified, and this is the substantive result: **rotate is specifically
resistant to the protocol, and that is what moves the round's best candidate.**
Rotate is the only candidate saving under 3.2% in every cell (−2.81 / −2.87 /
−2.67 / −3.10); every other candidate lands between −3.79% and −5.12%. Rotate is
first under A in all four cells and loses that lead under B in three of them —
at gqa2 s4 from 1st to 4th of the four (292.9 best under A, 284.7 under B against
chain's 280.6), at mha4 s4 from 1st to 4th, at gqa2 s128 from 1st to 3rd, and
only at mha4 s128 does it stay first. ⚠️ Inferred: this is the mechanism behind
F-161's table changing its winner from rotate under A to chain under B — not that
chain improved, but that rotate collected least from the protocol.

⚠️ Left as a question rather than an answer: *what* rotate does that the others
do not is not established here. It is the only candidate in that group carried as
a host-evaluated placement macro (`TILEMEGA_PLACEMENT=5`) rather than an emitted
Plan — but `legacy_grid_stride` is a macro too (`=0`) and behaves like the
Plan-carrying candidates, so the macro/Plan split is not the explanation either.

**Next step**, and it is the decomposition this entry could not do. The four-arm
attribution in `run_barrier.sh` (`neither`/`nowait`/`full`/`l1nosync`) splits L2
into `wait`, `notify`, `barrier` and `loop`, and it has only ever been run at the
header's default placement, which is why every decomposition in this round
describes legacy. Passing `-DTILEMEGA_PLACEMENT=5` through both `ON_FLAGS` and
`OFF_FLAGS` runs the identical decomposition at rotate without editing the script
(H1), and differencing it against the legacy decomposition already in
`raw_stair4` names the term rotate has removed. The specific prediction to test:
if rotate's `notify` share is already small, the release-ordered publish — which
F-156 measured as the one E3 step that moves — has little left to collect there,
and the protocol's remaining headroom on a placement becomes predictable from
that placement's `notify` share, which is the quantity this entry has just shown
the baseline does *not* predict.

## F-164 — `cp_lb_nosync` stops counting the spine that chaining co-locates, and chain's own queue floor is above the S2r-b target in all four reference cells

`TRACE_V2/analyze.py:201` computes every bound with one path search,
`longest(edge_weight, include_queue)`. A predecessor scheduled on the same
worker is a queue edge (`slot_row[pred]["worker"] == slot_row[s]["worker"]`),
and when `include_queue` is false that edge is skipped outright — the
predecessor's accumulated finish time is dropped, not repriced. The two bounds
this round scores against are `cp_lb_nosync = longest(0, include_queue=False)`
(`:270`) and `queue_lb = max(busy.values())` (`:266`), the busiest worker's own
total task time.

✅ Verified, and it appears as an identity rather than an inference. In
configuration A at gqa2 s4, rotate and chain report the *same* reconstructed
path — `cp_nodes` 20 for both, `cp_split_task_ns` 242688 ns for both — yet
`cp_lb_nosync` reads 242688 ns for rotate and 122880 ns for chain. The task DAG
is identical; what differs is that chaining places the spine on one worker, so
its edges become queue edges and `include_queue=False` discards them. The work
does not leave: chain's 242688 ns of path task time reappears as its `queue_lb`
of 241664 ns, against rotate's 41984 ns. At mha4 s4 the identity is plainer
still — chain's path task time is 488448 ns and its own `queue_lb` is 485376 ns.

⚠️ The consequence is a metric/mechanism interaction, and it runs against the
mechanism this round built: `cp_lb_nosync` is not placement-neutral, and it
under-reports by construction for exactly the placement EX-S2c exists to
produce. R3 §1 defines the ceiling as this quantity. No gate score moves,
because the ceiling used in §5 and in `verify.py` is the larger of the two
bounds — a dropped edge can only lower `cp_lb_nosync`, and `queue_lb` catches
the co-located work. What moves is the reading.

⚠️ This corrects a sentence in `docs/experiments/PLACE_EFT2/summary.md` §5 that
I wrote from this same table: "chaining does cut the critical path — chain's
`cp_lb_nosync` is about half rotate's at both seq 4 cells". It does not. The
halving is the excluded queue edges, and the equal `cp_split_task_ns` above is
the direct evidence. Recorded here rather than silently amended, per CLAUDE.md;
§5 is corrected in place and the ceilings it tabulates are unchanged.

✅ Verified, and it is the arithmetic half of S2r-b's 0/4. Against §7.3's fixed
targets, configuration A's floors are (µs):

| cell | target | chain `queue_lb` | rotate binding bound | admits the target for |
|---|---|---|---|---|
| gqa2 s4 | 218 | 241.7 | 242.7 (path) | neither |
| gqa2 s128 | 328 | 342.0 | 154.6 (queue) | rotate only |
| mha4 s4 | 362 | 485.4 | 244.7 (path) | rotate only |
| mha4 s128 | 528 | 603.1 | 297.0 (queue) | rotate only |

Chain is above target in all four cells with synchronization priced at zero, and
rotate is above it at gqa2 s4. These are traced builds, which F-131 measured at
up to 1.0167x the untraced median; deflating by that worst case leaves chain at
237.7/336.4/477.4/593.2 and rotate's gqa2 s4 path floor at 238.7, so every
exclusion survives — though gqa2 s128 survives by 8.4 µs and should be read as
marginal rather than settled. ⚠️ Only rotate and chain were traced
(`TRACE_ARMS`), so nothing is claimed here about the floors of `eft`,
`wavefront`, `balanced` or `legacy_grid_stride`.

✅ Verified, the measured half, and it is not marginal. Over six candidates x
four configurations x 25 paired rounds, the best median in each cell is 280.6
(chain B), 440.3 (eft B), 557.8 (chain B) and 833.6 µs (rotate B), against
targets of 218/328/362/528 — 1.29x, 1.34x, 1.54x and 1.58x. No candidate under
any configuration reached any target in any cell.

✅ Verified, and it separates the miss into two causes that need different
answers. Where the floor excludes the target the gap is arithmetic and no
protocol work can close it: gqa2 s4 for both traced candidates, and all four
cells for chain — the candidate F-161 ranks best under configuration B. Where
the floor admits the target the gap is overhead: rotate at B sits at 2.82x,
2.29x and 2.74x its own binding bound at gqa2 s128, mha4 s4 and mha4 s128, and
configuration B collects about 3% of it (F-161, F-163). S2r-b needs three of
four cells, and gqa2 s4 is not among the three available to any traced candidate.

**Next step**, and it is a target-setting change rather than a mechanism. §7.3
derives each target from `ceiling_A + 0.5 x (measured_A - ceiling_A)` with
`ceiling_A` taken from `TILEMEGA_PLACEMENT=0` traces, then scores it against a
different Plan's binary; F-163 found the same mismatch on the mechanism side and
§5 on the bound side. The concrete step is to trace the four untraced candidates
in configuration A — `TRACE_ARMS` already parameterizes this and it needs no new
code — and recompute per candidate whether that Plan's own
`max(cp_lb_nosync, queue_lb)` admits the §7.3 target at all. If none does at
gqa2 s4, that cell measures no mechanism this round built, and the next round's
gate should be anchored per candidate, as H8 already requires of the ceiling but
§7.3 does not yet require of the target.

## F-165 — `per_task=2` on the E3-1 gate line is arithmetic on a stated constant, and the three barrier counts in play are not the same unit

✅ Verified, read from the three sources named. R3 §1.2 states **6** CTA
barriers per task. `TileMega_skeleton.md:1075` states
"L2 kernel 在执行器层面每个 task 最多执行 5 次 `__syncthreads()`" — at most **5**
per task. `SYNC_V2/raw_barrier/census.tsv` counts **11** `bar_sync` in
`tilemega_l2_kernel` at `v2=0`, falling to 8 in the plain build and to 10 under
trace, in both models. The first two are dynamic per-task counts; the third is a
static instruction-site census over the whole kernel, including sites outside
the per-task loop. They are not three estimates of one quantity. §1.2's
requirement that the difference be recorded and not reconciled was already met
by `docs/TODO.md`'s EX-E3 row, which names all three and defers to SASS; what
this entry adds is what each number counts, and the consequence below.

✅ Verified: the gate mixes the two units. `verify.py:55-56` sets
`BARRIERS_PER_TASK_BASE = 5` and `BARRIERS_PER_TASK_GATE = 2`; `:203` computes
`drop` as the static census delta and `:204` computes
`per_task = BARRIERS_PER_TASK_BASE - drop`. The `per_task=2` printed on
`E3-1-barriers`' PASS line is therefore the skeleton's *stated* 5 minus a
*verified* static delta of 3. By CLAUDE.md's marking it is stated-minus-verified
rather than verified: no dynamic per-task barrier count was measured this round,
on either arm.

✅ Verified: the static delta of 3 is real, and all three dropped sites are on
the per-task path, so the subtraction is at least dimensionally defensible.
`TILEMEGA_BARRIER_V2` guards exactly four sites in `ModelHarness.cuh`. Three are
dropped in a plain v2 build: `:782`, the `__syncthreads()` closing `NotifyTask`,
once per task; and `:925` and `:943`, which bracket `RunTask` inside the task
loop. Both of the latter carry `#if !TILEMEGA_BARRIER_V2 || TILEMEGA_TRACE_V2`,
so a trace build keeps them and drops only `:782` — which is exactly the 11 → 10
the census reports for the trace variant against 11 → 8 for plain, and the two
trace paths are documented in the source as not interchangeable under v2.

⚠️ Inferred, and the reason the subtraction is not a safe model: the fourth
site, `:583` in `WaitTaskDependencies`, is not a removal. v2 makes that
`__syncthreads()` **unconditional**, where the off arm executes it only when
`task.wait_count != 0`. So v2 does not dominate the off arm site by site — for a
task with no waits the off arm executes no barrier there and v2 executes one.
The `≤ 2` bound is unaffected, since the site is counted in both arms, but a
`BASE - drop` subtraction cannot express a conditional becoming unconditional
and would not have caught it had it gone the other way.

⚠️ Correction, recorded rather than rewritten, per CLAUDE.md. F-152's title and
its sentence "deleting three of eleven barriers" read the census's 11 as a
per-task count; 11 is the static `BAR.SYNC` total of `tilemega_l2_kernel`.
F-152's paired ratios, its correctness counts and its conclusion are untouched by
this — the term it decomposes (`barrier`, 0.037–0.089 ms of a 0.44–1.24 ms
kernel) is measured directly and is not derived from the 11.

Next step: a dynamic per-task count is one counter, not an experiment — an
`atomicAdd` on a per-CTA slot beside each executor `__syncthreads()` under a
trace-only switch, read back per task. That measures the quantity §1.2's 6 and
the skeleton's 5 both name, and would settle the unit question instead of
recording it. Until it exists, `E3-1-barriers` should print the census delta it
verifies and either omit `per_task` or mark it stated.

## F-166 — Chaining loses to rotate on measured L2 in all six cells, and at seq 4 it loses where the model predicted it would win

Evidence: `docs/experiments/CHAIN/raw/summary.tsv` gate `S2c-d`, 25 paired rounds
per cell, each arm's L2 ratioed against `rotate`'s in the same round; correctness
from `raw/correctness.tsv`; predicted makespan from gate `S2c-b` in the same
file. Fresh processes this round (H6), rotating arm order.

✅ Verified, correctness first: 18 arm-cells, 900/900 processes, every cell
50/50 across `legacy_grid_stride`, `rotate` and `chain` at gqa2/mha4/real x
seq 4/128.

✅ Verified, the measured result:

| cell | legacy/rotate | chain/rotate | 95% CI | p | predicted chain/rotate |
|---|---|---|---|---|---|
| gqa2 s4 | 1.5210 | 1.0070 | [1.0035, 1.0104] | 2.861e-05 | 0.988 |
| mha4 s4 | 1.5345 | 1.0114 | [1.0106, 1.0124] | 1.298e-05 | 0.989 |
| gqa2 s128 | 1.3475 | 1.1368 | [1.1368, 1.1388] | 1.263e-05 | 1.067 |
| mha4 s128 | 1.3374 | 1.2581 | [1.2558, 1.2612] | 1.307e-05 | 1.213 |
| real s4 | 1.2996 | 1.0235 | [1.0182, 1.0292] | 3.115e-04 | 1.032 |
| real s128 | 1.1932 | 1.1329 | [1.1226, 1.1332] | 1.307e-05 | 1.135 |
| ALL, n=100 | 1.4763 [1.3517, 1.5209] | 1.0757 | [1.0124, 1.1368] | 8.433e-18 | — |

✅ Verified: **chain is slower than rotate in all six cells**, and the CI excludes
1.0 in all six. The penalty runs 0.70% at gqa2 s4 to 25.81% at mha4 s128. The
`legacy_grid_stride` control is 19.3–53.5% slower than rotate in the same rounds,
so the comparison resolves a placement effect an order of magnitude smaller than
the one it is being asked to see; chain's 0.70% is a measurement, not a floor
artifact.

⚠️ Inferred from pairing the measured and predicted columns, and it is the part
that localizes the failure rather than restating it: the model and the binary
agree at the four seq-128-class cells and disagree in sign at the two seq 4
reference cells. Predicted 1.213 measured 1.2581 (mha4 s128); predicted 1.135
measured 1.1329 (real s128); predicted 1.067 measured 1.1368 (gqa2 s128). But at
gqa2 s4 the solver predicted chaining would win by 1.2% and it lost by 0.70%, and
at mha4 s4 predicted a 1.1% win against a measured 1.14% loss — a swing of about
2 pp in each, carried by the 2 and 3 same-worker critical-path edges those two
Plans place. Those edges are exactly what the mechanism exists to create, and the
cost model books each as a removed hop at zero cost. F-164 shows the lower-bound
side of the same error: `cp_lb_nosync` stops counting the co-located spine
entirely, and the work reappears in `queue_lb` (241.7 µs against rotate's 42.0 at
gqa2 s4). So the same-worker edge is not merely mispriced, it is priced as a
saving when it is a serialization.

✅ Verified, and it bounds how much of this is recoverable by tuning: real s4 at
1.0235 is chain's best cell and is also the cell whose prediction was closest to
neutral (1.032), while `split_count` is 0 everywhere (F-158) and `fill_overflows`
is 0 in four of six cells. The loss is not coming from split repair or from the
capacity cap in the cells where chaining comes closest; it is in the price of the
edge itself.

Next step, and it is one measurement rather than a redesign: at gqa2 s4, where
chain and rotate differ by 1.15 µs of measured L2 and by two same-worker path
edges, read the realized start-to-start delay across each of those two edges out
of the existing trace dumps and compare it against the hop it replaced —
1230.5 ns on sm_89 by F-145's RMW+backoff64 row. If the serialization delay
exceeds the hop, chaining is structurally unprofitable whenever a spine
predecessor's task time exceeds the hop, which is a testable predicate the
extraction can evaluate before it co-locates an edge, and the fix is to refuse
the edge rather than to reweight it. That predicate also says what chaining needs
from an architecture: on sm_120 the hop floor is ~400 ns (F-145's sm_120 fit),
so the task-time threshold below which co-location pays is three times tighter
there, not looser.


## F-167 — Recovering elided task dependencies removes the placement-dependent trace bound

✅ Verified offline in R4, `docs/experiments/TRACE_V2/r4_rebuild/`: all 32 R3
placement dumps and all 12 W=1/2/4 dumps pass the floor inequality and the
nonnegative causal split closure. These are reanalyses of historical traces,
not new R4 performance measurements. Input paths and hashes are retained.

The unsimplified dependency windows are absent from `schedule.tsv`,
`waits.tsv`, and `events.tsv`: the first carries offsets, the second has
already undergone local omission and wait lifting, and the last identifies
publication rows. `analyze.py::dependency_graph` now reads the matching
generated `kDependencies0` table and validates every trace dependency slice.
DAG edges survive even when their endpoints share a worker. Executor edges
are explicitly distinct. All old metrics are retained with `_legacy` suffixes.

✅ Configuration A at gqa2 s4: rotate and chain both recover 20 nodes with
242688 ns task weight. At gqa2 s128 rotate's zero-sync bound changes from
89088 to 346112 ns; at mha4 s128 from 91136 to 697344 ns. The causal
reconstructions for those two traces are 454656/948224 ns against recorded
spans of 455680/948224 ns. Split intervals start after the predecessor's work
completes, so simultaneous waits on different CTAs are not repeatedly charged
to a serial path. The final publication tail is included; root pre-run delay
is excluded. No measured task duration is rescaled.

✅ Fixed-input tests preserve the same bound for W=1/2/4 and retain zero-tick
nodes on tied paths. The residual cross-configuration variation comes from
observed task durations, not W-dependent omission of DAG edges.

✅ Corrected W=4 HOL is lower than W=1 in all four window cells. For example,
gqa2 s4 changes from 483328 to 62464 ns and mha4 s128 from 47318016 to
23089152 ns (sum over workers). The previous claim that the window reclaimed
no HOL relied on lifted waits being mistaken for readiness. The raw paired
E2E_TIME ratios remain unchanged: the analyzer repair explains 0% of the
recorded end-to-end regression. It changes its interpretation, not its timing.
⚠️ Inferred: scan/probe work can still exceed reclaimed stalls. This offline
reanalysis cannot allocate the timing delta between execution overhead and
hardware noise; a new paired ablation is required. W remains 1 by default.

Next step: export the full semantic task DAG with future traces, including
exact-ISL variants, instead of requiring archived generated sources. Reprice
all candidate floors before freezing targets; only rotate and chain have the
32 R3 placement dumps.

## F-168 — R3's publication reduction is relaxed at the atomic instruction

✅ Verified by a new R4 compilation of the gqa2 generated model with R3 B
flags, followed by ptxas and cuobjdump. Evidence:
`docs/experiments/SYNC_V3/premise_audit/compile.json`, `gqa2_b.ptx`,
`gqa2_b.sass`, and the two publication excerpts. `ArriveEvent` emits
`atom.global.add.u64` without a `.sem` qualifier at both publication sites;
ptxas lowers the unused results to `RED.E.ADD.64.STRONG.GPU`. The preceding
`NotifyTask` fence remains `membar.gl` / `MEMBAR.SC.GPU`.

The instruction's default ordering is relaxed when `.sem` is absent, as
specified by [NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf).
This does not say the complete protocol lacks a release sequence: the
per-writer fence supplies ordering before the relaxed arrival. It does say
R4 §1.1's attribution to `red.release` itself is not the emitted mechanism.
The source comment in `ArriveEvent` explicitly describes the relaxed reduction.

⚠️ Inferred: R3's gain may come from eliminating the last-arriver test,
epoch publication and their fences, rather than moving ordering into the
atomic instruction. No new performance gain is claimed from this compilation.
Per R4 §11, the contradicted premise stops further protocol implementation.

Next step: distinguish relaxed arrival plus writer fence, explicit release
arrival plus writer fence, and the proposed single-writer-fence protocol as
separate measured configurations. Run B's five arms first; do not remove a
fence by treating STRONG.GPU as an explicit release qualifier.


## F-169 — R4 reproduces the silent no-barrier control in the single-writer litmus

✅ Verified in 3600 fresh processes on the RTX 4090, 72 cells at 50 processes
each, rebuilt from the unchanged `SYNC_V2/litmus.cu`. Evidence is
`docs/experiments/SYNC_V3/litmus_recheck/`: source/command provenance, raw
per-process logs, occupancy preflight, SASS census and instruction diff.
`per_writer` and `thread0_fence` each pass 900/900, with no launch error or
hang. No correctness claim is derived from a performance-only probe.

✅ All six sensitive cells (acquire fence off, grid 64/128/256, tile
1024/4096) observe no-fence mismatches in 50/50 processes. The no-barrier
control mismatches in 50/50 in five of these cells, but at grid=128,
tile=4096 it passes 50/50. Over its entire matrix it mismatches 848/900;
the other two passes are at acquire=1, grid=128, tile=4096. Therefore R4's
requirement that both negative controls fire in each readable cell is not met.
The legacy runner's PASS is not the R4 acceptance verdict: R4 verify.py reads
each raw RESULT line and reports C1-litmus FAIL, 5/6 sensitive cells.

This reproduces the exception already recorded in F-157, rather than the
stronger premise in R4 §1.1 that both negative controls passed their check.
Per R4 §11, §8.5 is unchanged and no single-writer release is enabled. The
premature C1 implementation from the previous assistant turn was reverted
before this audit; that ordering error is recorded, not erased.

⚠️ Inferred cause: the current no-barrier arm removes both the writer-side
and reader-side barriers but does not force an inter-warp store skew. At this
geometry, the omitted synchronization is not exposed by its schedule. This
is a detector limitation, not evidence that barriers are redundant.

Next step: construct an independently checked delayed-writer witness that
keeps the consumer acquire path fixed while removing only the producer
convergence, retains address reuse and small tiles, and explicitly verifies
that the reader runs before the delayed writer in the negative arm. Freeze
that construction before rerunning all required 50-process cells. Keep the
existing no-fence arm as a separate sensitivity check. Complete B's measured fence pricing before the ordered C1 implementation.
Unseal §8.5 only after that implementation and the required sensitive litmus
checks pass; a repaired detector alone does not complete C1.

## F-170 — The notify fence is a placement-dependent minority of protocol cost

✅ Verified in R4, `docs/experiments/FENCE/raw/paired`: five arms, both default
placement 0 and rotate 5, four reference cells, 25 rotated paired rounds in
one session, 1,000 fresh processes; all 200 full-arm processes pass. Unsafe
arms supply timing only. In gqa2 s4/s128, mha4 s4/s128 order, full-minus-nofence
medians are 15.360/22.528/31.744/56.160 us at default placement and
13.376/16.480/24.576/45.056 us at rotate. Paired fence/protocol medians are
0.2146/0.2857/0.2316/0.3307 and 0.0597/0.0575/0.0576/0.0730, respectively.

✅ Verified: the probe removes only NotifyTask's top-level device fence. Publication,
polling, CTA barriers and atomics remain. Fence/notify ratios can exceed one
at rotate because notify is measured with waits disabled whereas the fence
probe retains waits; these interacting contexts are not silently equated.
All finite signed differences are retained, including noisy negative paired
barrier differences. The full raw records, rather than filtered samples,
determine the median and bootstrap interval.

⚠️ Inferred design consequence, frozen before resumed C1: the single-writer
release can recover only part of a marginal fence delta, not the whole
protocol excess. C2 therefore specializes actual fine/aggregate publication,
and C3 prices the implemented local completion state rather than assuming
that it removes global polls. See `SYNC_V3/design.md` for the design decision.

## F-171 — A sensitive delayed-writer witness permits the optional single-writer release

✅ Verified in R4, `SYNC_V3/litmus_v3/scan`: 1,800 fresh processes, with address
reuse and cooperative writes at grid 64/128/256 and tile 1024/4096. Cache
visibility and producer-barrier sensitivity are separate suites. Every
positive per-writer/thread0-fence cell passes 50/50; every no-fence cache
cell and no-barrier delayed-writer cell mismatches 50/50. The latter keeps
consumer synchronization intact and delays nonpublisher warps on odd CTAs
by a fixed 2,000,000 cycles in every arm. Pilot sensitivity precedes the
formal run; no expected value or stress constant changes during that run.
F-169's insensitive 3,600-process rerun remains as raw evidence.

✅ Verified: C1 is implemented under `TILEMEGA_RELEASE_AFTER_BARRIER=0` by default.
Its enabled order is CTA barrier, thread0 device fence, publication. Four
reference correctness cells pass 50/50 each and six SEQSCAN subset cells
pass 50/50 each, including seq=2048 and past=512. The §8.5 historical rule
is retained with a subsequent v2.1 unsealing note after this fresh litmus.

✅ Verified: SASS/BARRIERS evidence in `FENCE/raw/sass` and `SYNC_V3/c1/sass` shows two
static MEMBAR.SC.GPU sites in either L2 kernel. The notify release moves from
before the CTA barrier to after it under the thread0 predicate. Participation
changes from 128 writer threads to one, or four issuing warps to one for the
128-thread reference CTA. Static instruction count does not decrease; the
prompt's static-count expectation was a dimensional mistake, not a measured
128-fold instruction-count reduction. Paired timing is reported separately
in the completed protocol ablation; correctness does not imply a speedup.

## F-172 — The publishing warp can finish a retained next-slot dependency before waiting

✅ Verified implementation behind default-off `TILEMEGA_ASYNC_PUBLISH`: after CTA writer
convergence, thread0 fences and syncwarp transfers ordering to publishing
lanes. Fine and aggregate arrivals use lanes 0 and 1 at a shared ArriveEvent
call site. R3 B had already removed the trailing CTA barrier, so simply
removing it again would have been an ineffective implementation of C2.

⚠️ Inferred ordering argument: each publishing lane reaches its own event
arrival before entering the next slot's dependency wait. Other warps may
already be waiting, but cannot prevent the publishing warp from finishing.
The CTA convergence before the next RunTask prevents TaskSmem reuse while
publication or another warp's wait is outstanding. Publication uses global
event rows and no TaskSmem field. No Prefetch is introduced; that remains
EX-E4 work. The retained kappa=2 grouped-dependency construction is recorded
in `SYNC_V3/c2_dependency`, rather than assuming sigma omission always holds.

✅ Verified: four reference cells pass 50/50 under C2, and all six SEQSCAN subset cells
pass 50/50. The separate kappa=2 configuration also passes all four reference
cells 50/50. Raw process logs and command/digest sidecars are retained; the
final verifier additionally reconstructs adjacent-slot dependency witnesses
from the trace tables: gqa2 s4/s128 and mha4 s4/s128 contain 68/68/108/140
retained adjacent-slot witnesses. All four trace executions pass. Performance
is measured with the full five-arm matrix.

## F-173 — Shared local completion replaces existing register tracking inside a window

✅ Verified by source inspection corrects the R4 premise: the existing W>1 materializer
already converts eligible same-worker edges to `slot_local_deps`, consumed
by per-thread `done_mask`; those edges are not globally polled. The new
`TILEMEGA_LOCAL_DEP_SMEM` uses shared head/completion words separate from
TaskSmem. Thread0 writes them before NotifyTask's convergence; tasks without
global out-events also converge so all warps can consume the shared state.
The next RunTask's convergence and §5.7.3 legality checks remain intact.

✅ Verified: all four reference cells pass 50/50 at both W=2 and W=4. The corresponding
window controls also pass 50/50. Static L2 MEMBAR.SC.GPU counts are three for
window/shared variants, while BAR.SYNC sites are eight for the window
control and ten for the shared variant, reflecting initialization and the
no-global-event completion path. These counts are not per-task dynamic
barrier counts. The five-arm matrix measures the net effect of those costs;
W remains opt-in and is not made the default.

## F-174 — Cluster-local arrivals preserve GPU forwarding and need target-hardware validation

✅ Verified implementation under default-off `TILEMEGA_CLUSTER_ARRIVE`: DSMEM fan-in uses
`atom.acq_rel.cluster.shared::cluster.add.u64` only where both the cluster
fan-in layout and `caps.cluster` are active. The local last arrival still
performs the GPU-scoped fence and forwarding publication needed by consumers
outside that cluster. Global event visibility is not silently reduced to a
scope that excludes a consumer.

✅ Verified: the previously forbidden RED/shard combination is now composable. A closed
shard contributes one global reduction per iteration, and RED consumers use
the number of nonempty shards as their monotonic trigger multiplier. This
composition passes all four reference cells 50/50 on sm_89 using global
shards. No counter resets or epoch changes were introduced.

✅ Verified: `cluster_compile/` contains successful sm_89/sm_120 PTX and cubin compilation;
only sm_120 PTX contains the cluster-scope arrival instruction. Both complete
reference models also compile for sm_120 with C1/C2, RED, cluster fan-in and
the new scope enabled (`cluster_model_sm120_composed/`). On sm_89, enabling
the new scope switch with `caps.cluster=false` gives byte-identical complete
SASS to C2 at both placements (`cluster_degenerate/sass/`). The final H2
artifact independently regenerates this fallback proof at the sealed head.

⚠️ Stated execution limit: no sm_120 kernel has run on this sm_89 workstation.
`SYNC_V3/run_sm120.sh` regenerates target-local runtime Plans and rotates
cluster_off/on with all other mechanism/probe arms in one paired session.
Its CPU SELF_CHECK passes; compilation and CPU checks do not establish the
cluster mechanism's hardware correctness or performance. That specific
validation remains for the target machine and does not block the sm_89 work.

## F-175 — The window no-wait probe had retained event reads

✅ Verified by source and SASS in `SYNC_V3/window_probe_fix`: with W>1,
UNSAFE_NO_EVENT_WAIT bypassed the blocking wait but not
ProbeTaskDependencies. Its nowait/neither timings therefore still included
look-ahead event polling and a CTA reduction. The initial partial ablation
was stopped and retained as `ablation_pre_probe_fix/`; no required cell had
25 complete rounds and no final performance conclusion uses that session.

✅ The repaired unsafe path returns ready before reading events. The gqa2
W=2 nowait L2 kernel loses one BAR.RED and one ATOMG polling site. Both
reference safe W=2 kernels remain byte-identical after recompilation under
the same source paths. The initial control comparison differed only in
source identifier headers; the binaries were rebuilt using the original
paths instead of stripping evidence lines. The final verification log and
raw SASS are committed with the repair.

✅ All 32 reference-window and four real-width-window unsafe binaries were
rebuilt. Full, nofence and l1nosync images are unchanged; B pricing and the
safe correctness matrices remain applicable. The complete seven-configuration
matrix is restarted in one fresh paired session, retaining the same 25-round
coverage, attribution equations and research threshold. Independent chain
and frozen-candidate work continues while this dependent item is repaired.

## F-176 — Cost-aware extraction must also price the queue placement it creates

✅ Verified in R4: the strict extension test uses the calibrated hop curve and the successor task weight from the same cost-model source as extraction. Equality rejects; capacity formulas and hard Plan legality checks remain. The default-off implementation also ranks ready tasks by remaining critical work, prices sibling-SM sharing in finish_on, and breaks equal finishes toward fewer path hops. Four existing feedback rounds are fixed for the selected recipe, with a cost-off four-round matched control and the original zero-feedback chain retained. See CHAIN2/design.json and README.md.

✅ The isolated price-only variant with the same four feedback rounds still gives 40 hops and 597673.9223 ns on mha4 s128. The completed recipe gives 35 hops and 446326.144 ns, against rotate at 35 hops and 437123.7068 ns. These are simulator path-record results, independently counted from CHAIN2/final/replay/path and CHAIN2/diagnostic/minimal/path. Replayed materialized chain sources match the measured Plans byte for byte. Four reference correctness cells pass 50/50.

✅ Fresh-process paired GPU measurements (25 rotated rounds, configuration A):

| cell | arm | L2 ms | ratio to rotate | 95% CI |
|---|---|---|---|---|
| gqa2 s4 | rotate | 0.290816 | 1.00000 | [1.00000, 1.00000] |
| gqa2 s4 | chain | 0.291616 | 1.00011 | [1.00000, 1.00330] |
| gqa2 s4 | chain_control | 0.292640 | 1.00363 | [1.00296, 1.00671] |
| gqa2 s4 | original | 0.292864 | 1.00671 | [1.00406, 1.00749] |
| gqa2 s128 | rotate | 0.455680 | 1.00000 | [1.00000, 1.00000] |
| gqa2 s128 | chain | 0.500896 | 1.09930 | [1.09888, 1.10112] |
| gqa2 s128 | chain_control | 0.514912 | 1.12986 | [1.12817, 1.13034] |
| gqa2 s128 | original | 0.518144 | 1.13708 | [1.13687, 1.13708] |
| mha4 s4 | rotate | 0.576512 | 1.00000 | [1.00000, 1.00000] |
| mha4 s4 | chain | 0.575488 | 0.99972 | [0.99768, 1.00813] |
| mha4 s4 | chain_control | 0.581632 | 1.01377 | [1.01101, 1.01593] |
| mha4 s4 | original | 0.549856 | 1.01223 | [1.01040, 1.01399] |
| mha4 s128 | rotate | 0.855040 | 1.00000 | [1.00000, 1.00000] |
| mha4 s128 | chain | 0.967680 | 1.13119 | [1.13043, 1.13291] |
| mha4 s128 | chain_control | 1.077248 | 1.25988 | [1.25854, 1.26127] |
| mha4 s128 | original | 1.076224 | 1.25796 | [1.25625, 1.26005] |
| real s4 | rotate | 4.407232 | 1.00000 | [1.00000, 1.00000] |
| real s4 | chain | 5.107776 | 1.15996 | [1.15881, 1.16875] |
| real s4 | chain_control | 4.500480 | 1.02139 | [1.02045, 1.02367] |
| real s4 | original | 4.500480 | 1.02115 | [1.02053, 1.02325] |
| real s128 | rotate | 6.499328 | 1.00000 | [1.00000, 1.00000] |
| real s128 | chain | 7.163776 | 1.10213 | [1.10194, 1.10242] |
| real s128 | chain_control | 7.364608 | 1.13293 | [1.13263, 1.13345] |
| real s128 | original | 7.363584 | 1.13280 | [1.13235, 1.13315] |

✅ Unique excluded cost-model DAG edges: gqa2 s4/s128 412/538124, mha4 s4/s128 908/2127388, real s4/s128 25952/34394336. Exact hop/queue prices and multiplicities are retained in final/on/rejected_extensions.tsv. These count unique edges in the selected pass, not repeated DP visits. Reference worker-SM maps come from the existing traced calibration; real-width uses the modulo model.

✅ The replayed critical path still adds queue edges: gqa2 s128 10→20 and mha4 s128 32→55, despite unchanged hop counts 17/35. Real s4 path task weight rises from 3718.801 to 4722.158 us. These are direct node/edge records.

⚠️ Inferred next correction for slower cells: finish_on still approximates sharing and does not propagate all downstream queue blocking. free_ns tracks task weights without separately pricing NotifyTask work still executed on those queues; a fresh phase trace must quantify that contribution. Price extension by the simulator makespan increment with affected queue edges, rather than only the successor weight. The hop test remains part of that redesign; reverting it is not the outcome. No ChainDP change is included.

## F-177 — Completed-protocol gains and shared-window costs are measured together

✅ Verified from SYNC_V3/ablation: seven configurations, two placements, four reference cells and five probes, 25 rotated rounds in one session (7000 fresh processes). The required real-width seq=4 legacy matrix adds five configurations and 625 fresh processes. The superseded partial matrix with the window probe coverage hole is excluded. All full processes pass.

| cell | paired contrast | L2 ratio | 95% CI | latency reduction |
|---|---|---|---|---|
| gqa2_s4_p0 | c1/baseline | 1.00881 | [1.00527, 1.01006] | -0.88% |
| gqa2_s4_p0 | c2/c1 | 1.00000 | [0.99760, 1.00262] | 0.00% |
| gqa2_s4_p0 | local2/window2 | 0.96364 | [0.96347, 0.96568] | 3.64% |
| gqa2_s4_p0 | local4/window4 | 0.96373 | [0.96136, 0.96593] | 3.63% |
| gqa2_s4_p0 | local2/c2 | 1.01562 | [1.01199, 1.01687] | -1.56% |
| gqa2_s4_p0 | local4/c2 | 1.02102 | [1.01906, 1.02190] | -2.10% |
| gqa2_s4_p5 | c1/baseline | 0.99719 | [0.99638, 1.00000] | 0.28% |
| gqa2_s4_p5 | c2/c1 | 1.00690 | [1.00385, 1.00722] | -0.69% |
| gqa2_s4_p5 | local2/window2 | 0.95449 | [0.95139, 0.95486] | 4.55% |
| gqa2_s4_p5 | local4/window4 | 0.95444 | [0.95163, 0.95486] | 4.56% |
| gqa2_s4_p5 | local2/c2 | 0.98208 | [0.98170, 0.98466] | 1.79% |
| gqa2_s4_p5 | local4/c2 | 0.98214 | [0.98195, 0.98566] | 1.79% |
| gqa2_s128_p0 | c1/baseline | 1.00359 | [1.00344, 1.00376] | -0.36% |
| gqa2_s128_p0 | c2/c1 | 1.00000 | [0.99989, 1.00166] | 0.00% |
| gqa2_s128_p0 | local2/window2 | 0.97577 | [0.97433, 0.97581] | 2.42% |
| gqa2_s128_p0 | local4/window4 | 0.97428 | [0.97393, 0.97585] | 2.57% |
| gqa2_s128_p0 | local2/c2 | 1.03419 | [1.03359, 1.03424] | -3.42% |
| gqa2_s128_p0 | local4/c2 | 1.03590 | [1.03419, 1.03628] | -3.59% |
| gqa2_s128_p5 | c1/baseline | 1.00187 | [1.00000, 1.00230] | -0.19% |
| gqa2_s128_p5 | c2/c1 | 1.00000 | [0.99971, 1.00231] | 0.00% |
| gqa2_s128_p5 | local2/window2 | 0.97807 | [0.97780, 0.97992] | 2.19% |
| gqa2_s128_p5 | local4/window4 | 0.97817 | [0.97598, 0.97821] | 2.18% |
| gqa2_s128_p5 | local2/c2 | 1.02507 | [1.02480, 1.02588] | -2.51% |
| gqa2_s128_p5 | local4/c2 | 1.03189 | [1.02989, 1.03226] | -3.19% |
| mha4_s4_p0 | c1/baseline | 1.00485 | [1.00330, 1.00727] | -0.49% |
| mha4_s4_p0 | c2/c1 | 1.00347 | [1.00000, 1.00478] | -0.35% |
| mha4_s4_p0 | local2/window2 | 0.96453 | [0.96233, 0.96577] | 3.55% |
| mha4_s4_p0 | local4/window4 | 0.96343 | [0.96130, 0.96465] | 3.66% |
| mha4_s4_p0 | local2/c2 | 1.01166 | [1.00957, 1.01304] | -1.17% |
| mha4_s4_p0 | local4/c2 | 1.01424 | [1.01202, 1.01452] | -1.42% |
| mha4_s4_p5 | c1/baseline | 0.99818 | [0.99636, 0.99994] | 0.18% |
| mha4_s4_p5 | c2/c1 | 1.00000 | [0.99818, 1.00182] | 0.00% |
| mha4_s4_p5 | local2/window2 | 0.95288 | [0.95280, 0.95455] | 4.71% |
| mha4_s4_p5 | local4/window4 | 0.95288 | [0.95132, 0.95459] | 4.71% |
| mha4_s4_p5 | local2/c2 | 0.99801 | [0.99635, 0.99818] | 0.20% |
| mha4_s4_p5 | local4/c2 | 0.99971 | [0.99636, 1.00143] | 0.03% |
| mha4_s128_p0 | c1/baseline | 1.00592 | [1.00443, 1.00761] | -0.59% |
| mha4_s128_p0 | c2/c1 | 1.00087 | [0.99916, 1.01203] | -0.09% |
| mha4_s128_p0 | local2/window2 | 0.97488 | [0.97261, 0.97951] | 2.51% |
| mha4_s128_p0 | local4/window4 | 0.97715 | [0.97418, 0.98341] | 2.29% |
| mha4_s128_p0 | local2/c2 | 1.04291 | [1.02774, 1.05428] | -4.29% |
| mha4_s128_p0 | local4/c2 | 1.04626 | [1.04203, 1.05428] | -4.63% |
| mha4_s128_p5 | c1/baseline | 1.00980 | [0.98191, 1.04833] | -0.98% |
| mha4_s128_p5 | c2/c1 | 0.99768 | [0.97232, 1.00577] | 0.23% |
| mha4_s128_p5 | local2/window2 | 0.97864 | [0.95036, 1.00335] | 2.14% |
| mha4_s128_p5 | local4/window4 | 0.97362 | [0.95005, 1.02558] | 2.64% |
| mha4_s128_p5 | local2/c2 | 1.03815 | [1.00868, 1.05963] | -3.81% |
| mha4_s128_p5 | local4/c2 | 1.03443 | [1.00112, 1.06433] | -3.44% |

✅ Static instruction counts remain separate from these measured gains: C1 changes notify-fence participation from 128 threads to one while static L2 MEMBAR.SC.GPU remains two. Window variants have three MEMBAR sites; shared variants have ten BAR.SYNC sites versus eight for the register-state window control. W remains disabled by default.

✅ The neither probe also retains local completion convergence: its W=2 L2 SASS has six BAR.SYNC sites in the register control and eight in the shared variant, for both models (local_probe_sass/). The report lists paired full and neither changes separately. The registered full-minus-neither marginal excludes local state-management work retained by neither; a smaller research ratio must not be presented as the same percentage of full-kernel speedup. The gate equations are unchanged.

⚠️ Inferred remaining window cost is localized in ProbeTaskDependencies and WindowAcquireSlot: candidate wait scans and a CTA reduction per probe remain even after local completion is shared. The next design should batch candidate readiness reductions and retain all four Plan legality checks, rather than assume that shared flags remove pre-existing global polls.

## F-178 — The registered default-placement protocol target remains the research test

✅ Verified from the fresh paired matrix. Every configuration is evaluated on all four default-placement cells; one configuration must achieve at least three. The representative maximizes achieved cells and then minimizes geometric mean protocol/barrier, with fastest end-to-end time reported separately. The ratio threshold 1, coverage, 25 rounds and bootstrap rule were not relaxed. research_rule.json was committed before the corrected matrix session.

| cell | configuration | historical ratio | current ratio | 95% CI | historical excess gap closed |
|---|---|---|---|---|---|
| gqa2 s4 | local2 | 2.23 | 1.36253 | [1.31362, 1.37320] | 70.53% |
| gqa2 s128 | local2 | 2.49 | 1.80645 | [1.78648, 1.83317] | 45.88% |
| mha4 s4 | local2 | 2.1 | 1.27840 | [1.18731, 1.34628] | 74.69% |
| mha4 s128 | local2 | 2.91 | 1.79608 | [1.57021, 2.31856] | 58.32% |

✅ Achieved cells: 0/4; required: 3/4. The full five-arm and L1 ratios, including other configurations, remain in ablation.tsv and research_all.tsv. Negative gap closure is retained.

⚠️ Inferred residual causes: WaitTaskDependencies still issues the acquire fence from all threads when waits exist; EventPoll still uses atomicAdd(ev,0) with EVENT_LOAD_POLL=0; ArriveEvent still updates globally visible counters. The next concrete steps are an acquire-fence price probe and sensitive cooperative-acquire litmus, plus the existing load-poll ablation under WAIT_POLICY=1 with a nonempty SASS diff (F-162). The SOLO direct-epoch path is also disabled in the RED combination; single-member arrivals publication needs its own monotonicity/visibility check. Joint kappa/placement search must price both fewer global events and added readiness/queue delay. The unsafe full-minus-neither contrast includes dependency serialization as well as synchronization instructions; it is not a sum of fence opcodes.

## F-179 — Frozen ceilings stay attached to the original candidate Plans

✅ All 24 frozen targets reproduce from their candidate-specific configuration-A traces and none is below its own floor. The three W=1 protocols are measured on every unchanged frozen candidate; target_positions/positions.tsv records their positions. Balanced uses placement 4, matching the frozen trace; its earlier unused placement-0 compile is archived and was not measured. The new Chain2 Plan is reported separately and does not inherit the old chain floor. No target entry changes after commit 03053089.

✅ 4/24 fixed candidate/cells are at or below their frozen target in the best measured W=1 protocol. These are fresh untraced timings against trace-observed floors, not proof of a target-independent hardware lower bound.

## F-180 — R5 reproduces the mean path cost but finds a different measured tile

✅ Verified in eight fresh processes using digest-checked R4 configuration-A
trace executables; `PHASE/audit/` retains every raw log, dump and build digest.
`PHASE/audit.py` uses the R4 corrected DAG path. The gqa2/mha4 seq=4 paths
contain 20/40 nodes. At rotate their mean durations are 12.186/12.288 us,
p50 4.096/4.096 us, p90 23.552/23.552 us and maxima 43.008/43.008 us.
All-node means are 9.114/7.966 us, with p50 3.072/2.048 us. At seq=128,
rotate path means are 17.510/17.382 us, p50 24.576 us, and maxima
53.248/54.272 us. Every placement's complete distribution is in durations.tsv.

✅ The actual generated baseline is BF16 128x128x16, stages=3, split=1;
F-128's 16x64x16s2k16 example is not the source used by R4's floor table.
Fresh rotate measured/floor values are 1.20588/1.32721/1.18333/1.37555
(gqa2 s4/s128, mha4 s4/s128). The motivating magnitude is reproduced,
while interpreting the mean as every node's duration is incorrect.

⚠️ Stated before measurement: R5 FORK5 uses the four rotate reference cells;
legacy and real-width are separate diagnostics. S3-b uses a fresh rotate +
R3 B protocol control; the prompt's 0.7129 ms is legacy placement's historical
geometric mean. No threshold or required coverage changes.

## F-181 — Task phases freeze R5's search direction before implementation

✅ Verified from twelve new phase dumps (two models and real-width, two
sequences, legacy/rotate), with raw ns and cycle boundaries retained in
`PHASE/raw/trace/`. The independent script prints exactly:

```
FORK5 rule=3 load_share=0.006 fixed_share=0.232 math_share=0.763 cells=4
```

⚠️ Stated decision: rule 3 excludes EX-E4 this round and directs EX-S3 to
joint tile/split/kappa/residency selection. This line and its raw evidence are
committed before the first EX-S3 implementation. The four rotate reference
cells determine the median; legacy and real-width do not change the rule.

⚠️ Inferred limitation: mainloop includes interleaved arithmetic, copies and
waits, so 76.3% does not establish arithmetic saturation. SIMT loops have no
separable first-operand prologue; their load_wait is zero by the documented
boundary contract, not by a claim of free memory access. Epilogue includes the
existing executor barrier tail, also exported separately. The measured first
prologue exposure permits at most about 0.6% median path reduction if eliminated
alone; deeper operand stalls would require a different experiment. Real-width
instrumentation perturbation is being measured separately and is not hidden
inside the reference-cell fork.

## F-182 — Phase instrumentation closes the task interval without changing the fork

✅ Verified: the phase-enabled reference builds pass 50/50 fresh processes in
all four cells. In twelve placement/model/sequence pairs, the largest reference
median on/off ratio is 1.045860, below 1.05. All 135568 original-matrix nodes have
nonnegative phases and exact four-part closure against run_end-run_begin.
`PHASE/raw/{correctness,measure,trace}` contains the raw evidence; ns and
clock64 boundaries are both retained. The independently compiled sm_120 phase
body passes compilation on the 4090, with no sm_120 execution claimed.

✅ The original four rotate reference cells keep the frozen F-181 rule. The
full CP/all-node phase tables and operand-byte associations are reproduced by
`JOINT/collect.py`. Phase-only builds omit event-line timestamps: the retained
historical hop columns are unavailable, explicitly marked by
`phase_event_timestamps_available=0`; they are not used as phase costs.
The initial real-s128 rotate snapshot is slower than its repeated timing cohort;
a single snapshot is not a robust estimate of the target's median task cost.

⚠️ Inferred: first-prologue exposure is small; it does not bound stalls inside
the pipelined mainloop. The four phases partition time, not independent
hardware bottlenecks. SIMT load_wait=0 is a structural boundary convention.

## F-183 — Fast bounds fit the budget but do not preserve the required ranking

✅ Fresh CPU evaluations in `SIMULATOR/r5/evaluations.tsv`: full simulation's
worst reference/real-width times are 140259.531/138348.284 us, failing the
1000/10000 us gates. Coarse evaluation's maxima are 151.007/619.341 us;
preparation is separately timed. On the historical 18-point calibration set,
coarse Spearman is 0.561240985, full Spearman 0.880288958, and actual top1 has
coarse rank 1. `ranks.tsv` retains the k=1..5 cost/quality tradeoff. Top-k=3
includes boundary ties, preserving uncertain candidates instead of resolving
ties using GPU measurements. The simultaneous budget/ranking gate fails.

✅ Replaying 68 R3/R4 dumps yields absolute relative errors p50 0.045399639,
p90 0.108475523 and maximum 0.138277099. The 24 windowed dumps use their
observed execution order as a FIFO approximation; they are not a window
execution prediction. The remaining 44 are W=1. Raw inputs and per-evaluation
CPU timings remain under `SIMULATOR/r5/`.

⚠️ Inferred model parameters, not instruction latencies: publication
1074.1284026707756 ns and consumer wait 1764.3682109690692 ns are fitted from
R4 marginal probes and corrected paths, retaining primitive hop
1235.411801 ns. Actual runtime stage-wide publishing flags are used; a minimal
cross-consumer mask is an idealization, not the current publisher set.
Trace-observed task weights disable a second occupancy stretch. Historical
replay is an explanatory check and does not establish unseen-tile accuracy.

⚠️ Stated degradation: EX-S3 enumerates the full requested axis catalog, but
projects three distinct priority geometries plus L1/R4 seeds, with all retained
kappa/residency combinations and all six placements. Capacity deferrals are
explicit in screen.tsv; they are not renamed bound pruning. Unmeasured catalog
points have no empirical percentile, so S3-c cannot be certified from top-3.

## F-184 — Joint configuration selection shortens task cost and uses more width

✅ Four reference cells achieve S3-b, using 25 new rotated confirmation rounds
after a separate five-round pilot and frozen choice. Selected/control median
ratios and bootstrap 95% CIs (gqa2 s4/s128, mha4 s4/s128) are
0.409039 [0.405965,0.413588], 0.549580 [0.548132,0.550373],
0.460194 [0.458923,0.462054], 0.553799 [0.550798,0.559244].
All four selected configurations also pass 50/50 fresh correctness processes.
The control is the same-session rotate + R3 B protocol, not historical 0.7129 ms.

✅ Selected tile/split/kappa/CTA-per-SM/placement:
32x16x64s2/1/1/1/EFT; 32x16x32s2/1/1/5/EFT;
32x16x64s2/1/1/1/EFT; 32x16x16s2/1/2/5/EFT.
Corrected CP lengths change 228352→89088, 320512→190464,
452608→185344, 645120→393216 ns. Node counts change 20→20,
20→19, 40→40, 40→40. queue_lb/CP changes
0.1704→0.9425, 0.4473→0.8871, 0.1719→0.9061, 0.4270→0.9271.
`JOINT/comparisons.tsv` and the raw trace directories preserve both graphs.
This is primarily cheaper nodes and broader execution, not removal of stages.

✅ Fresh ChainDP's uniform L1 seed is 32x16x16s2 split1, not the R4
128x128x16s3 split1 baseline. Relative to the same-cell top1 L1-geometry arm,
selected/top1 is 0.829630, 0.948498, 0.847684 and 0.944559; their 95% CIs all
exclude 1. The last contrast changes kappa, not tile geometry. Per-operator
L1 DP also chooses split4/8 in some GEMMs; its complete fresh output is in
`PHASE/chain_dp/`. Thus the gain against R4 must not all be attributed to the
L2 objective: adopting a better geometry accounts for part of it.

⚠️ Inferred mechanism: at seq4, the real baseline M=128 leaves only 4/128
logical rows active in a full tensor-core tile; M=32 reduces this padding.
N=16 distributes independent output tiles more widely. K=64 reduces the
number of mainloop iterations in the short-sequence winner. The current
70%-area/30%-operand-size extrapolator does not price K-loop fixed cost and
predicts K16/32/64 ties that fresh measurements separate. Calibrating per-K
iteration setup/copy/wait costs is the concrete next model change.

## F-185 — Numerical admissibility is a configuration constraint at real width

✅ The initial real-s4 32x128 split8 shortlist and real-s128 128x128 split2
shortlist fail one CPU-BF16 golden element, while L0.5/L1/L2 agree exactly.
Both original control cells pass. These are new-configuration failures, not
regressions in an existing configuration or evidence of a synchronization bug.
`JOINT/raw/real_s*/numeric_diagnostic/details` reproduces the offending values:
s4 index3224 is 1.34375 versus 1.3046875 (delta0.0390625, tolerance0.0368750021);
s128 index6597 is -0.138671875 versus -0.118652344
(delta0.0200195312, tolerance0.0178984385). No tolerance or golden changes.

✅ The real-s4 expanded split sweep admits 1/2/16/32 and rejects 4 in one
fresh process each; these probes are not 50/50 claims. Geometry exclusions
reference raw failed logs and apply independently of placement, since even
L0.5 fails. Rejected pilot/build evidence remains in rejected_split8 and
rejected_split2. New admissible shortlists get fresh pilot/confirmation cohorts.

⚠️ Inferred cause: FP32 split partial accumulation followed by BF16 rounding
changes the arithmetic reduction order and propagates through later layers.
`GemmCombineTaskBody` already preserves rounding before residual addition;
removing that boundary would not be a valid repair. The next extension is
per-GEMM split admission: locate the first diverging stage, keep it unsplit,
and retain parallel splits at stages whose measured error stays admissible.
This is a specific search constraint to solve, not a conclusion that split-K
has no benefit.

## F-186 — Real-width selection moves the active floor to worker queues

✅ Final admitted real-s4 and real-s128 selections each pass 50/50 fresh
processes. Their 25-pair selected/control ratios are 0.873807459
[0.872940156,0.874120290] and 1.009736191
[0.997029998,1.039465695]. The latter does not establish improvement.
Selected configurations are 32x16x16s2 split1/kappa1/residency2/EFT and
64x128x16s2 split1/kappa1/residency2/wavefront. Raw pilot, frozen choices,
confirmation and correctness are under `JOINT/raw/real_s{4,128}/`.

✅ CP lengths change 4661248→2136064 ns (40→36 nodes) and
4775936→4687872 ns (37→40 nodes). Selected queue bounds are
3757056/6644736 ns, exceeding CP by factors 1.758869/1.417431.
Thus the active floor becomes queue work, unlike the reference cells.
Selected L2/own-L1 is 0.917369/0.949472 and L2/own-floor is
1.098392/1.049468. These floors use fresh traced task costs and are not
geometry-independent hardware limits. The s128 shortlist needs new valid
64x128 K32/K64 split1 measurements and explicit K-loop/residency pricing;
those proposed alternatives have not been measured in this round.

## F-187 — Selected phase costs and same-geometry split diagnostics

✅ Supplemental selected CP setup/load/mainloop/epilogue fractions are:
gqa2 s4 0.162500/0.062500/0.675000/0.100000;
gqa2 s128 0.100529/0.031746/0.793651/0.074074;
mha4 s4 0.163743/0.064327/0.643275/0.128655;
mha4 s128 0.094488/0.013123/0.800525/0.091864;
real s4 0.015707/0.004760/0.943360/0.036173;
real s128 0.008827/0.004646/0.951220/0.035308.
`JOINT/selected_phases.py` derives these from new private-slot dumps; they
do not rewrite the pre-search FORK5. Mainloop includes copies and waits,
so its dominance is not proof of arithmetic saturation.

✅ At fixed 32x128x64s2 and 20480 first-tile operand bytes, the admitted
real-s4 split1/16 diagnostic gives mean GEMM load_wait 3334.095/2951.381 ns
and 8262.421/7369.500 cycles. CP load shares are 0.010402/0.077406.
Both snapshots pass correctness. Splitting changes the DAG and shortens
individual task work, increasing relative first-load exposure despite a
smaller mean wait. This association is not an independently confirmed
end-to-end performance comparison. Raw `operand_probe/` and derived
`operand_split_probe.tsv` retain the evidence.

## F-188 — Exact streaming repairs preparation without weakening legality

✅ Graph-only runtime projection can defer unused wait cardinality, but
`AttachProjectedEventMetrics` rejects an uncounted projection. Default counted
projection is unchanged. The runtime-projection and execution-simulator tests
pass; streamed small-cell DAGs, weights, predictions and all six generated
sources match the earlier implementation byte-for-byte. Exact cache reuse
requires identical dependency relation text and stage counts; a seq8 mismatch
is rejected. CPU evidence is under `JOINT/self_check/`.

✅ MHA seq2048 preparation emits 262144 nodes and a 9806 MB textual DAG.
Coordinate-pair materialization previously exhausted host memory; streaming
integer adjacency and exact relation reuse let preparation complete. The two
selected geometries' DAG byte hashes agree, while placement, residency and
kappa are re-solved independently. Sorted adjacency run encoding is lossless:
each original is decoded and SHA256-checked before bulky text is replaced.
This is storage compression, not graph sparsification. Initial failed/aborted
attempts are retained in the raw preparation directories.

✅ sm_120 runners pass SELF_CHECK on 4090, including compute-capability and
inherited-environment rejection, disk-budget failure checks, parser checks,
and compilation of contention plus a full phase-instrumented model object.
No sm_120 GPU execution occurred. Target scripts export inputs, calibrate and
solve Plans locally. Publication fitting now separately collects TRACE_V2
and rejects phase-only zero event timestamps; the historical fitted parameters
reproduce exactly after this guard. A different target FORK5 is recorded, and
rule1 explicitly leaves conditional prefetch unresolved instead of claiming it.

## F-189 — Final R5 acceptance distinguishes speedup from search-model completion

✅ The selected four reference configurations pass 50/50 each, both selected
real-width configurations pass 50/50 each, and all twelve re-solved SEQSCAN
cells pass 50/50 (600 fresh processes total). The independent phase builds
also pass four reference cells at 50/50. No golden tolerance was changed.
The final raw verifier completes every gate before returning its failure code;
its full output and evidence paths are in `JOINT/verify_output.txt`.

✅ Phase closure checks cover 135568 original and 113544 supplemental nodes,
with no negative component and zero maximum relative closure error. Default
OFF SASS is byte-identical for both reference models. The final evidence
manifest names the last source/documentation commit, and the direct child
contains only SASS identity artifacts. sm_120 execution remains pending by
explicit round scope; both target scripts complete 4090 SELF_CHECK.

⚠️ Stated incomplete gates: S1c-a (full-simulator time), S1c-b (coarse ranking)
and S3-c (empirical full-catalog top-3% coverage) remain FAIL. The verifier
returns 1; no gate is weakened. S3-b independently passes 4/4 reference cells.
Real-s128's confidence interval includes parity. The reduced candidate set,
K-loop extrapolation error, numerical exclusions and the real-width queue
floor are concrete remaining work, not claims of a completed optimal search.
The pilot-frozen real-s4 choice is retained even though another kappa ranks
slightly faster in confirmation; selection is not changed after seeing it.

## F-190 — R6 K-loop waits distinguish K16, K32 and K64

✅ Verified: the optional `TRACE_KLOOP` probe retains every R5 phase column,
adds no synchronization, and writes accumulated loop counters only from
thread0 after the loop. All six selected R5 configurations pass 50/50 fresh
processes (four reference cells and two real-width cells). Raw logs and
slot-private stamps are in `COSTMODEL/raw_kloop/`; `analyze_kloop.py` checks
that loop body + exposed wait + outside-loop fixed work exactly equals each
instrumented GEMM mainloop in clock cycles. The existing cp.async wait and
CTA rendezvous are timed together, so this is operand availability latency,
not a claim about pure DRAM latency.

✅ Verified, on corrected-path GEMM mainloops: exposed-wait shares are
0.164841/0.367150/0.188533/0.529201 for gqa2 s4/s128 and mha4 s4/s128.
The median loop-iteration costs are 494.574/519.355/504.267/318.945 ns;
median outside-loop fixed costs are 84.156/219.341/52.239/67.038 ns.
Real-width s4/s128 exposed-wait shares are 0.638735/0.420782, with
266.040/538.262 ns per iteration and 52.492/102.693 ns outside-loop fixed
cost. These absolute values use each task's measured ns/cycle ratio.
The paired geometric alternatives remain separate raw observations, not an
end-to-end speedup claim.

```
FORK6 rule=1 mainloop_exposed_wait_share=0.278 kloop_fixed_share=0.014 cells=4
```

⚠️ Scope limitation: this line partitions instrumented GEMM K-loops. SIMT
mainloops interleave loads and arithmetic and are retained explicitly as
unmeasured by this extension. Dividing measured GEMM wait by all CP mainloop
cycles instead gives a 0.161782 median lower bound; no SIMT wait is imputed.
Thus the GEMM-scoped rule is not a verified whole-task-pipeline rule. The
R7 decision must carry this measurement limitation rather than treating the
unmeasured SIMT region as zero-latency arithmetic.

## F-191 — R6 unifies semantic task costs without resolving coarse ranking

✅ Verified: `NonGemmStageNs` has no remaining caller. `Evaluate`, `ChainDP`
and `CouplingInterfaceDP` consume the same derived task-work path. Physical
read/write cardinalities come from access relations; scalar launch/storage
resources and dataflow declarations come from backend TaskBody traits. The
command `rg -n 'StageKind|NonGemmStageNs' lib/Solver/{CostModel,ChainDP,CouplingInterfaceDP,TaskModel}.cpp`
prints no matches. Operator recognition and ownership projection elsewhere in
Solver are distinct from a per-operator nanosecond formula. The complete
`rg -n "StageKind|NonGemmStageNs" lib/Solver` output is archived in
`COSTMODEL/stagekind_audit.txt`; its five remaining files implement parsing,
alignment, ownership, retained-prefix accesses and attention validation.
The standalone
`COSTMODEL/check_body_admission.log` checks the shared decomposition to
1.45519e-11 ns. Batch evaluation is compared with independent individual
queries on 152 scalar/collective points; immutable work-class reuse is exact,
not a sampled approximation (`reuse_traffic_check.log`, `check_prepared.log`).

✅ Verified, historical calibration validation with fresh CPU evaluations:
18-point full Spearman is 0.884416924665 versus the R5 0.880288958 threshold;
coarse Spearman is 0.766769865841, below 0.85. Raw evaluator rows are in
`COSTMODEL/calibrated_replay/evaluations.tsv`; the measured reference is
`SIMULATOR/raw/time/l2.tsv`. These are not fresh GPU speedup measurements.
The 68-dump replay relative errors, recomputed as
`abs(predicted_ns / measured_ns - 1)`, are p50 4.539964%, p90 10.847552%
(using the same percentile function as R5), max 13.827710%. R5's corresponding reported values were
4.54%/10.85%/13.83%. The coarse ranking gate remains failed.

✅ Verified: full evaluation, with immutable graph preparation separately
reported, peaks at 2480.201 us for a reference point (mha4 seq512 legacy),
and 2873.300 us for real-width. Thus the reference 1 ms budget still fails;
the real-width 10 ms budget passes. Preparation peaks at 145561.384 and
242049.983 us respectively. Omitting seq512 would hide the failed budget.
Shared dependency sets, worker histograms, and the exact flat-hop recurrence
remove the previous dense event propagation work, but do not make preparation
or every reference evaluation meet the target.

⚠️ Inferred next work: `PlacementSolvePass.h` still converts the calibrated
whole-stage `CombineStageNs` estimate to a task price by dividing by waves.
This combine conversion does not derive the actual partial/residual traffic
of each tile. `PriceTaskInstances(..., active_ctas=1.0)` and observed-duration
simulation also do not model the occupancy-dependent change in service time
of a new geometry. These are concrete remaining pricing problems, not a claim
that unifying the retired non-GEMM branches has completed every cost model.

## F-192 — R6 closes the concrete placement producer and preserves legacy tables

✅ Verified: the MLIR placement pass writes `mode`, `params`, `window`,
`policy`, `resident_only`, `grid_map` and `resident_limit_map` directly on
`PlacementOp`. Materialized EFT tables are carried in the CG module attribute,
so the serialized graph is self-contained. Legacy `map=[0]` remains valid.
The one-point solve records theta, kappa, geometry and compiled resident-grid
constraints; it does not pretend a point-specialized EFT table is an interval
solution. Higher residency is admitted from a whole-kernel CUDA occupancy
query, not a maximum over individual TaskBody estimates.

✅ Verified: `WRITEBACK/roundtrip_gqa2_s4/{cg_plan,host_plan}.tsv` are byte equal
for 2696 nodes. Baseline and current legacy schedule/waits/events tables are
byte equal in all 12 comparisons under `WRITEBACK/legacy_matched/identity`.
Each baseline executable links its own baseline host archive: R6 enlarged
`TargetSpec`, so combining old headers with the current host library would be
an ABI error. Full default SEQSCAN is 600/600 (two reference models, seq
4/128/2048, past 0/512, fifty fresh processes each). All 49 CTest cases pass
after rebuilding the grid-aware driver (`WRITEBACK/grid_ctest.log`).

⚠️ Scope: production solves are available through `--solve TARGET.json`;
the legacy import/codegen route remains for compatibility. This closes the
previously missing concrete producer/consumer channel, not an assertion that
all parameter intervals or all graph families have an automatic solution.

## F-193 — R6 bounds the currently legal fusion family before expanding search

✅ Verified (model evaluation, not an executed fused kernel): the selected
six CGs contain respectively 2/2/4/4/4/4 supported adjacent RoPE→KVAppend
pairs, ordered gqa2 s4/s128, mha4 s4/s128, real-width s4/s128. Every adjacent
logical pair is attempted using `DeriveLogicalFusionCandidate`; unsupported
pairs and exact rejection reasons remain in `JOINT2/fuse_upper/*_selected.tsv`.
This is the existing FusionPass family; the report does not infer eligibility
for a broader multi-producer or split-partial fusion.

The optimistic model bound uses the prescribed historical fixed share 0.232
of the pair's separate task envelope, plus the resource-path reduction when
only internal intermediate traffic is free. The latter invokes the same
`TaskInstanceNs` path with access-derived `TaskMemoryTraffic`, retains external
traffic and arithmetic, and charges no extra shared copy, fused barrier,
recomputation or occupancy loss. It is capped at the separate envelope.
The old net prediction (`separate_task_ns - fused_task_ns`) remains alongside
this upper-bound calculation; it is not substituted for an upper bound.
Summing pair envelopes is optimistic about their critical-path exposure.
This is a calibrated model bound with the current wave-price assumptions,
not a proof about unmeasured physical execution.
The common 0.232 fixed fraction is the prompt's extrapolation assumption,
not a measured fixed fraction for every RoPE/append pair. Even removing the
entire modeled pair envelope (including all its arithmetic) caps the largest
share at 0.034358, still below 0.10. Thus the current-family decision does not
hinge on interpreting the common fixed fraction as a per-pair measurement.

✅ Verified: the maximum bound divided by the selected candidate's measured
`max(cp_corrected, queue_lb)` is 0.012657, below the frozen 0.10 decision line.
Raw price rows, source CG hashes, removed-node and intermediate-byte counts,
and arithmetic are in `JOINT2/fuse_upper/`; `verify.py` recomputes the bound
and its denominator rather than reading `bounds.tsv` or `decision.txt`.

```text
FUSE6 enter_r7=0 maximum_bound_share=0.012657 cells=6
```

⚠️ Inferred next step: this keeps the current narrow Fuse family out of the
R7 outer search. To reach a larger bound, extend logical adjacency/ownership
composition in `TaskModel.cpp::ComposeModelCandidate` to the rejected
multi-producer and split-combine cases, then reprice their actual eliminated
nodes and traffic. The existing Fuse direction gate remains unchanged;
no new fusion decision or fused TaskBody was implemented in R6.

## F-194 — R6 task-zero audits localize the remaining price mismatch

✅ Verified: `COSTMODEL/audit_selected.cpp` reproduces the production task-zero
query for every runtime stage in the six frozen selected CGs. Its inputs,
source hashes and raw prices are in `COSTMODEL/selected_prices/`. The Python
companion joins physical `(stage, logical_task=0)` to fresh selected trace
slots; it also reports all-task stage medians separately. This is not an
assumption that boundary tasks share task zero's work.

For non-combine stages, medians across stages of measured/task-zero-price are
1.449 / 1.902 / 1.449 / 1.313 / 2.213 / 2.007 (gqa2 s4/s128, mha4 s4/s128,
real-width s4/s128). For the four split configurations, the corresponding
combine ratios are **23.282 / 25.320 / 31.399 / 12.885** (gqa2 s4, mha4 s4,
real-width s4/s128). The global-timer tick is 1024 ns; zero-length scalar
observations are retained, not clamped into positive samples.

One concrete gqa2 s4 example is runtime stage 2: the production conversion
predicts 74.858 ns, while task zero measures 2048 ns and the stage median is
1024 ns. In real-width s128 stage 2 the prediction is 3417.347 ns versus
26624 ns for task zero and the stage median. At gqa2 s128 runtime stage 8,
the non-combine prediction is 2118.708 ns versus 36864 ns for task zero and
31744 ns for the median. The price error is therefore not only publication
or a small tie-breaking constant.

⚠️ Inferred repair order: replace the whole-stage-to-task conversion at
`PlacementSolvePass.h`'s `projected.combine` branch with access-derived
FP32 partial/residual reads, output stores and backend-declared per-thread
iteration work; calibrate the task's fixed term at the compiled geometry.
Then calibrate occupancy-sensitive service in `PriceTaskInstances`, which
currently receives `active_ctas_per_sm=1.0`, rather than treating observed
new-geometry task duration as occupancy-independent. Re-run the frozen
ranking and end-to-end gates after these changes. This audit does not change
the R6 selected candidates or relabel a failed gate as passed.
The attention example also points to `AttentionChunkTaskBody.h`'s thread-zero
softmax loops: `ScalarDataflow.h` currently describes their phase order but
not their one-lane serial iteration count. Carry this implementation work
into backend traits and calibrate the sequential instruction term; a peak
SFU/CUDA throughput price cannot by itself represent that loop latency.

## F-195 — R6 proves a finite symbolic domain without replacing unfit winners

✅ Verified: four template families (grid-stride, rotate, band, wavefront)
cover every integer seq in [1,128] at grids 256 and 340 for the gqa2 CG
recorded in `SYMBOLIC/complete/provenance.json` (M32 N16 K16, split4, kappa1,
compiled residency witness 512). The 166 certificate pieces establish total
ownership, bijection/dense slots, acyclicity of semantic/queue/grouped-event
edges, resident-grid legality and a dependency-span bound. The non-power-of-two
wavefront case exhausts all 128 integer points as ISL proof pieces; it does
not infer an interval proof from five samples. These pieces are not binary
variants. All generated benchmark binaries retain one geometry variant.

✅ Verified: forty endpoint/interior full Plan comparisons (seq 1/32/64/96/128,
two grids, four families) are byte-identical to native host materialization.
The four combined-grid maps are carried in **one serialized CG** and forty
read-back evaluations also agree (`SYMBOLIC/cg_roundtrip.log`). Reusing parsed
ISL maps in this evidence driver removes repeated parsing of the same large
map; the domain and every comparison point are unchanged. Source maps,
certificates, complete Plan tables and the serialized CG are retained.

✅ Verified: `SYMBOLIC/fit_native/` independently compares each of the six
actual selected Plans against four native-checked templates at its own grid.
Only mha4 s128 matches rotate. The other five EFT selections fall outside
these four families; they remain materialized EFT Plans. This does not prove
that no other quasi-affine family can express them.

✅ Verified, CPU only: the same carried CG evaluated at grids 256 and 340 is
compared with fresh six-catalog solves (`SYMBOLIC/cross_grid/`, sixteen full
Plan comparisons). The fresh winner is wavefront at 256 and EFT at 340.
Thus a symbolic template is reusable over the proven grid alternatives,
while a winner and its materialized table are not presumed invariant. The
original compiled resident limit is retained; the experiment does not change
the physical SM count or move a 4090 table onto another architecture.

⚠️ Scope: divisors are constant within the two grid branches. An unbounded
symbolic grid divisor is not affine/Presburger. This evidence is specific to
the recorded graph and finite interval; it is not a proof for all six winner
geometries, arbitrary sequence lengths, or an sm_120 performance result.

## F-196 — R6 anchors public architectures and quantifies the remaining import gaps

✅ Verified, source inspection: the archived public configurations give
Llama-3.2-1B 16 layers, hidden 2048, intermediate 8192, 32 query/8 KV heads,
head_dim 64, RoPE theta 500000, RMS epsilon 1e-5; Qwen3-1.7B has 28 layers,
hidden 2048, intermediate 6144, 16 query/8 KV heads, head_dim 128, RoPE theta
1000000, epsilon 1e-6. RoPE scaling, vocabulary/tied-head settings and context
limits are in `MODELS/dimensions.tsv`. Qwen uses its official public config.
Meta's config download returned HTTP 401; the archived Unsloth public copy
is explicitly identified as a redistributor source and dimensions were
cross-checked against Meta's public SKU/implementation. URLs, bytes, hashes
and failed requests are preserved in `MODELS/sources/manifest.json`.

✅ Verified, repository inspection: there are 11 current TaskKind values,
not the prompt's assumed 16. The coverage table distinguishes shape-capable
backends from verified architecture import. Gaps include token embedding,
final-head import, Llama's normalization epsilon, the backend's BF16 RoPE
angle rounding versus the public FP32 computation, and Qwen's per-head Q/K
normalization. No missing operator was implemented in this audit.
`MODELS/extension_sites.tsv` counts 15 conditional integration sites for a
new distinct task/ownership shape, including three harness dispatch sites.
The unified cost path needs access semantics and backend traits/dataflow;
it does not require adding another operator-specific nanosecond formula.

✅ Verified, explicitly degraded execution scope: one `tilemega-compile`
command consumes the exported `.pt2`, performs the internal import bridge,
queries compiled occupancy, solves geometry/kappa/placement/residency,
writes CG and emits CUDA. Its output is byte-identical to the source with
50/50 fresh-process numerical PASS results and sixteen output comparisons
per process (`MODELS/llama_mlp/direct_pt2.command.json`, `correctness/`).
The graph comprises sixteen independent, already-normalized Llama-width MLP
regions with distinct random weights and explicit boundary inputs. It is
neither a sequential decoder nor the maximal whole-graph covered subset;
covered attention regions have not been assembled into that subset.

The fifty single-launch observations (warmup=0, repeat=1) have L2 median
2.016880 ms, range 1.973248–2.034688 ms; L1 median is 3.050624 ms.
These are diagnostic execution times, not a steady-state full-model speedup.
No pretrained weights or Hugging Face runtime dependency were needed.

⚠️ Inferred next work: first expose exact epsilon and FP32 RoPE semantics,
then lift embedding/final-head and per-head normalization ownership; extend
the existing structural region import to the covered attention cuts and
assemble the largest connected/exportable covered graph before promoting
this anchor to the EX-V1 main benchmark. `MODELS/subset.md` retains the exact
cut boundaries and regeneration command.

## F-197 — R6 improves both real-width controls while reference gates fail

✅ Verified: all six frozen selected configurations pass 50/50 fresh
processes. Selected reference SEQSCAN adds 400/400 at seq 4/128 and past
0/512. Each cell has 25 same-session rotated rounds of control, R5 champion
and the three predicted candidates. Separate five-round pilots determine the
frozen choices; the confirmatory samples do not choose a new winner.
Raw inputs are indexed by `JOINT2/cells.json`; process metadata and complete
trace dumps remain next to each cell's `choice.json`.

The unique research gate **J-a passes 2/2**. Real-width s4 selected/control
is 0.556566970 [0.556037286, 0.577495619]; s128 is 0.927801799
[0.913513489, 0.948331047]. Against the fresh R5 champions the ratios are
0.637556076 [0.636955526, 0.638260870] and 0.934261122
[0.905858521, 0.972870662]. These are paired bootstrap 95% intervals,
not ratios to archived R5 timings. Full selected medians are 2.626560 and
6.368480 ms; L2/L1 is 0.926678/0.954487 and measured/floor is
1.113281/1.141560.

✅ Verified: **J-b fails both real-width cells**. The selected semantic
CP/queue bounds are 1275904/2359296 ns and 3277824/5578752 ns;
queue/semantic-CP is 1.849117 and 1.701968. The fresh R5 champion ratios
are 1.751788/1.428188 (historical R5: 1.7589/1.4174). Queue service decreases
in absolute terms while the semantic path decreases more. The search's
binding objective is not mathematically a constraint on their ratio.
Replacing the denominator with a queue-containing union path would make
this gate tautological; that substitution was not made.

✅ Verified: **J-e fails 4/4**. Selected/R5-champion ratios, ordered gqa2
s4/s128 and mha4 s4/s128, are 1.291205752 [1.285477178, 1.292035398],
1.253164557 [1.252869015, 1.254237288], 1.345579955
[1.342342342, 1.352684233], and 1.087647778 [1.085632000, 1.089210291].
Their upper bounds exceed the unchanged 1.02 line. **J-d fails** as well:
predicted top1 ranks 1/3, 3/3, 1/3, 1/3, 3/3, 3/3 across the six cells,
so only 3/6 land in the measured top two (required at least four).
Five selected placements are EFT; mha4 s128 selects rotate. Every evaluated
geometry includes all six placement families, but the finite outer capacity
is still an explicitly degraded search, not a global optimum certificate.

⚠️ Inferred next work: the F-194 combine and serial-attention price errors
remain large enough to change split, geometry and placement ranking. Repair
those service prices and the active-residency term before enlarging the
candidate domain. If queue/semantic-CP <=1 remains a required admissibility
condition, search that feasible subset explicitly and measure its latency
tradeoff; minimizing the maximum alone does not enforce the condition.
The real-width gains do not erase these failed reference/ranking gates.

## F-198 — R6 remeasures placement and protocol on the selected geometries

✅ Verified: B1 completes six cells, ten configurations per cell, five probe
arms per configuration and 25 rotated fresh-process rounds: 7500 processes.
The complete-execution arms pass 1500/1500. Each cell contains the six
placement families, R3 B disabled, and cumulative C1/C2/C3(a) additions.
W remains 1, so the window-only C3(a) local-dependency mechanism is inert.
Raw logs, process metadata, generated sources and full trace dumps are in
`REBASE/raw/`; choices were committed before these measurements began.
These are solver-selected diagnostic geometries, not qualified optima:
the failed J-b/J-d/J-e gates in F-197 remain failed.

| Cell | Rotate / legacy, paired median [95% CI] | R3 B on / off, paired median [95% CI] | +C1 / B | +C1+C2 / B | +C1+C2+C3(a) / B |
| --- | --- | --- | --- | --- | --- |
| gqa2 s4 | 0.763348 [0.759781, 0.764688] | 0.899306 [0.896412, 0.901265] | 0.997849 | 0.993831 | 0.998494 |
| gqa2 s128 | 0.739130 [0.734291, 0.743233] | 0.955636 [0.954281, 0.958065] | 0.999790 | 0.997054 | 1.000000 |
| mha4 s4 | 0.711779 [0.711055, 0.713568] | 0.925926 [0.910370, 0.967742] | 1.000000 | 1.003344 | 1.000521 |
| mha4 s128 | 0.734715 [0.733359, 0.736485] | 0.992579 [0.990724, 0.994424] | 0.987394 | 0.989136 | 0.990637 |
| real s4 | 0.943148 [0.933309, 0.948023] | 0.978700 [0.978103, 0.983907] | 1.001633 | 1.002379 | 1.001890 |
| real s128 | 0.954921 [0.952561, 0.981870] | 0.989346 [0.940345, 1.013014] | 0.999155 | 1.000825 | 1.000670 |

Placement comparisons fix R3 B; protocol comparisons fix the selected
placement. Thus these are conditional effects, not independent factors to
multiply. R1's historical pooled rotate/legacy ratio was 0.6705. R3's default
placement cumulative B reductions were 5.16/3.48/4.79/3.74%; R4's 36%
protocol/barrier reduction coexisted with slower full execution. Those
historical percentages do not transfer to the current geometries. All five
intervention differences, floor and L1 ratios, and their paired intervals
are retained in `REBASE/attribution_*.tsv`.

✅ Verified limitation: real-s128 has **44/250 nonpositive barrier pairs**
(all strictly negative; none zero). Pair start-time separation has median
59.017 s and maximum 1213.832 s. The unchanged EFT full L1 control ranges
from 6.316000 to 13.919040 ms; nofence L1 ranges from 6.320128 to 13.634560
ms. Exact per-function SASS comparison establishes that nofence changes
only L2, while l1nosync changes only L1 (`REBASE/probe_audit/`). The large
control variation cannot be identified as the direct instruction cost of
the removed L2 fence. Its underlying physical cause is not established.

The analyzer retains every signed difference and every paired ratio. It
does not clamp negatives, discard slow processes or replace them with
additional samples. A zero denominator would make the complete ratio
statistic undefined, rather than silently removing that pair. Removing the
analyzer's extra positivity assertion changes neither the prompt's gates
nor the prescribed subtraction; it corrects an unsupported implementation
assumption. Causal isolation of the real-s128 protocol components is an
explicitly degraded result, even though sample coverage is complete.

⚠️ Inferred next work: record clocks, power and allocation information per
process; rotate the five arms within each configuration and independently
rotate configuration order. `REBASE/run.py` currently rotates all fifty
arms as one list, allowing a pair to straddle almost a full round. Diagnose
the common timing bands before interpreting these differences as physical
service costs. The present samples and their signs remain unchanged.

## F-199 — real:s128's mode-5 control probe reports MISMATCH on sm_120, one element over the BF16 bound

✅ Verified on an RTX 5090 (sm_120, CUDA 12.8), reproduced identically from
two independent invocations (`CHAIN2/prepare_sm120.py` and
`JOINT/target_pipeline.py`, both compiling and running the same
`TILEMEGA_PLACEMENT=5` control probe against `REALMODEL/raw/work/r2sim_s128`):
both report `RESULT status=MISMATCH` with byte-identical L1/L2 output hashes
(`4f1e075b02046216`) and byte-identical diagnostics. This is not a flaky or
environment-dependent result — the same element fails the same way every
time — and it is not a stale-fixture artifact: `export_real.py:124` fixes
`torch.manual_seed(20260906)`, so the regenerated `r2sim_s128` here is the
same model instance F-189 measured on sm_89.

`ModelHarness.cuh:2646` (`Compare`) applies a per-element bound of
`1.6e-2 + 1.6e-2*|expected|` for BF16. With `TILEMEGA_DIFF_DUMP=20` set, the
probe names the single offending element:

```
E2E_DIFF_ELEM pair=l05_vs_l0 tensor=0 index=6597 actual=-0.137695312
  expected=-0.119628906 delta=0.0180664062 tolerance=0.0179140642
E2E_DIFF l05_vs_l0_mismatch=1 max_abs=0.046875 max_rel=5859.375
  l1_vs_l05_mismatch=0 max_abs=0 max_rel=0 l2_vs_l1_mismatch=0 max_abs=0 max_rel=0
```

Exactly one element, out of the whole L0-vs-L0.5 comparison, exceeds its
tolerance — by 0.00015, about 0.85% over the bound. Every other reported
`max_abs`/`max_rel` pair in the same log (up to `max_rel=5859.375`) belongs to
elements whose `expected` is near zero, which is why a large `max_rel` alone
does not imply a second mismatch; the harness's own mismatch counter, not
`max_rel`, is authoritative. `l1_vs_l05` and `l2_vs_l1` both show zero
mismatches — the generated megakernel (L1/L2) agrees exactly with the
standalone L0.5 reference; only L0.5-vs-PyTorch (L0) disagrees, and only at
this one element.

⚠️ Inferred, not yet isolated further: F-189 records this exact `real:s128`
configuration passing 50/50 on sm_89 with no tolerance change, so the
divergence is plausibly a BF16 GEMM rounding-order difference between the two
architectures' tensor core paths reaching a quantization boundary on sm_120
that sm_89 does not reach — but no instruction-level comparison has been done
here to confirm that mechanism over an alternative (e.g. a different reduction
order inside this one GEMM tile). Reproduction: `TILEMEGA_PLACEMENT_BASE_DUMP=1
TILEMEGA_DIFF_DUMP=20 <binary> <r2sim_s128 fixture>`.

This is a hard gate, not a script bug: `CHAIN2/prepare_sm120.py` and
`JOINT/target_pipeline.py` both require every mode-5 control probe to report
`RESULT status=PASS` before deriving anything from it, and neither loosens
that check for `real`. Per CLAUDE.md, the tolerance is not moved to match this
result. Until this is resolved, both scripts' `real` arm cannot run on sm_120;
`gqa2`/`mha4` are unaffected (neither uses `r2sim_s128`).

## F-200 — Cluster/DSMEM validation on the RTX 5090: primitives and megakernel both clean; the residency hazard from 2026-09-07 is unfixed in source but not retriggered

✅ Verified on the same RTX 5090 (sm_120) as F-199, `HEAD=e741e5c38`. Three
scripts, all sm_90+-only (the one mechanism this hardware generation adds that
sm_89 cannot exercise at all):

**`CLUSTER/run_notify_cluster_sm120.sh`** (T1.3-B, L2 traffic attribution for
the cluster/DSMEM path) — ✅ already complete, committed 2026-09-08
(`e305a9f4c`), not rerun here: `cluster_dim` ∈ {1,2,4,8} × {gqa2,mha4} ×
{load_lines,cluster} × {seq 4,128}, 32/32 cells 50/50. SASS: `load_lines`
carries 0 `UCGABAR` instructions, `cluster` carries 4, at every dim.

**`CLUSTER/run_on_cluster_gpu.sh`** (Part 7, DSMEM primitives + real BF16
megakernel) — this script was last touched 2026-09-07 (`raw_5090/`,
`c4f412353`) and never had its findings committed back into the script or into
this file. Both are done now.

*What 2026-09-07 found, from `raw_5090/debug.txt`* (⚠️ stated, that session's
own account, not independently re-derived here): the `two_level_barrier`
primitive passed `stage` — not the per-stage-independent `arrivals[stage]`
counter's own iteration, which starts at 0 — as `StageBarrier`'s monotonic
iteration argument, so stage ≥ 1 needed `clusters*(stage+1)` arrivals against a
counter that only ever reaches `clusters*1`: permanent deadlock. Fixed there
by passing `0u`, after which primitives passed 200/200 at cluster sizes 2/4/8.
The real megakernel arm then reproduced a second, substantive problem: gqa2
passed 50/50 at cluster dims 1/2/4 and **hung** at dim 8; mha4 was never
reached. Diagnosis: `TILEMEGA_GENERATED_RESIDENT_GRID` (`ModelHarness.cuh`)
sizes the resident grid from flat CTA occupancy (`num_sms *
ActiveBlocksPerSM(...)`) and the cluster path only rounds that number down to
a multiple of `TILEMEGA_GENERATED_CLUSTER_DIM` — it never asks how many
*clusters* the device can actually keep co-resident. At dim 8 the flat-CTA
math authorized more simultaneous clusters than the hardware could schedule
together; the first batch entered `StageBarrier` and the rest could never
become resident to join them.

*What was fixed here, before rerunning* — the 2026-09-07 primitive fix had
never been committed; the script itself still carried the original `stage`
argument. Fixed the same way (`0u`), plus three unrelated bugs found while
preparing the rerun: `find_mlir_dir()` referenced an undefined
`TILEMEGA_ROOT` instead of the script's own `root` (`set -u` would abort it);
the script's internal `cmake` call had no way to pass `TILEMEGA_LIT_DRIVER`,
which this MLIR install requires (same requirement as the top-level build);
and the megakernel hash extraction, `grep -o 'l1=[0-9a-f]*'`, matched inside
`E2E_TIME`'s unrelated `l2_over_l1=1.05...` field before ever reaching the
real `E2E_HASH` line, recording every hash as the literal string `1` — fixed
by anchoring the grep to the `^E2E_HASH ` line first. None of these four
touch what the script measures; all are namefixes/plumbing. Also added a
`timeout` (default 60s, `MEGA_HANG_TIMEOUT`) around both the primitive and
megakernel binary invocations, so a real hang is now recorded as a counted,
reported non-pass (a new `hangs` column in `megakernel.tsv`) instead of
blocking the script — and the host — indefinitely.

✅ Rerun in full, `MEGA_RUNS=50` (default), `RUNS=200` (default):

| primitive | cluster_size 2 | cluster_size 4 | cluster_size 8 |
|---|---|---|---|
| dsmem/pairwise/two-level (combined) | 200/200 | 200/200 | 200/200 |

| model | dim | barrier_cluster_asm | pass | hangs | hash |
|---|---|---|---|---|---|
| gqa2 | 1 | 0 | 50/50 | 0 | `4c544373c0add101` |
| gqa2 | 2 | 8 | 50/50 | 0 | `4c544373c0add101` |
| gqa2 | 4 | 8 | 50/50 | 0 | `4c544373c0add101` |
| gqa2 | 8 | 8 | 50/50 | 0 | `4c544373c0add101` |
| mha4 | 1 | 0 | 50/50 | 0 | `853cf67220dbad5f` |
| mha4 | 2 | 8 | 50/50 | 0 | `853cf67220dbad5f` |
| mha4 | 4 | 8 | 50/50 | 0 | `853cf67220dbad5f` |
| mha4 | 8 | 8 | 50/50 | 0 | `853cf67220dbad5f` |

Every dim, both models: 50/50, zero hangs, and — with the hash bug fixed —
bitwise-identical output within each model across every cluster width,
including dim 8. `barrier_cluster_asm` (`UCGABAR` count) is 0 at dim 1 and 8
at dim ≥ 2, the same signature §7.3 verified by cross-compilation. mha4, never
reached in 2026-09-07's run, is now measured at every dim and is clean.

⚠️ **The dim-8 hang did not reproduce, but its cause is still in the source,
unaddressed.** Read directly, not inferred: `ModelHarness.cuh`'s resident-grid
computation is unchanged since 2026-09-07 —
`grid -= grid % TILEMEGA_GENERATED_CLUSTER_DIM` is still the entire cluster
adjustment, with no query of how many clusters the device can actually hold
resident together. (`TILEMEGA_RESIDENCY_CAP`, a few lines above it, is an
unrelated caller-supplied override for the joint-search experiment, not an
automatic cluster-aware capacity check.) The most likely explanation is that
today's exported models produced different task/occupancy numbers than
2026-09-07's — `CODEGEN_SUMMARY` here reports `gqa2: tasks=34 couplings=42`
and `mha4: tasks=68 couplings=86`, uncompared against that session's counts,
which were not recorded — landing the flat-CTA grid within whatever the
RTX 5090 can genuinely keep co-resident this time, rather than the mechanism
having been repaired. ⚠️ Stated, not verified: no instrumentation here queries
actual hardware cluster-residency capacity to confirm this explanation over
alternatives. **The hazard is a live one**: any future model/kernel shape
whose flat-CTA occupancy authorizes more simultaneous clusters than the
device can schedule together will deadlock the same way, silently, with the
harness's own hang now at least caught by the `timeout` added here rather
than blocking indefinitely — but not prevented.

## F-201 — R6 continuation derives combine work and corrects replay ownership

**✅ verified.** `DeriveCombineTaskInput` now derives each instantiated reduction
from its semantic node and access relations, including the tail task and the
physical tile/element ownership map. FP32 partial reads and BF16 residual reads
have distinct byte cardinalities. The backend declares the zero-seeded extra
addition; the solver does not introduce a per-operator latency formula.
`Evaluate`, `ChainDP` and `CouplingInterfaceDP` use `CombineTaskStageNs`, and the
placement producer prices actual combine instances rather than dividing a
whole-stage estimate by waves. Compiled residency is part of each price and
its cache key. `task_element_work_test` independently enumerates both ownership
forms and typed tail traffic; the full 49-test CTest suite passes.

**✅ verified.** The calibration replay previously imported element ownership
although the measured sources declared tile ownership, and copied task-zero
prices across a stage. For example, the historical RoPE stage had 16 tasks
while the imported task domain had 8. `COSTMODEL/evaluate.cpp` now reads the
original source's ownership flags, checks cardinality and prices every task
coordinate. On the unchanged historical 18-point calibration set, fresh CPU
re-evaluation gives coarse Spearman **0.876160990712** and full Spearman
**0.896800825593**. C-c's 0.85 and C-b's 0.880288958 gates therefore pass.
This is a correction of calibration replay inputs, not a fresh GPU speedup.

Evidence: `docs/experiments/COSTMODEL/closure_replay/{evaluations.tsv,ranks.tsv}`,
`docs/experiments/JOINT2/closure/ctest_current.log`, and the retained historical
`docs/experiments/SIMULATOR/raw/time/l2.tsv`. The new 68-dump replay keeps the
publication, consumer wait and visibility constants unchanged; p50/max absolute
relative errors remain 4.53996%/13.82771%. CPU timing and new-search performance
are separate gates and are not inferred from the improved calibration rank.

## F-202 — R6 writes every placement in a finite theta interval through CG

**✅ verified.** `SolveAndWritePlacementInterval` invokes the six-family solver
at every integer seq in a requested interval, retaining each point's winner.
The CG carries the materialized interval as an array of tables, and Codegen
emits one runtime variant whose host selects the table by seq. The compiler's
`--seq-begin` option chooses geometry, kappa and residency at the upper endpoint,
then holds them and past/grid fixed while solving placements over the interval.
This is exact bounded materialization, not an unbounded affine fit or a proof
of joint geometry optimality over the interval. Out-of-range workloads and
inconsistent seq/past/grid metadata are rejected.

**✅ verified.** A single generated binary passes **50/50 fresh processes at
seq 1, 2, 3, 4 and 5**, with past=3. All six placements are evaluated at each
point. Independently solved arrays, written CG, generated table arrays and
actual host queue dumps agree byte-for-byte at 546/580/614/648/682 nodes. CG
serialization leaves generated CUDA identical; three malformed interval cases
(seq gap, different past, different grid) are rejected. This focused campaign
uses residency=1 and capacity=1 without compiled resource probes; it is not the
performance search. Materialized EFT winners are not replaced by a weaker
symbolic family to make a certificate pass.

Evidence: `docs/experiments/WRITEBACK/interval_closure/README.md`,
`gqa2.cu.interval.tsv`, `correctness/s*/r*.{log,json}`, `direct_s*.tsv`,
`host/s*/schedule.tsv`, and `roundtrip_v3.log` in the same directory. The raw
verifier adds an explicit interval gate and checks the generated/host arrays;
the previously checked single-point round trip remains covered separately.

## F-203 — R6 maximal covered Llama graph fails numerical admission

✅ **Verified.** The continuation imports all supported regions of the public
Llama-3.2-1B architecture: 16 connected residual layers, Q/K/V/O projections,
KV append, attention, SwiGLU and the vocabulary head. Only the previously
audited embedding, RMSNorm parameter and FP32 RoPE gaps are external inputs.
There are 98 inputs and 66 checked outputs; the solver writes and generates
209 runtime stages. This supersedes the earlier independent-MLP scope as an
implementation, but does **not** supersede its correctness result with a pass.

✅ **Verified.** Two solver-selected geometries (64x128x16s2 split8 and split4)
and three explicitly manual split1 diagnostic geometries fail the same final
residual element: actual -0.41796875, expected -0.44921875, absolute difference
0.03125, fixed tolerance 0.0231875014. All other 65 outputs pass; L0.5, L1 and
L2 agree bit for bit. This new graph was never an admitted baseline. Its
maximal-graph 50-process gate is FAIL; the first failure stops that admission
under §9.2, without changing tolerance, seed, output set or residual edges.

✅ **Verified.** An independent FP64 recomputation localizes the initial error
to V[0,463] of the first layer. The FP64 accumulator -0.450195362966042 is
just below the BF16 midpoint -0.4501953125; CPU and FP64-rounded values are
-0.451171875, whereas GPU gives -0.44921875. Substituting only the captured GPU
V into the diagnostic CPU attention removes all three context discrepancies.
Later residual additions amplify the first rounding difference. Diagnostic
interventions never replace the frozen golden.

⚠️ **Inferred next implementation.** Accurate/compensated accumulation or
selective recomputation near BF16 midpoints in `GemmStageTaskBody.h`, with
backend cost traits, should address the first cause. The full downstream
attention/residual chain must then be revalidated; this is not yet a proven
fix. The controlling rule for this numerical stop is §9.2 after failed
admission; the missing-operator exclusion is not a blanket prohibition on
backend bug fixes. Unlock requires a demonstrated accuracy fix and unchanged-
golden revalidation. Record the downstream A1 timing exclusion, rather
than claiming the older independent-MLP 50/50 closes the maximal graph.

Evidence: `MODELS/subset.md`, `covered_llama*/correctness/r0.log`,
`covered_geometry_probes/*/run.log`, and
`covered_llama_admitted2/diagnostic/{cpu_buffers_aligned.tsv,first_v_rounding.json,first_context_isolation.json}`.

## F-204 — R6 fills the outer bounds and preserves simulator predictions

✅ **Verified implementation.** The continuation found that outer candidate
work/CP/queue bounds were zero and the finite budget was ordered by an L1
heuristic. `PreparePlacementProblem` now supplies the same derived-cost task
graph to both outer bounds and the six-placement inner catalog. The outer
priority is max(work, semantic CP, queue pigeonhole bound); each candidate's
raw bounds and evaluated/deferred/pruned status are emitted. The grid upper
bound uses SM count and hardware thread capacity, not guessed compiled
occupancy. Exact compiled occupancy still controls inner legal residencies.
Final campaigns use `JOINT2/bounded_search`; earlier `closure_search` results
are superseded diagnostics, not final acceptance evidence.

✅ **Verified.** Incremental event-row reuse equals an independent full graph
rebuild in 36 model/placement comparisons (`JOINT2/closure/event_reuse.log`).
The compact fixed-duration recurrence keeps queue end times per worker and
avoids unused per-node publication/wait/readiness arrays. Random DAG/FIFO
cases and non-topological node ids agree with the independent event heap.
All 40 replay predictions, including both CP definitions and queue bound,
remain byte-identical across five numeric columns.

✅ **Verified budget result.** Cold full evaluation, including plan readiness,
peaks at 2633.135 us on references and 2548.311 us on real-width in
`COSTMODEL/closure_compact/evaluations.tsv`. J-c still FAILs its reference
1000 us gate; cached-only timing does not replace it. Shared graph preparation
is separately reported. Finite candidate capacity remains an explicit
degradation, with no optimality claim. The next budget work is to reduce
remaining cold readiness/recurrence memory traffic and allocation while
retaining independent timestamp equivalence and the full timing boundary.

## F-205 — R6 proves current wavefront winners at their actual grid

✅ **Verified.** The refreshed gqa2 s4 and mha4 s4 winners use wavefront at
grid 512. Each unchanged template is now proved for every integer seq in
[1,128], with all legality/residency/level predicates true in 128 singleton
ISL certificates per model. Complete native versus template tables agree at
seq 1,32,64,96,128. These are proof pieces, not additional GPU variants. The
earlier four-family grid-256/340 proof is retained separately and is not
presented as proof of a grid-512 winner.

✅ **Verified scope.** The current seq128 EFT winners differ from all four
tested template families; their exact materialized plans remain selected.
This is a counterexample to those specific fits, not a proof that no other
quasi-affine expression can describe the finite placement. The finite W2
interval carrier retains exact point solutions without substituting a worse
template.

Evidence: `SYMBOLIC/bounded_certificates/{gqa2_s4,mha4_s4}/`, including the
solved-CG hash, proof-process commands, every certificate and all native tables;
`SYMBOLIC/bounded_fit/` for all four reference winners.

## F-206 — R6 stores shared successor regions as exact intervals

✅ **Verified implementation.** Shared dependency groups now store consecutive
node runs as half-open intervals. Sparse rows retain the original vector when
interval pairs would require more storage. Original visitation order, holes,
repeated entries and non-topological node ids are preserved. The exact same
interval-backed rows serve readiness, the event heap, the fixed-duration
recurrence and the binding bound. The immutable source graph still uses its
original representation; this does not claim that initial CG enumeration has
been removed.

✅ **Verified.** All 49 CTest tests pass after this change. Independent
DAG/FIFO/heap checks preserve task timestamps, and 36 full versus incremental
event-graph evaluations agree. All 40 calibration rows remain byte-identical
across five prediction columns. On mha4 s512, shared successor storage holds
17,120 integers instead of 66,224; on real s128 it holds 21,448 instead of
92,192. Frozen CUDA plans and kernels are unchanged by this CPU representation.

✅ **Verified budget result.** A new evaluation process records a maximum
per-plan full time of 1985.803 us for references and 2073.203 us for real-width.
The reference gate remains FAIL, 985.803 us above its 1000 us limit. Shared
graph preparation is separately exposed (reference max 258470.395 us, real
max 125221.387 us), and cached-only recurrence never replaces the full metric.
These single-process budget observations are not a paired speedup claim.
The remaining work is to reduce cold readiness/recurrence traffic and retain
CG relation regions through preparation, instead of enumerating every edge.

Evidence: `COSTMODEL/closure_intervals/{evaluations.tsv,run.log,build.json,command.json,prediction_equivalence.json}`,
`JOINT2/closure/{ctest_intervals.log,event_intervals.log}`, and the read-only
outer-search stack sample in `JOINT2/closure/search_profile/`.


## F-207 — R6 selects retained-prefix work from the CG state effect

✅ **Verified.** The continuation audit found one remaining `StageKind`
branch in `ScalarTaskWork.cpp`: it selected the retained-prefix access region
for element-owned cache updates. The region cardinality was already derived,
but the dispatch still violated the intended separation of cost and operator
kind. It now follows the CG `kv_cache` state effect and the existing ownership
field, requiring a read-write effect. No per-operator byte/time constant was
introduced. `COSTMODEL/stagekind_audit.txt` records the new whole-tree grep;
remaining tags perform parsing, ownership projection, alignment constraints or
attention-contract validation, not pricing dispatch.

✅ **Verified.** The independent element test counts old and newly appended
regions at seq {1,5}, past {0,3,7}, under two scalar stage tags, and rejects an
invalid read-only state effect. All 49 CTest tests pass (132.04 s). A fresh
40-plan CPU evaluation preserves all five prediction columns byte for byte.
The latest cold full-evaluation maxima are 1555.545 µs for references and
2028.897 µs for real-width; the reference 1000 µs gate still fails. This source
change does not optimize the simulator recurrence: differences from the prior
1985.803/2073.203 µs process are repeated-measurement variation, not an attributed
speedup. Evidence: `COSTMODEL/closure_effects/` and
`JOINT2/closure/state_effects/`.

## F-208 — R6 target runners derive their calibration domain locally

✅ **Verified locally; target execution unverified.** The sm_120 orchestration
now generates unmaterialized calibration sources for five geometry probes from
its local exported inputs, fits local phase observations, and derives the
search-domain file and its provenance hash from those observations. No sm_89
worker/slot table or calibration-domain measurement is reused. The explicit
calibration shapes are an experimental design, not a selected production
configuration. The target runner also re-materializes its selected reference
plans at past 0/512, runs the 50-process selected SEQSCAN cells, and writes local
J and B1 analysis tables.

✅ **Verified.** On sm_89, generating the explicit-root SEQSCAN sources at seq 4,
past 0/512 reproduces both existing CUDA and CG artifacts byte for byte. All
five unmaterialized calibration sources generate successfully. Four runner
SELF_CHECK executions pass guard/parser and sm_120 compilation checks; none
launches an sm_120 kernel. Existing-control numerical failures are distinguished
from unsafe probes and new-candidate exclusions and trigger the global-stop
record. Evidence: `WRITEBACK/portable_seqscan/`,
`COSTMODEL/portable_sources/`, `JOINT2/closure/portable_analysis/`, and each
runner's `selfcheck_sm120_portable/` directory.


## F-209 — R6 bypasses point allocation only for proven integer boxes

✅ **Verified.** `VisitFiniteRelation` now first checks a relation piece's box
structure, then proves equality with its integer min/max box before directly
enumerating coordinates. Coupled coordinates, modular holes and other pieces
retain the general ISL enumerator. The carry loop handles `LONG_MAX` without
increment overflow; empty sets, overlapping pieces, callback exceptions and
coupled/modular cases are tested. No dependency, event, ownership or legality
rule changes. All 49 CTest tests pass (120.60 s).

✅ **Verified diagnostic, not a GPU speedup claim.** Complete emitted point
multisets match the previous enumerator for the five already frozen reference
and real-s4 winners, on both dependencies and requested events. This includes
4,574,080 mha4-s128 dependency points. A fresh CPU diagnostic reports that
relation's enumeration at 5,874,810 versus 1,187,660 µs, and real-s4 dependencies
at 519,813 versus 52,783 µs. Event enumeration stays on the general route where
box structure is absent. These single diagnostic pairs are not the simulator
budget measurement or a claim about end-to-end solve time. The change directly
addresses the stack sample in dense ISL point enumeration; dense successor
storage and non-box relations remain concrete follow-up work. Evidence:
`JOINT2/closure/finite_relations/` and `JOINT2/finite_relation_audit.cpp`.


## F-210 — R6 enumerates dense dependency slices without changing points

✅ **Verified.** Four-coordinate relation pieces with two varying coordinates
can be sliced on one task coordinate. Each exact box slice is enumerated
natively; every other slice uses ISL. The dispatch requires a dense first
slice, preserving the general route for sparse/modular relations. Complete
point multisets, including overlapping pieces, match the original enumerator
on all five frozen winners' dependency and event relations. Band, modular-hole,
empty-set, callback-error and integer-endpoint cases remain covered. All 49
CTest tests pass (121.47 s). No search domain, target, cost, dependency, or
legality condition changes.

✅ **Verified diagnostic.** A fresh CPU comparison measures the 4,574,080-point
mha4-s128 dependency relation at 5,855,250 versus 838,770 µs; the 1,529,728-point
gqa2-s128 relation at 2,889,220 versus 472,635 µs. These single-process
enumeration comparisons are not paired GPU results or the cold full-plan
budget gate. Event relations retain essentially the same route. Dense edge
storage still remains after enumeration; retaining shared successor regions
through preparation is the next concrete change if preparation dominates.
Evidence: `JOINT2/closure/finite_slices/` and `finite_relation_audit.cpp`.


## F-211 — R6 validates complete carriers and native queue vectors

✅ **Verified.** The earlier W-d interval checks compared worker/slot arrays,
which alone did not prove equality of every executable table. The supplemental
`WRITEBACK/full_roundtrip/` campaign independently re-solves seq 1..5 and injects
those placements into RuntimePlanDesc, bypassing CG-generated arrays and the
interval carrier. All ten fresh CG/direct executions pass. All fifteen full
`schedule.tsv`, `waits.tsv` and `events.tsv` pairs are byte-identical. The
geometry, past length, grid and finite interval remain those of F-202; this
is stronger carrier validation, not additional interval optimization.

✅ **Verified.** S-b now also compares the actual MaterializedPlan.queue vectors
against evaluation of the already proved symbolic expressions: forty original
family/grid/seq cases and fifteen points for three fitted current winners.
All 55 pairs match byte for byte, including worker count, dense slots, stage
and logical-task identifiers. The existing forty pi/sigma comparisons and
ISL certificates remain in place. Native projection binds theta first, as
the host does; an initial test harness that projected unbound theta spent
time constructing unused symbolic wait counts and was replaced before
collecting the complete accepted campaign. No production legality rule or
winning placement changed. Evidence: `SYMBOLIC/queue_roundtrip/`; both
supplemental obligations are required by `JOINT2/verify.py`.


## F-212 — R6 corrected-search confirmation passes the real-width research gate

✅ **Verified.** After access-derived combine pricing and nonzero outer-bound
repairs, all six selections were independently frozen before confirmation.
Each cell has 25 rotated five-arm fresh-process rounds and 50/50 numerical
checks; the selected reference SEQSCAN subset adds 400/400. Real-width s4 and
s128 selected/control ratios are 0.547671353 [0.539770996, 0.568571846] and
0.944310839 [0.943726539, 0.981891257]. J-a passes both cells. Selected/R5
champion ratios are 0.619535397 and 0.937299504, respectively.

✅ **Verified failures retained.** Real-width queue/semantic-CP is now
2.317659352 and 2.182635901, so J-b fails both cells despite lower latency.
Their selected CP/queue floors are 979968/2271232 ns and 2618368/5714944 ns.
The binding objective does not itself constrain that ratio. Reference
selected/R5-champion ratios are 1.090909091, 1.330033937, 1.175841796 and
1.309298156; every CI upper bound exceeds 1.02, so J-e fails four cells.
Predicted-top1 measured ranks are 3/3, 1/3, 3/3, 2/3, 3/3, 3/3; J-d is
2/6, below four. No post-confirmation reselection or gate change occurred.

✅ **Verified price discrepancy; causal refinement inferred.** Joining every
physical task coordinate with its prepared price gives normal Attention
stage median measured/price ratios 2.5009, 3.8937, 2.5406, 4.2438, 2.9119
and 2.3203 across the six cells. Normal GEMM stage medians are 1.4284,
1.4296, 1.0713, 1.4296, 0.9205 and 0.8240. Real-width RMSNorm medians
are 2.4127 in both cells. Whole-plan replay accuracy does not eliminate
these geometry/resource-specific errors. Next: calibrate serial active-lane
service in backend scalar traits and TaskInstanceNs, including the
thread-zero softmax loop, and compiled-residency dilation; measure the
latency tradeoff of an explicit queue/semantic-CP feasibility restriction.
The current data establish the errors, not a causal percentage attributable
to any one loop. Evidence: `JOINT2/bounded_search/`,
`JOINT2/comparisons_gqa2_mha4_real.tsv`, `COSTMODEL/bounded_prices/`.

## F-213 — R6 recomputes the supported fusion bound on corrected selections

✅ **Verified calculation; model upper bound, not executed fusion.** The
supported adjacent RoPE-to-KVAppend pairs remove 16/512/64/2048/128/4096
physical nodes and 8192/262144/32768/1048576/65536/2097152 global bytes
across the six selected cells. The fixed-plus-traffic bound divided by the
measured binding floor is 0.015050/0.006267/0.015426/0.006421/0.001168/0.001999.
Even deleting each supported pair's entire modeled envelope gives a maximum
share of 0.041874. This is an optimistic bound under the model's wave
assumptions and the prescribed 0.232 fixed-share extrapolation, not a proof
about other fusion families. The existing fusion direction gate is unchanged.

```text
FUSE6 enter_r7=0 maximum_bound_share=0.015426 cells=6
```

The frozen 10% rule therefore does not place the currently supported family
in R7's search outer loop. A broader fusion investment must first extend
the legally supported ownership families and recompute their traffic and
fixed-work envelope. Evidence: `JOINT2/bounded_fuse/`.


## F-214 — R6 rebases attribution and bounds its causal interpretation

✅ **Verified.** B1 contains 7500 fresh processes: six cells, ten
placement/protocol configurations, five arms, 25 rotated rounds. All 1500
full arms pass correctness; unsafe arms retain their timing records. All
60 full-arm trace dumps are present. The R1-compatible pooled median of
100 reference rotate/legacy pairs is 0.895931995 [0.870870871, 0.976366322],
versus historical 0.6705. Real-width's separate 50-pair statistic is
0.963573913 [0.963184743, 0.967151369]. No ratio is normalized or filtered.

The six individual rotate/legacy medians are 0.756756757, 1.048076923,
0.867256637, 0.980922393, 0.962977702 and 0.967801254. With geometry
and selected placement fixed, R3-B on/off medians are 0.896026354,
0.962955927, 0.895833333, 0.951038576, 0.975695806 and 0.977504432.
These are conditional paired comparisons, not independent multiplicative
contributions. The complete five-arm differences and binding floors are
in `REBASE/bounded_analysis/`; signed differences, nonpositive barriers,
and original hardware telemetry are retained.

✅ **Verified control discrepancy; cause unresolved.** At the required W=1,
all six C2/C3 kernel sets are byte-identical. Nevertheless the original
mha4-s128 C3/C2 ratio is 1.098168734 [1.096054889, 1.099957100]. An
independent 100-process diagnostic rotates four labels, two for each
unchanged executable. C3/C2 becomes 1.000243380 [0.996597648, 1.002112713];
C2's duplicate-label ratio is 1.000000000 [0.974068071, 1.002002833], and
C3's is 0.998732449 [0.970772337, 1.001898457]. All 100 pass numerical
checks. This repeat does not replace the original 7500 samples or their CIs.

⚠️ **Unresolved causal attribution, stopped under §9.2.** The apparent
original C2 gain/C3 penalty cannot be assigned to local-dependency behavior.
No original sample is rejected. The next specific diagnostic controls CUDA
module/instruction addresses and allocation/runtime context while retaining
same-binary duplicate labels; clock/power association alone is insufficient.
This limits mechanism interpretation, not the reported raw comparison or
the independent J gate calculations. The real-s128 probe audit also checks
unchanged L1 under nofence and unchanged L2 under l1nosync; all 250 barrier
differences remain, including 25 nonpositive values. Evidence:
`REBASE/bounded_raw/`, `bounded_w1_identity/`, `bounded_w1_repeat/`,
`bounded_probe_audit/` and `bounded_analysis/`.

## F-215 — R7 separates the two precision defects from the R6 numerical failure

✅ **Verified.** Both defects R6 recorded are real and both are fixed. The
normalization epsilon is no longer the hardcoded `1.0e-6f` of
`RMSNormTaskBody.h`: it is read off the `add(variance, eps)` literal inside each
matched normalization, carried in `ModelPlan::norm_epsilon`, generated as
`TILEMEGA_NORM_EPSILON` and cross-checked against `ModelSpec` at run time. A
Llama-3.2-1B configuration now generates `1e-05f` and a Qwen3-1.7B one
`1e-06f`, from the graph rather than from a model name or a dtype. The rotary
phase is no longer rounded to model storage: when the exported frequency table
is FP32 the position, the table read and the angle stay FP32, and only the
cosine and sine are rounded once, which is what the reference implementations
do before they multiply.

✅ **Verified.** Neither defect explains F-203. Four builds of R6's frozen
covered graph -- baseline, epsilon only, RoPE only, and both -- give the same
`V[0,463] = -0.44921875` and the same single failing output of 66
(`MODELS2/ablation/admitted2/ablation.tsv`). This is structural, not a null
result: `MODELS/export_covered.py` cuts the embedding, both per-layer
normalizations, the rotation and the final normalization out of the covered
region, so that graph contains no `kRMSNorm` and no `kRoPE` stage at all (113
`kGemm`, 32 `kAdd`, 32 `kKVAppend`, 16 `kAttention`, 16 `kElementwise`). An
epsilon no stage reads and a rotation no stage performs cannot move an output.
R7 §1(三) attributes the A1-subset failure to these two defects; for this graph
that attribution does not hold, and the ablation is the evidence.

Evidence: `MODELS2/ablation/admitted2/{ablation.tsv,*/run.json}`,
`MODELS2/subset.md`.

## F-216 — R7 admits the maximal connected Llama graph by settling one rounding

✅ **Verified.** `V[0,463]` of the first layer is `v_proj` applied to a graph
input: one 2048-term FP32 dot product, with nothing upstream. Its FP64 value
-0.450195362966042 sits 5.05e-8 -- about 1.69 FP32 ulp -- from the BF16 midpoint
-0.4501953125, against a realistic FP32 K-loop error some thirty times larger.
No association of that sum decides the rounding reliably.

✅ **Verified.** `TILEMEGA_MIDPOINT_REFINE` recomputes in FP64 exactly those
elements whose accumulator lies within `TILEMEGA_MIDPOINT_GUARD` (2^-6) of a
BF16 ulp of a midpoint. A BF16 product is exact in FP64, so only the FP64
summation error remains, around 2^-40 of the FP32 one. Both rounding sites are
covered: the unsplit CUTLASS epilogue and the split-K combiner, the latter
through a new `GemmInvocation::k_total`, since a chunk cannot recover the
undivided K. With it the whole first-layer V matches its FP64 rounding in every
element, all 66 outputs pass, and 50 fresh processes are 50/50 with one binary
-- tolerance 0.0231875014, output set, seed and residual edges unchanged from
the run that failed. The switch is off by default, so the generated code without
it is what it was before the pass existed.

Evidence: `MODELS2/admission/admitted2/correctness/r{0..49}.{log,json}`,
`MODELS2/ablation/admitted2/refine/run.json`.

## F-217 — R7 fills the three operator gaps, at a higher change cost than audited

✅ **Verified.** Three families landed. The token embedding is a new
`TaskKind::kEmbedding` owning one token row; its identifiers keep the exported
index tensor's own width (`TILEMEGA_TOKEN_ID_BITS`, 64 for a `torch.int64`
export) and occupy that many model elements, because widening an index to BF16
storage would alias vocabulary rows above 2^8. The per-head query/key
normalization is a new `TaskKind::kQKNorm`: the arithmetic is the row body's,
but the ownership is one (token, head) rather than one token, so `OwnershipOf`,
the device `ActiveBlocks`, the host task count and the runtime projection all
extend. The final normalization is recognized after the last layer's residual,
outside `DecoderLayerPattern`, and emitted as a stage of its own, together with
the vocabulary projection that reads it.

⚠️ **Stated: the audited cost was an underestimate.** R6's
`MODELS/extension_sites.tsv` lists 15 conditional sites. The embedding alone
touched 19, and four of them are outside that table: `PlanTaskKind` is a
separate enumeration from the plan role; the CG's known task kinds and the
arithmetic signature table each need an entry; and a generated per-family
runtime switch is required, which the table could not have predicted because it
is a consequence of the default-build SASS identity rule rather than of the
operator (F-218). `ScalarTaskWork.cpp`'s conditional was not needed: both new
families reuse `kTilePerBlock`. The QK normalization touched 16 sites, the final
normalization 2.

Evidence: `MODELS2/subset.md`, commits `ef71973ec`, `35d8751fb`, `66c6497a4`.

## F-218 — A new TaskBody family costs default-build SASS unless it is generated

✅ **Verified.** Adding `case TaskKind::kEmbedding` to the three device
dispatches moved the two reference models' default-build SASS: branch targets
shift by 0x60 and the diff runs to 54747 lines, although no reference model has
an embedding stage. The switch is over a runtime value, so an unreachable case
is still code. Compiling the case out behind a generated
`TILEMEGA_EMBEDDING_RUNTIME`, which Codegen defines only when the plan names the
family, restores byte identity: the SASS then differs from baseline only in
nvcc's compile-path identifier string. `TILEMEGA_QK_NORM_RUNTIME` follows the
same rule.

✅ **Verified.** With both families off by default, `gqa2.cu` and `mha4.cu`
compiled against the baseline tree's headers and host archive and against
HEAD's are byte-identical in SASS.

## F-219 — R7 counts a gather by the row it reads, not by the table it might read

✅ **Verified.** The embedding's table read is data dependent: the row index is
a value, so no rectangle expresses it and `ElementAccess` refuses it. The whole
table is the only sound rectangular cover, which is what the coupling
derivation uses and what sends the operator to the I2 relaxation -- but read as
traffic it overstates the work by the whole vocabulary, which would dominate
the solve. The operator therefore declares a complete element read: the count is
one row of `hidden` elements per token, which is exact. `DeriveTaskWork` now
skips the rectangular projection for a data-dependent operand whose tensor has
such a declaration, since the exact pass already replaces the count.

⚠️ **Stated.** The declared read carries a zero offset on the vocabulary axis.
Only its cardinality is consumed; the location fact stays in the data-dependent
operand map. A consumer that reads this relation as a footprint rather than as
a count would be wrong, and nothing currently does.

## F-220 — R7's whole-decoder path runs, and it is what first exercises A1 and A2

✅ **Verified.** `MODELS2/export_full.py` exports a complete decoder -- token
embedding, every layer, the final normalization and the vocabulary head, with
the rotary table computed on the host per config (`rope_scaling` applied) and
registered as an FP32 buffer. `tilemega-compile` imports, solves, writes back
and generates it in one command. A Llama-shaped model generates
`TILEMEGA_NORM_EPSILON 1e-05f`, `TILEMEGA_ROPE_FP32_PHASE 1`,
`TILEMEGA_TOKEN_ID_BITS 64` and three `kRMSNorm` stages; a Qwen3-shaped one
generates `1e-06f` and two `kQKNorm` stages. Both run correct against their CPU
golden with L0.5, L1 and L2 agreeing bit for bit.

✅ **Verified: the FP32 rotary path was unreachable before this round.** The
importer required every model input to agree with the model's storage dtype, so
an FP32 frequency table -- the only thing that sets `rope_fp32_phase` -- was
rejected before the layer loop could look at it. The check is now deferred: the
identifiers and the phase table are named as the two legitimate exceptions by
the loop that consumes them, and any other disagreement still throws.

⚠️ **Stated.** Two frontend robustness fixes were needed and both are
architectural facts, not accommodations: `DecoderLayerPattern`'s input
normalization is now unordered, because the published modeling code writes
`self.weight * hidden_states` while the archived reference graph scales then
weights; and `aten.reshape.default` and `aten.slice.Tensor` are layout-only.

## F-221 — R7 localizes the real-model solve cost to the outer bound pass

✅ **Verified.** The full-width Llama-3.2-1B graph (1663 FX tasks, 2174
couplings, 47 guards) did not finish solving. After 23 minutes of CPU the search
evidence file `auto.cu.search.tsv` was still empty: the run had not reached the
capacity-bounded search at all.

✅ **Verified: the cost is in the outer bound pass, before the capacity gate.**
`SolveExport` (`include/tilemega/Solver/CompilerSearch.h`) builds its candidate
list by iterating the geometry domain crossed with `split ∈ {1,2,4,8,16,32}`,
and for each pair it calls `TorchExportImporter::Import(path, ...)` on the `.pt2`
again and then `PreparePlacementProblem` and `PreparePlanBounds`. Each iteration
therefore re-parses the export and re-prepares the whole model; the per-iteration
`IMPORT_DEGRADED` line is what makes this countable from the log. With this
round's 5-geometry domain that is 30 full imports and 30 preparations before one
candidate is evaluated, and `--search-capacity` bounds only what comes after.
Timed directly: the import counter advanced 6 -> 7 over 180 s, so one outer pair
costs about three minutes on this graph, against the 178 ms R6 measured on the
reference models. Thirty pairs is therefore roughly 90 minutes before the first
candidate is evaluated, which is why the run shows no search row rather than a
slow one.

✅ **Verified: the per-pair cost is the import, and most of it is not
hoistable.** `Import` is `ImportBridgePlan(ReadExportBridge(path), ...)`, whose
stages split at granularity: `ReadExportBridge`, `BuildModelPlan` and
`LiftSemantics` do not read `ImportOptions::gemms`, while `Instantiate` and
`CouplingDerivation::Derive` do. Running `tilemega-import` alone on this graph
took over 2.5 minutes against a per-pair cost of about 3, so the import
dominates the outer iteration and the granularity-dependent derivation dominates
the import.

⚠️ **Inferred next implementation, in this order.** (a) C1-b as written: keep
relation intervals and shared successor regions through the bound computation
instead of materializing every dense edge. This is the fix for
`CouplingDerivation::Derive`, which is where the cost actually is. (b) Hoisting
the parse, the plan build and the lifting out of the outer loop is still correct
and still worth doing -- R6 did the equivalent for the inner evaluation (F-191)
and the outer bound pass has none of it -- but it saves only the prefix ahead of
granularity, not the 30x the loop structure alone would suggest. An earlier
revision of this entry had (b) first with a 30x ceiling; that ordering was wrong
and is corrected here rather than silently dropped.

⚠️ **Correction, recorded rather than rewritten, per CLAUDE.md
(2026-09-19).** The first paragraph's "did not finish solving" was read off a
run still in progress. That run completed: exit 0, `elapsed_ns 12982079965549`
(3.606 h) in `/root/r7_work/llama/solve.json`, 361 rows in
`auto.cu.search.tsv`, `SOLVE_SUMMARY evaluated=12 deferred=69`. The search was
reached and bounded by capacity, not starved before it. Everything the
paragraph infers from the empty file -- that the outer pass is where the cost
is -- survives, because the 81 candidates and the three and a half hours say
the same thing; only "did not finish" is wrong.

⚠️ **Correction: the ordering in the inferred next step is backwards.**
(a) attributes the cost to `CouplingDerivation::Derive` and expects C1-b to fix
it. C1-b is now implemented and measured (F-229): the bound stage it removes is
0.11% of preparing this graph, and of 41 profile samples **none** is in the
enumeration it removes while 32 are in `BuildModelPlan`'s pattern matcher --
which (b) names and (a) discounts. The attribution to `Derive` was inferred
from the stage split, not sampled, and is not what the sampler sees. (b) is the
lever; C1-b is kept on its own merits (exact, up to 3.91x on the bound stage,
and it is what makes the bound pass stop building an edge set it never reads).

Evidence: `/root/r7_work/llama/solve.log` (not committed: the export and its
fixture are 5.6 GB), `E2E_REAL/summary.md` §4 and §11.
Corrections' evidence: `/root/r7_work/llama/solve.json`;
`E2E_REAL/prepare/solve_profile.tsv`; F-229.

## F-222 — R7 B0 completes the exposed-wait denominator; FORK7 clears 0.15 by 0.0005

✅ **Verified.** The line the R7 prompt §5.1 requires, verbatim:

```
FORK7 rule=1 whole_pipeline_exposed_wait_share=0.151 gemm_share=0.149 simt_share=0.002 cells=4
```

FORK6's `cp_kloop_wait_share=0.278` was quoted against critical-path **GEMM
mainloop** cycles: it excluded every SIMT body, and within a GEMM it excluded
setup, the first-operand wait and the epilogue. B0 keeps FORK6's four cells and
their frozen `selected` configurations verbatim and widens the denominator to
the whole critical path — every task's `run` interval, all kinds. The two
numbers are one measurement under two scopes, not a disagreement.

✅ **Verified: the SIMT probe adds no synchronization.** `TILEMEGA_TRACE_SIMT`
brackets barriers the bodies already executed — `RMSNormTaskBody::RunRow`'s
reduction barriers, `AttentionChunkTaskBody::RunTask`'s two — with `clock64`
reads on thread 0 only. No atomic, no new barrier, no store inside a polling
loop; the instrumented body issues exactly the barriers it issued before.
`QKNormTaskBody` delegates to `RunRow` and inherits the probe. The macro
defaults to 0 and `#error`s without `TILEMEGA_TRACE_PHASE`; default-build SASS
identity is stamped in `E2E_REAL/sass_identity/`. Correctness on the traced
build: 200 fresh processes (50 per cell x 4 cells), 200/200
`RESULT status=PASS`, all exit code 0, one recorded session.

⚠️ **Verified but not robust: the margin over the gate is 0.0005.** The gate and
the statistic were fixed before measurement (threshold 0.15; per-cell median over
fresh processes, then median over cells — FORK6's aggregation). It clears:
unrounded **0.15055** against 0.15, rule=1. The emitted line carries three
decimals, so reading "0.151 against 0.150" as a 0.001 margin overstates it
twofold. Per H6 and this file's rule the threshold is not moved and the margin is
recorded. With four cells the median is the mean of the two middle ones, mha4 s4
(0.1275) and gqa2 s128 (0.1736), whose own round-to-round ranges (0.1240–0.1322,
0.1695–0.1767) are each roughly fifteen times the margin. Taking one round at a
time, the four-cell median clears 0.15 in **6 of 9 rounds** (min 0.1485, max
0.1521); the envelope from each cell's round extremes is [0.1468, 0.1544]. rule=1 is the honest reading of the pre-registered statistic,
but the measurement does not separate this pipeline from the threshold, so B1
must not be justified by this line alone.

✅ **Verified: the verdict does not depend on which denominator was chosen.**
"Whole pipeline" was fixed in advance as the critical path, because that is
FORK6's denominator and keeping it is what makes the two lines comparable. The
same numerator over **all** tasks rather than only the critical path gives 0.176
(per cell 0.1302, 0.1715, 0.1796, 0.1842). The critical-path figure is the one
the line reports; the alternative is disclosed rather than substituted, since
choosing after seeing both is what H6 forbids. That the weaker of the two still
clears is a better argument for rule=1 than the 0.0005 margin is.

✅ **Verified: where the wait actually is.** Median over rounds, then over cells:
0.116 of the path is GEMM wait **inside** the K-loop (a CTA on its own
`cp.async` and its own rendezvous — intra-task, which cross-task pipelining does
not reach), 0.033 is the GEMM first-operand wait (`setup_end ->
first_operand_ready`, which is the head and is reachable), 0.002 is SIMT barrier
wait. The head component is sequence dependent: 0.041/0.043 at s4 against
0.011/0.025 at s128. So rule=1 does not say B1 recovers 15%.

⚠️ **`simt_share=0.002` is a lower bound and must not be read as "the SIMT
bodies are busy".** Two structural limits: thread 0's barrier time is a lower
bound on CTA idle time, because a thread that arrives last waits for nobody; and
a body with **no barrier at all** reports exactly zero by construction, not by
measurement — `raw/segments.tsv` shows `barriers=0` for rope, kvappend and
elementwise, whose loads still stall. A load-to-use bracketing was designed and
rejected: it needs `memory`-clobbered stamps that serialize the load against its
use and inflate the quantity being measured.

⚠️ **Inferred: B1 is sized against the head share, not the wait share.** SIMT
bodies occupy 35–45% of the critical path and their heads (`setup + wait`) are
3.5–8% of it; GEMM heads are 21%. rope and kvappend spend 38–55% of their body
in `setup` alone, and rmsnorm's epilogue is 27–31% of its body. That is the
overlappable region — work that runs too late rather than idle time, a different
claim and a different fix — and it is large where measured idle time is near
zero. FORK7 is architecture specific; the sm_89 line does not carry to sm_120
(`PHASE2/run_sm120.sh` is write-only per H7, self-checked here with
`SELF_CHECK=1`).

Evidence: `PHASE2/summary.md`, `PHASE2/raw/fork7.txt`, `PHASE2/raw/cells.tsv`,
`PHASE2/raw/analysis.tsv`, `PHASE2/raw/segments.tsv`.

## F-223 — F-40's closed form needs a 1 KiB per-CTA term before it can budget shared memory

✅ **Verified on RTX 4090 / sm_89.** F-40 predicts the driver exactly on 52 of
the 56 (cell, page) pairs swept for R7 B1-b. The four it misses are the same row
in every cell: an appended page of 4096 B, i.e. `smem=20480`, where
`102400/20480 = 5` divides exactly. F-40 says 5 CTA/SM; the driver says 4.

The driver reserves shared memory per CTA on top of the request. Adding one
kibibyte makes the form exact on 56 of 56:

```
ctas_smem = floor(102400 / (smem + 1024))
```

Bisected on gqa2 s128: `dyn=19456` holds 5 CTA/SM and `dyn=19584` drops to 4,
and `(19456 + 1024) * 5 = 102400` exactly.

⚠️ **Why F-40 never saw it.** Its sm_89 fit was 605 register-bound, 150
smem-bound and 322 tie, and the term only bites when the shared-memory limit is
the binding one *and* the request lands on an exact boundary. That is precisely
the regime a shared-memory prefetch page moves these kernels into. Quoting F-40
unamended would have authorised a 4096 B page and cost 20% of residency
silently — the grid is sized from this number
(`ModelHarness.cuh:2931-2946`), and an unmeetable `RESIDENCY_CAP` is rejected
rather than lowered.

✅ **Verified: the four R7 cells are register bound at 5 CTA/SM with 128-thread
CTAs**, so the free page is 3072 B on all of them, and 84992 B on the two s4
cells, which run at `RESIDENCY_CAP=1`. Anchored at page 0 against the harness's
own `E2E_RESOURCE` line on the runs that executed; the driver agrees for both
the L1 and L2 kernels on all four cells. The instrumented build (B0's probe) and
the production build differ by up to 10 registers and both give 5, so B0's probe
cost no occupancy.

⚠️ **Stated only for sm_89.** The opt-in cap, the reservation and the register
file all differ on sm_120.

Evidence: `PIPELINE/summary.md`, `PIPELINE/raw/occupancy.tsv`,
`PIPELINE/raw/occupancy_cells.tsv`, `PIPELINE/occupancy.py`,
`PIPELINE/occ_probe.cc`.

## F-224 — R7 lifts §8.6's union lifetime for two appended pages, and residency survives

✅ **Verified on RTX 4090 / sm_89.** R7 B1 is the one round allowed to change
the §8.6 invariant, and the change is narrower than "the union is gone". The
union is untouched: one explicit union, capacity `max_i(sizeof(SharedStorage_i))`,
lifetime the whole dispatch. What is new is **two pages appended after it** —
`kSmemBytes = sizeof(TaskSmem) + 2 * TILEMEGA_PREFETCH_PAGE_BYTES`, alternating
on `slot & 1`. The pages cannot join the union and cannot take a max: the
earlier slot's epilogue and the later slot's operand fetch have to hold their
own page at the same time, which is exactly the overlap. So what is lifted is
the assumption that shared memory has one lifetime scale, the dispatch; a page's
lifetime spans two adjacent slots.

✅ **Verified: occupancy before and after, against F-223's amended closed form
and against the driver.** Six cells × three arms = 18 rows, all built from the
sources and macros the timed binaries use; the driver's
`cuOccupancyMaxActiveBlocksPerMultiprocessor` and the closed form agree on
18/18.

| cell | smem before → after | appended | regs before → after | CTA/SM before → after | residency | keeps it |
|---|---:|---:|---|---:|---:|:--:|
| gqa2 s4 | 16384 → 18432 | 2048 | 94 → 96 | 5 → 5 | 1 | ✅ |
| gqa2 s128 | 16384 → 18432 | 2048 | 86 → 96 | 5 → 5 | 5 | ✅ |
| mha4 s4 | 16384 → 18432 | 2048 | 94 → 96 | 5 → 5 | 1 | ✅ |
| mha4 s128 | 16384 → 18432 | 2048 | 85 → 88 | 5 → 5 | 5 | ✅ |
| real s4 | 16384 → 32768 | 16384 | 85 → 88 | 5 → **3** | 2 | ✅ |
| real s128 | 16384 → 32768 | 16384 | 144 → 146 | 3 → 3 | 2 | ✅ |

The `Prefetch`/`Wait`/`Compute` split costs 2 registers on four cells, 3 on two,
and 10 on `gqa2 s128`; no cell crosses a register-bound residency step.
`keeps_residency` is 1 on 18/18: every arm still fields the residency its cell
was selected at, which is what makes B1-e a comparison R6 would also have
accepted. `real s4` is the instructive row — it gives up two driver CTAs and is
still fine, because its `RESIDENCY_CAP` is 2; at a selected residency of 4 or 5
the same page would have been rejected at launch rather than quietly lowered.

✅ **Verified: the page has to hold a whole row, or the mechanism compiles and
issues nothing.** `PrefetchBytes` refuses a copy wider than the page. The
reference models' hidden 512 fills the 1024 B default exactly; the real model's
4096 B scale row needs 8192, and at the default the real cells report
`E2E_PREFETCH slots=12208 declared=32 issued=0 queue_heads=0 page_bytes=1024` in
three fresh processes that still pass — the arm runs the instruction sequence
with no copy in it (`PIPELINE/raw/real_s4/page_probe/`, built by
`PIPELINE/page_probe.py`). This is the harness counter, not an inference:
across the six cells at their correct pages, `declared == issued` on every one.

⚠️ **Stated only for sm_89**, and only behind `TILEMEGA_PREFETCH_RUNTIME`, which
defaults to 0; the default build appends no bytes and its SASS is stamped
identical to the baseline tree (`E2E_REAL/sass_identity/`).

Evidence: `TileMega_skeleton.md` §8.6, §5.3.1; `PIPELINE/summary.md` B1-b;
`PIPELINE/raw/occupancy_arms.tsv`; `PIPELINE/occupancy.py`;
`PIPELINE/raw/*/correctness_head/`.

## F-225 — pipelining is a dimension of sigma, and its credit is a bandwidth share

✅ **Verified (host unit test `pipeline_sigma`, `test/unit/pipeline_sigma_test.cpp`).**
R7 §5.2(b) asks that pipelining be part of the placement decision rather than a
post-pass. It is wired in three places, each with an example that fails if the
wiring is removed:

```
PIPELINE_FRONTIER stages=102 with_frontier=68 rope_element_reads=12
PIPELINE_PRICE tasks=232 priced=16 mean_share=0.016181 page512_priced=0
PIPELINE_REALWIDTH page1024=0 page4096=0 page8192=0 stages=209 prefetch_bodies=0 stage_kinds=0:113 3:32 4:16 5:16 10:32 
PIPELINE_BOUNDS queue_lb binding_path makespan flags
PIPELINE_TABLE accepted rejected-head rejected-length
PIPELINE_SIGMA PASS frontier pricing bounds table
```

1. **The cost model prices the derived frontier** (`TaskModel.cpp:58-105`). A
   task is priced twice, reading its prefetch operand globally and locally, and
   the difference is the credit — granted only where the executor would issue:
   on the derived frontier, declared by the body, whole 16-byte lines, and
   inside the page the binary was built with. On `gqa2`, 16 of 232 tasks are
   credited at a mean 1.6% of their own time; re-prepared at a 512 B page —
   which the 1 KiB scale row does not fit — **0** are.
2. **The objective counts the overlap inside `max(CP, queue_lb)`**
   (`ExecutionSimulator.cpp:272-286`, `:594-635`):
   `pipeline_gain[next] = min(prefetch_ns[next], task_ns[n])` for each queue
   adjacency, applied to the simulated makespan, the queue lower bound and the
   binding path alike. R7's required distinction — same worker adjacent and
   pipelinable, against same worker but not — is the `queue_next[n] >= 0` gate:
   a queue head gets no credit, so splitting the same two tasks across two
   workers makes the credit vanish. That is what makes it a property of sigma
   and not of the task. The credit is clamped to the predecessor's body: a 4 ns
   fetch under a 1 ns task is worth 1 ns, not 4.
3. **The plan carries the flags back to CG** (`ExecutionSimulator.h:221`,
   `PlacementPlan.h:57-63`, `PlacementSolvePass.h:46-59`,
   `CGDialect.cpp:382-392`, `PlacementPlan.cpp:83-92`). The dialect verifier
   rejects `pipeline={1,1}` (slot 0 is a queue head) and `{0}` (must cover every
   node), so an inconsistent sigma fails verification rather than convention.

`PIPELINE_REALWIDTH` runs the same pricing over R6's admitted Llama graph at
1024, 4096 and 8192 bytes and credits **0** tasks at every page. That is not
the page refusing a wide row: the graph carries no body that declares a
`Prefetch` at all, so the gate closes one step earlier (F-228). The check
asserts the implication it cares about — no credit without such a body — and
prints the stage census so the reason is read off rather than argued.

⚠️ **The credit is a bandwidth share, not a latency model.** It is the
difference between pricing an operand globally and locally under the calibrated
cost model: no L1-hit term, no queueing term. It is therefore an upper bound on
what the overlap can be worth *in the cost model's own units*, and it is not a
prediction of measured gain — F-226 measures that, and the two do not agree.

✅ **Verified: the frontier is derived, never annotated.** `no_producer` is set
from the plan's write set at CG construction (`Frontend.cpp:227-229`) and the
read-only frontier is split out of each task's read work in `TaskWork.cpp`;
`Codegen.cpp:436-442` only emits what the CG already carries. Checked against
EX-E4's hand rule on **all 102 stages** of both reference models, 102/102
agreeing, with the element-wise RoPE reads (12) counted separately because they
read a phase table rather than a producer's output.

Evidence: `PIPELINE/summary.md` B1-d; `PIPELINE/raw/pipeline_sigma.log`;
`test/unit/pipeline_sigma_test.cpp`.

## F-226 — The cross-task copy is overlapped on every eligible slot, and the eligible slots are too few to pay for it

✅ **Verified (B0 phase trace, 900 fresh processes, 900 PASS; end-to-end, 900
fresh processes, 900 PASS).** R7 §5.2's B1-c asks for direct evidence that
adjacent slots on one worker overlap, and B1-e for the paired end-to-end ratios
with no threshold. The two disagree about the mechanism's worth, and the reason
is measured rather than argued.

**The overlap happens.** `prefetch_wait_cycles` brackets the issue and the
`cp.async.wait_group` in both timed arms, so they compare slot by slot. On the
issuing slots the wait collapses:

| cell | issuing slots | prefetch wait | inline wait | difference | net per issue |
|---|---:|---:|---:|---:|---:|
| gqa2_s4   | 16   | 430.8  | 2134.5 | 1682.5 | 96.0 |
| gqa2_s128 | 512  | 483.0  | 2090.5 | 1591.5 | 682.5 |
| mha4_s4   | 32   | 448.5  | 2196.5 | 1665.5 | 87.0 |
| mha4_s128 | 1024 | 488.5  | 2236.5 | 1707.2 | 2298.2 |
| real_s4   | 32   | 1179.5 | 3689.5 | 2578.5 | 2759.5 |
| real_s128 | 1024 | 299.0  | 3347.0 | 2936.8 | 1527.8 |

Cycles, not nanoseconds: `clock64()` is per-SM and the phase dump carries no
worker column, so the schedule has no common time base. Every figure is a
difference of two reads taken by one slot on one SM. The difference is positive
on **2640 of 2640** eligible slots, and on the slots the rule refuses the two
arms agree to within 5-9%, which is the instrument's floor with nothing in
flight.

⚠️ **On three cells the fetch is relocated, not removed.** The pipelined arm
issues slot `s`'s copy inside slot `s-1`'s body, where the wait bracket cannot
see it. Charged there, the predecessor's body grows by 1586.5 of the 1682.5
cycles the wait lost on `gqa2_s4` (94%) and 1578.5 of 1665.5 on `mha4_s4`
(95%), leaving 87-96 cycles net. `gqa2_s128` relocates 57%, `real_s128` 48%;
on `mha4_s128` and `real_s4` the predecessor's body is *shorter* in the
pipelined arm (-591, -181 cycles) and no relocation cost is visible. The two
cells where relocation is near-total are exactly the two whose end-to-end
`prefetch/inline` is furthest above 1 (1.0154, 1.0142), which is the direct
answer to why overlapping was slower there than not overlapping at identical
storage.

**The ceiling is one to three orders of magnitude below the price.** Taking
every cell at its best -- net recovery per issue, ignoring relocation where the
delta is negative -- times the slots that issue, against every slot's body
summed:

| cell | recovered | all bodies summed | share |
|---|---:|---:|---:|
| gqa2_s4   | 1 536     | 8 908 861     | 0.017% |
| gqa2_s128 | 349 440   | 151 397 291   | 0.231% |
| mha4_s4   | 2 784     | 20 122 179    | 0.014% |
| mha4_s128 | 2 353 357 | 342 350 643   | 0.687% |
| real_s4   | 88 304    | 2 292 666 857 | 0.004% |
| real_s128 | 1 564 467 | 2 552 256 443 | 0.061% |

The denominator is aggregate work across all workers, not a makespan -- a
per-SM clock cannot give one -- so with balanced workers this is a ceiling on
what the overlap could move end to end, not a prediction.

**End to end, paired, no threshold** (R6's `selected` as control, three arms
rotated within each round, 20000-draw bootstrap of the median, seed 20260919):

| cell | control median (ms) | prefetch/control | inline/control | prefetch/inline |
|---|---:|---|---|---|
| gqa2_s4   | 0.116736 | 1.0614 [1.0604, 1.0619] | 1.0442 | 1.0154 |
| gqa2_s128 | 0.244736 | 1.0293 [1.0291, 1.0295] | 1.0286 | 1.0004 |
| mha4_s4   | 0.228336 | 1.0673 [1.0640, 1.0678] | 1.0497 | 1.0142 |
| mha4_s128 | 0.502864 | 1.0326 [1.0310, 1.0334] | 1.0284 | 1.0040 |
| real_s4   | 3.879920 | 0.9918 [0.9913, 0.9923] | 0.9934 | 0.9981 |
| real_s128 | 6.524448 | 1.0225 [1.0215, 1.0239] | 1.0211 | 1.0009 |

Above 1 is slower: a **negative result on five of six cells**. `inline` pays
the same two pages and issues the same copy but overlaps nothing, and it
accounts for almost all of the regression (1.0286 of 1.0293, 1.0284 of 1.0326,
1.0211 of 1.0225). The cost is the mechanism's fixed price -- 2-10 extra
registers on every body and 2048-16384 appended bytes, F-224 -- paid by every
slot, while only 0.26-8% of slots are eligible to overlap at all (F-225,
`E2E_PREFETCH`). On `real_s4` that same fixed cost lands 0.7% on the other side
of zero at a 3.88 ms cell, which is a storage effect and not an overlap gain.

⚠️ Per R7 §0 item 2: this is a negative result under a mechanism that is
present and firing on every eligible slot, and it is **not** evidence that
cross-task pipelining is without value. What is measured is that *this*
page-based frontier prefetch, over a population of 0.26-8% of slots, cannot
repay the storage and the ABI split it charges to 100% of them. The quantity to
change first is the population -- which operands reach the frontier and how
many bodies declare one -- not the copy. The reference models' weights are
small enough to sit in L2, which is why §5.2 makes the real-width cells primary;
they are also the two cells with the smallest ceilings, 0.004% and 0.061%.

Evidence: `PIPELINE/summary.md` B1-c and B1-e; `PIPELINE/raw/overlap_head.tsv`;
`PIPELINE/raw/e2e_head.tsv`; `PIPELINE/raw/*/phase_head/`; `PIPELINE/overlap.py`.

## F-227 — The solver's page and the executor's page are one number kept in two places

✅ **Verified (host unit test `pipeline_sigma`; `E2E_PREFETCH` counters in the
B1 runs).** The prefetch credit is gated on the same three tests twice, once
where the objective is computed and once where the copy is issued:

| | solver | executor |
| --- | --- | --- |
| frontier | `input.prefetch_operand>=0` over `no_producer` | `no_producer[operand.buffer]` |
| alignment | `page[i]*bytes%16` | `width % 16u != 0u` |
| capacity | `fetched>prefetch->page_bytes` | `width > TILEMEGA_PREFETCH_PAGE_BYTES` |
| where | `lib/Solver/TaskModel.cpp:89` | `ModelHarness.cuh:1004-1015` |

The capacity row is the one that can disagree, because the two numbers are set
by different mechanisms: `--prefetch-page-bytes` on the solve
(`PlacementSolvePass.h:31`, default 1024) and `TILEMEGA_PREFETCH_PAGE_BYTES` on
the build (`ModelRuntime.h:36-37`, default 1024). Both directions were measured
rather than argued:

* **Solver page below the row.** Re-preparing `gqa2` at a 512 B page, which the
  1 KiB RMSNorm scale row does not fit, credits **0** of 232 tasks where the
  1024 B page credits 16 (`PIPELINE_PRICE … page512_priced=0`).
* **Executor page below the row.** The real cells' hidden 4096 makes an 8192 B
  scale row. Built at the default page, three fresh processes of `real_s4`
  report `E2E_PREFETCH slots=12208 declared=32 issued=0 queue_heads=0
  page_bytes=1024` and still pass: the ABI's `Prefetch` phase is present on 32
  slots and every copy is refused. That measurement is why `run.py`'s
  `PAGE` map lifts the real cells to 8192 and leaves the reference cells at the
  default — the page is chosen to hold one row, not chosen to be large.

The page is part of the solve's price-cache key
(`PlacementSolvePass.h:143`), so a page-matched re-solve is a real second solve
rather than a cache hit; F-228 relies on that.

⚠️ **Stated: nothing enforces the agreement.** The flag and the macro are set
independently, and today the only thing that reports a mismatch is
`E2E_PREFETCH`'s `declared`/`issued` pair after the fact. A solve run at a
larger page than the binary will count overlap the binary never performs, and
the objective will be optimistic by exactly the credited share. R7 leaves this
as a recorded gap: emitting the page into the generated source and checking it
at load would close it, and belongs with the ABI rather than with B1.

Evidence: `PIPELINE/raw/pipeline_sigma.log`; `PIPELINE/raw/real_s4/page_probe/`;
`PIPELINE/raw/*/correctness*/*/r*.log`; `PIPELINE/summary.md` B1-b.

## F-228 — The admitted Llama graph passes both arms 50/50 and cannot exercise the mechanism

✅ **Verified (100 fresh processes, `PIPELINE/raw/llama/correctness/`).** B1-a's
third population is the maximal connected Llama graph — R6's admitted geometry,
fixture, seed and tolerance, the same artefact R7's A-a re-admits
(`MODELS2/run_admission.py`). Replayed at HEAD and built with the mechanism at a
4096-byte page, **50/50 PASS on `prefetch` and 50/50 on `inline`**, no
mismatch. The gate is met.

✅ **Verified: and the mechanism has no client on this graph.**

```
E2E_PREFETCH slots=15486 declared=0 issued=0 queue_heads=0 page_bytes=4096 inline=0
PIPELINE_REALWIDTH page1024=0 page4096=0 page8192=0 stages=209 prefetch_bodies=0
                   stage_kinds=0:113 3:32 4:16 5:16 10:32
```

`declared=0`, not `issued=0`: this is a step earlier than F-227's page refusal.
`ScalarPrefetchOperand` returns an operand only for `kRMSNorm` and `kQKNorm`,
and the census above — 113 `kGemm`, 32 `kKVAppend`, 16 `kElementwise`, 16
`kAttention`, 32 `kAdd` — contains **no normalization stage of either kind**.
R6 admitted this graph by passing both per-layer normalizations in as graph
inputs, so they were never in it. The solver agrees for the same reason: the
pricing credits 0 tasks at 1024, 4096 and 8192 bytes alike, and the unit test
asserts the implication rather than the number (no credit without a body that
declares a `Prefetch`).

⚠️ **This is the round's real-width structural case, not its real-width timed
case.** EX-E4 asks that the benefit be judged at real width because a reference
model's weights may sit in L2. This graph is real width and credits nothing for
a structural reason; the real-width *timed* evidence is `real_s4`/`real_s128`
(F-226). A Llama graph that could exercise B1 is one whose norms are inside the
export — which is what R7's A4/A5/A6 make importable and what `MODELS2`'s full
export carries. Running B1 on that graph is not in R7's B1 scope and is left
recorded rather than done.

✅ **Verified: the replayed geometry moved, and R7's sigma is not why.** The
replay selects `64x128x16s2split2kappa1r3` where R6 recorded `split4`
(floor 2.22748e6 against 2.29662e6, predicted 2.37874e6 against 2.44849e6).
Two controls, both re-solves rather than cache hits — the page is part of the
price-cache key (F-227):

| solve | `auto.cu` | `auto.cu.search.tsv` | winner |
| --- | --- | --- | --- |
| default page 1024 | `1a42e943…` | `0c47b65e…` | split2 |
| `--prefetch-page-bytes 0` | `1a42e943…` | `0c47b65e…` | split2 |
| `--prefetch-page-bytes 4096` | `1a42e943…` | `0c47b65e…` | split2 |
| R6 `covered_llama_admitted2` | `dfbf24ad…` | `9974c571…` | split4 |

The three R7 solves are byte-identical in the source and the search table, and
their shortlists differ only in the output-path columns: every `floor_ns` and
`predicted_ns` of all three ranked rows is the same with pipelining priced at
zero as with it priced at the page the binary uses. That is exactly what
`prefetch_bodies=0` predicts. **Stated:** what did move the geometry is the 65
commits between R6's recorded source (`0039d4a9f`) and R7's baseline — 109 to
the tree this replay ran at. Recorded as a difference, not chased: B1 does not
need R6's geometry, it needs the same graph.

Evidence: `PIPELINE/raw/llama/`, `llama_sigma0/`, `llama_sigma_page/`;
`PIPELINE/raw/llama/regeneration.tsv`; `PIPELINE/raw/pipeline_sigma.log`;
`PIPELINE/llama.py`.

## F-229 — C1-b's interval bound is exact and up to 3.9x faster, and moves the real model's search capacity by nothing

✅ **Verified: the bound never needs the edges.** A dense piece of the projected
dependency relation `[consumer stage,task] -> [producer stage,task]` is a
complete bipartite block, and the longest-path relaxation such a block induces
is one maximum over its producers applied to each of its consumers. §6 C1-b is
that observation: `VisitFiniteRegions` now hands a dense piece to a `region`
callback as its two inclusive intervals, `PrepareRelationBounds`
(`include/tilemega/Solver/RelationBounds.h`) walks the same DAG
`PreparePlanBounds` walks but pays a block's side lengths instead of their
product, and `PreparePlacementProblem(module, options, &bounds)` skips filling
`graph.successors` altogether. Pieces with coupled coordinates or modular holes
still arrive edge by edge, so the two callbacks together see the same relation.

✅ **Verified: it answers the same two numbers.** `test/unit/relation_bounds_test.cpp`
(52/52 ctest, `relation_bounds` new) asserts agreement on a complete bipartite
block, wide dense slices, modular holes, overlapping pieces that repeat an edge,
a three-stage chain with a diagonal, the empty relation and a cyclic
self-reference, with per-node distinct costs so every longest path in the
fixture has its own length. On the nine real cells of
`E2E_REAL/prepare/prepare_bounds.tsv` the two arms agree on `work_ns` and
`critical_path_ns` to under 1e-9, `identical=1` on every row.

✅ **Verified: 1.03x-3.91x on the bound stage, no cell regressed.** Bound stage
only; `share` is that stage's part of the whole preparation, which
quasipolynomial task pricing dominates.

| cell | seq | nodes | edges | dense ms | interval ms | speedup | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa2_s4 | 4 | 5256 | 32784 | 21.9 | 21.3 | 1.03x | 0.88% |
| gqa2_s128 | 128 | 14592 | 1529728 | 449.4 | 227.4 | 1.98x | 15.44% |
| gqa2_s128 | 512 | 58368 | 21061120 | 2028.2 | 873.9 | 2.32x | 42.23% |
| mha4_s4 | 4 | 11760 | 74976 | 32.1 | 30.9 | 1.04x | 0.83% |
| mha4_s128 | 128 | 34816 | 4574080 | 1101.5 | 464.6 | 2.37x | 22.94% |
| mha4_s128 | 512 | 139264 | 65482240 | 3565.4 | 1201.0 | 2.97x | 35.25% |
| real_s4 | 4 | 24304 | 418016 | 52.1 | 47.9 | 1.09x | 0.37% |
| real_s128 | 128 | 60032 | 39726016 | 1110.5 | 284.1 | 3.91x | 6.94% |
| llama | 4 | 15486 | 217796 | 68.8 | 65.8 | 1.05x | 0.11% |

The two 512 rows are the `_s128` coupling graph re-projected at 512, not solved
cells of their own; they are here because the edge count is quadratic in seq.
The gain tracks edges per node, which is what the block representation removes:
2.2 edges per node at `gqa2_s4` buys 3%, 662 at `real_s128` buys 3.91x.

⚠️ **Verified negative: the search capacity stays 12.** §6 C1-b's acceptance is
whether the searchable space widens, and on the real model it does not, for a
reason that is measured rather than argued. One `PreparePlacementProblem` of the
full Llama graph at seq 4, past 3 costs **63.67 s**, of which the bound stage is
**68.8 ms — 0.11%**. Removing all of it would shorten an outer pair by about a
thousandth. The 41-sample profile `E2E_REAL/prepare/solve_profile.tsv` says
where the rest is: 32 samples in `PatternMatcher::DependsOn` /
`OperandConstraint` / `FxNodeRecord`, 9 in ISL quasipolynomial pricing, and
**none** in `isl_set_foreach_point`. §9.3's fallback applies as written --
capacity stays 12, recorded, and B2/B3 are not held.

Sampling stopped at 41 rows because `tools/tilemega-compile` was relinked while
gdb was attached to the still-running old image, after which every backtrace
came back with no top frames. The measured process kept its old image; only the
sampler's view of it broke. The discarded rows are not in the file.

⚠️ **Stated: `VisitFiniteRelation`'s own decomposition is unchanged.**
`VisitFiniteSliceRegions` now takes the slice-width threshold as a parameter;
the point-expanding wrapper keeps R6's 255, so `VisitFiniteRelation` emits the
same pieces in the same order and the default build stays bit-identical (H2).
Only the bounds client sees the finer decomposition, at 16: measured on this
machine an ISL slice query costs about 44 us against about 1.6 us per enumerated
point, so a slice must be about 28 wide to repay a client that expands it, while
a client that consumes the box profits at any width. 16 is the lowest threshold
that does not regress the short cells and the fastest on the long ones — 1 does
regress them, by 38% at `gqa2_s4`.

Evidence: `E2E_REAL/prepare/`, `E2E_REAL/prepare_bounds.cpp`;
`test/unit/relation_bounds_test.cpp`; commits `6500dab7e`, `1a72531b6`.

## F-230 — Building the model plan once halves the outer search's import cost, and capacity still stays 12

✅ **Verified: the plan does not depend on the import options.** `BuildModelPlan`
is the pattern match over the exported graph that F-229's profile put 32 of 41
samples in, and it reads neither the tile geometry nor the split. `SolveExport`
(`include/tilemega/Solver/CompilerSearch.h`) now builds it once and hands the
same plan to every import in the outer enumeration, the shortlist and the
per-stage refinement (commit `9083dd187`). Default build unchanged (H2).

✅ **Verified: 2.51x on the whole search, byte-identical evidence.** Two complete
searches of the Llama export at seq 4, past 3, capacity 12, same target and
domain: control `02e1a2c81` (plan rebuilt per import) 12982.1 s, hoisted
`f6b00ac11` 5167.2 s. The two `auto.cu.search.tsv` are byte-identical (361
rows, md5 `d42b6bb8c2df1fc55d25950f535093b9`), so the hoist moved the cost and
nothing else. The runs were not concurrent, so the ratio carries the machine's
load as well as the change.

✅ **Verified: 2.00x on the contention-controlled import rate.**
`E2E_REAL/prepare/import_rate.tsv` counts `IMPORT_DEGRADED` lines every 30 s in
the hoisted run and in a still-running pre-hoist process while both shared the
machine: over 4631 s, 36 imports (128.6 s each) against 18 (257.3 s each). The
control is at capacity 1 and the hoisted run at capacity 12; the two are
comparable because the outer enumeration imports one coarse module per
(geometry, split) pair regardless of capacity, and both were inside that
enumeration for the whole window. An earlier draft of this number, 3.44x, came
from a short early window and is superseded.

⚠️ **Verified negative, degraded form under §9.3: capacity stays 12.** §6 C1-b is
scored on the searchable space alone. One import of the real model still costs
minutes, and a capacity of 13 would add one more full solve of it to every
compile, so the round records the two speedups and leaves the capacity where it
was. `verify.py`'s C1-b row is `FAIL [hard]` marked `DEGRADED`; it is not
claimed as passed.

Evidence: `E2E_REAL/prepare/README.md`, `import_rate.tsv`, `imports.sh`;
commit `9083dd187`.

## F-231 — Per-stage kappa is wired end to end and moves nothing on the two reference models, because their winning family is stage-major

✅ **Verified: the mechanism.** `κ` is a per-producer-stage runtime field:
`PlacementSolveOptions::stage_kappa` projects each stage's events at its own
coarsening, the solved plan carries `tilemega.solved_stage_kappa`, codegen emits
`TILEMEGA_EVENT_KAPPA_TABLE` behind `TILEMEGA_EVENT_KAPPA_PER_STAGE`, and the
runtime reads a wait's coarsening from the table. The table is indexed by
*projected* stage, which split-K makes longer than the model's stage list
(gqa2 s4 split16: 30 codegen stages, 44 projected), so `PreparePlacementProblem`
refuses a table of the wrong length and the solver reports the length
(`SOLVE_STAGE_KAPPA stages=`). A pinned table is re-solved on the winner rather
than relabelled: a coarser event groups later producer tasks into the wait and a
slot order solved for uniform `κ` could put a consumer ahead of one of them on
its own worker. With the flag off nothing is written; default SASS identical
(H2). Commits `0f3273484`, `b2c6b404e`, `da8560ba6`, `3eae93344`, `2bc51b6e6`,
`6df89362f`, `be9f11dec`, `c51c838d3`, `c2a5a89c5`, `f6b00ac11`; ctest 52/52.

✅ **Verified: 200/200 fresh processes.** Two arms per model at seq 4, past 3,
capacity 12: the descent (`searched`) and a `1,2,4,...` table pinned over every
projected stage (`forced`, 44 stages on gqa2 and 88 on mha4). gqa2 50/50 and
50/50, mha4 50/50 and 50/50, the two arms of a model being different binaries.

✅ **Verified: no difference between per-stage and global `κ`.** The coordinate
descent (88 trials on gqa2, 176 on mha4, each a full placement solve) priced
every trial exactly at the incumbent: `moves=0`, `uniform_ns=per_stage_ns`
(169097 and 352991). §5.3 asked for exactly this to be recorded when it happens.

⚠️ **Inferred: the invariance belongs to the family, not to the pricing.** In
the same search the task-by-task families do move with global `κ` (`eft`
166272→174419 on gqa2, 348652→352810 on mha4; `chain` and
`legacy_grid_stride` likewise) while the stage-major ones do not (`wavefront`,
the winner on both models, and `rotate`). In a stage-major slot order every
producer of a consumer sits at a lower slot on every worker, so the wait is
resolved before the consumer's slot at any grouping. The winner is chosen on
`(floor, predicted)` and `wavefront` has the lower floor; `eft`'s smaller
predicted makespan at `κ=1` is not what the R6 ordering selects, and that
ordering is not changed here. A model whose winner is task-by-task is the first
place a non-trivial table would show.

Evidence: `E2E_REAL/stage_kappa/` (`README.md`, `results.tsv`, per-arm
`solve.log`, `auto.cu.search.tsv`, `correctness/`), `E2E_REAL/stage_kappa.py`.

## F-232 — Segmented geometry inside an interval is legal and materializes byte-identically, and the cut lands where the curves cross

✅ **Verified: one solve produces both arms.** `--seq-begin 1 --segments 2
--segment-candidates 4` on the gqa2 SEQSCAN export prices 4 candidate geometries
at all 16 integer points of `[1,16]` (`auto.cu.segments.tsv`, 64 rows) and emits
the segmented build (`auto.cu`, two variants) beside the fixed one
(`auto.cu.fixed.cu`, one variant) from the same invocation, exit 0 in 772.4 s.
The 15 `#define TILEMEGA_*` lines are identical between the two sources: the
difference is the plan table the runtime selects by `seq` (1487 against 920
generated lines), not the launch geometry. Commit `1d7f2884b`.

✅ **Verified: the cut is at the crossing, not at a midpoint.** `SEGMENT_SUMMARY`
reports `winner=32x16x64s2split8 winner_ns=2.68866e+06` (what the point search
picks at theta), `fixed=32x16x64s2split16 fixed_ns=2.67977e+06` (the best single
geometry over the interval) and `segmented_ns=2.67944e+06 cut=14`. At `seq=13`
`split16` prices 166892 ns against `split8`'s 166986; at `seq=14` it is 167552
against 167544, so 14 is the first point where `split8` is the cheaper of the
two and the segmenting search cuts exactly there.

✅ **Verified: legality at every point, on the S5 ISL path.** `segment_proof`
runs one fresh process per integer point: 16/16 `proved=1 failed=0`, each
reporting `bijective=1 dense=1 acyclic=1 resident=1` for its own segment's
geometry. The certificate is S5's, applied per segment rather than per interval.

✅ **Verified: endpoints and interior materialize identically.** `segment_check`
re-solves every point's table on the graph of the segment that owns it:
`segments=2 points=16 endpoints=4 interior=12 interval=1..16 kappa=1
residency=4 serialization=byte_identical`, with `diff=0` on all 16
`SEGMENT_MATERIAL` lines (5084 to 5152 plan nodes per point).

✅ **Verified: 500/500 fresh processes on gqa2.** Five seqs inside the interval
(1, 4, 13 below the cut; 14, 16 above it, both endpoints included) times two
arms times 50 rounds, one binary per arm across all five seqs (segmented
`632ba5073543a6c4`, fixed `f254ff53eeef2c8c`).

⚠️ **Inferred: the benefit is below what this cell can resolve, and that is
what the prediction says too.** 20 paired rounds per seq give
segmented/fixed 0.9926, 0.9986, 1.0147, 0.9997, 0.9999 at seq 1, 4, 13, 14, 16 --
straddling 1.0 with a 1.5% spread between seqs -- against a predicted 0.34% gain
over the point winner and 0.012% over the best single geometry. The candidate
set is the reason: `split16` and `split32` price *identically* at all 16 points,
the same exact `(floor, predicted)` tie the D-b counterfactual found along
`kappa`, so four candidates carry three distinct curves and the two that cross
do so shallowly. A model whose curves cross steeply inside the interval is where
a gain would show; §5.4 asks for the number and sets no threshold.

✅ **Verified: the second model repeats every part of it.** mha4 (88 projected
stages against gqa2's 44) over the same interval: `SEGMENT_SUMMARY
candidates=4 points=16 winner=32x16x64s2split8 winner_ns=5.64824e+06
fixed=32x16x64s2split16 fixed_ns=5.63044e+06 segmented_ns=5.62978e+06 cut=14`;
16/16 proof points `proved=1 failed=0`; `segments=2 points=16 endpoints=4
interior=12 serialization=byte_identical` with `diff=0` on all 16
materialization lines; 500/500 fresh processes over the same five seqs and two
arms (binaries `0e45849dc946412a`, `14cb03bd16da7e16`); and 20 paired rounds per
seq giving 0.9984, 0.9992, 1.0063, 1.0001, 1.0002. The proof is the long pole:
its 16 points ran in parallel for 11 hours, the tail points at seq 12-16 taking
more than 10 hours of CPU each.

⚠️ **Inferred: the cut is a property of the geometry pair, not of the model.**
Both models cut at 14, both pick `split16` below the cut and `split8` above it,
and on mha4 the crossing is again exactly there (`split16` 352687 against
`split8` 352876 at seq 13; 354007 against 353991 at seq 14). The relative gains
agree to the digit as well: 0.012% over the best single geometry and 0.33% over
the point winner on mha4, against 0.012% and 0.34% on gqa2. Two models that
differ by a factor of two in projected stages produce the same cut and the same
margin, which says the margin belongs to how close the split-K curves run, not
to the size of the graph.

Evidence: `E2E_REAL/segments/` (`README.md`, `correctness.tsv`, `timing.tsv`,
`gqa2_i1_16/` and `mha4_i1_16/`, each with `solve.json`,
`auto.cu.segments.tsv`, `proof/p*/`, `check.log`, `correctness/`, `timing/`),
`E2E_REAL/segments.py`, `E2E_REAL/segment_proof.cpp`,
`E2E_REAL/segment_check.cpp`.

## F-233 — The maximal connected Qwen3 graph runs and the megakernel matches the reference bit for bit; the CPU golden parts company with both at depth

✅ **Verified: the refusal, localized to one operator and one constant.**
Qwen3-1.7B's full decoder does not compile: `tilemega-compile: local reduction
requires an exact unit indexing axis: l0.s03.qknorm semantic l0.s03.qknorm
operand 0 axis 1 dim r index 128*floordiv(c, 128) + 1*r`.
`lib/Analysis/TaskWork.cpp:279-283` requires a reduction axis to be indexed by
exactly one unit term; the per-head Q/K RMSNorm reduces over `r` inside a head
while the flattened channel axis carries `128*floordiv(c, 128)`, an offset that
is invariant in `r` but not a unit term, with 128 being `head_dim`. The check
throws, so it aborts the solve rather than skipping the candidate.

✅ **Verified: hoisting only that operator does not produce a compilable
graph.** `MODELS2/export_full.py --hoist-qk-norm` feeds Q and K in already
normalized. The graph imports (`{"tasks": 2899, "couplings": 3738, "guards":
307}`) and then fails at `lib/Frontend/Frontend.cpp:819`, `explicit plan
requires stages and observable outputs`: `DecoderLayerPattern` does not match a
layer whose Q and K arrive as graph inputs, so `BuildModelPlan` falls through to
the covered-region path and returns no stages. Reproduced at 1, 2 and 28 layers.

✅ **Verified: the A-a cut list carries over and gives the graph.** Cutting the
token embedding, both per-layer RMSNorms, RoPE and the final RMSNorm -- exactly
`MODELS/export_covered.py`'s cuts, reused as a module rather than restated --
cuts the per-head normalization with RoPE, because it sits between the
projection and the rotation. What remains connected is the residual chain, the
V/cache/attention/O chain and the SwiGLU chain of all 28 layers plus the
vocabulary projection: import `{"tasks": 1096, "couplings": 1233, "guards":
196}`, `evaluated=12 deferred=75`, codegen `tasks=562 couplings=504 stages=365`,
winner `64x128x16s2split2kappa1r3` on `chain`, 2235 s of solve.

✅ **Verified: 50/50 fresh processes agree with the reference implementation
bit for bit.** Every round prints `E2E_HASH l05=40f7e79310c4b01c
l1=40f7e79310c4b01c l2=40f7e79310c4b01c` and `l1_vs_l05_mismatch=0
l2_vs_l1_mismatch=0`: one distinct `E2E_DIFF` line over all 50 rounds, 113 of
the 114 checked outputs exactly equal.

❌ **Not met: §4.5 A-b is 50/50 against the CPU golden, and that is 0/50.** The
one output that differs is the hidden state after 28 residual additions
(`buffer=728`): 190 of 8192 elements outside `1.6e-2 + 1.6e-2*|expected|`,
identically in all 50 rounds, `max_abs=0.09375`.

⚠️ **Inferred: chained bf16 rounding, priced.** The largest absolute
differences are 0.09375 on values of 2.5 to 4.4 -- 2.1% to 3.8% relative, five
to ten bf16 ulps -- and the offending elements are the smaller ones (median
`|expected|` 0.399 against 1.25 over all 8192). A depth sweep of the same
export, solved and run the same way, gives 0, 2, 44 and 190 elements outside
tolerance at 4, 8, 16 and 28 layers (5 fresh processes each, 50 at 28), with
`max_abs` 0.031, 0.047, 0.063, 0.094: the graph passes at depth 4 and the
divergence grows about as the square root of the depth. The tolerance is one
relative constant, so it prices a 28-deep chain of bf16 roundings the same as a
single layer. Both sides are legitimate bf16 evaluations with fp32 accumulation
inside each GEMM; the expected values are not moved, and the difference is
recorded instead. A-a's Llama graph at 16 layers passes 50/50 under the same
formula and the same `MIDPOINT_REFINE=1`, so the sweep establishes the trend
within one model rather than a universal depth limit.

Evidence: `E2E_REAL/qwen3/` (`README.md`, `solve.json`, `solve.log`,
`correctness/`, `depth.tsv`, `residual_cancellation.txt`, `dump_run.log`),
`MODELS2/export_covered_qwen3.py`, `E2E_REAL/residual_cancellation.py`.

## F-234 — The TaskBody ABI was bound to one architecture in a header, and nothing compared it to the device

✅ **Verified: the binding was a constant.** `ModelHarness.cuh` read
`using HarnessArch = cutlass::arch::Sm80;` for every build, so a Plan solved
for sm_89 compiled its bodies against the sm_80 capability table.
`TargetSpec` was already capability-driven and `Target/ArchDispatch.h` already
carried `Caps<Arch>` for Sm80/89/90/100/120; what did not exist was any path
from the solved target to the instantiation, and `lib/Codegen` never saw a
`TargetSpec` at all.

✅ **Verified: the architecture now travels with the Plan.**
`tmexec.solved_arch` is written beside the other solved attributes,
`lib/Codegen` emits `TILEMEGA_ARCH_ID` and `TILEMEGA_ARCH_TAG` behind the same
"compile option disagrees" guard the launch parameters use, the device pass
asserts the identifier against `__CUDA_ARCH__`, and `RunModel` compares the
device's own `major*100+minor*10` before any launch. A gqa2 cell prints
`E2E_ARCH plan=sm_89 plan_id=890 device=sm_89 device_id=890` and passes with
all three levels bit-identical.

✅ **Verified: a disagreement is a hard failure.** The same harness built
`-arch=sm_80 -DTILEMEGA_ARCH_ID=800` and run on this sm_89 device — which CUDA
JITs happily — prints the mismatch to stderr and exits **2** before any
launch. §4.1 asks for exactly that: no degradation.

⚠️ **Inferred: nothing changed on this machine.** `Caps<Sm89>` derives from
`Caps<Sm80>`, so binding to Sm89 selects the same capability set; what changed
is that the binding is the Plan's statement instead of a header's assumption.

Evidence: `BACKEND/be1_arch/` (`README.md`, `generated_macros.txt`,
`plan_arch_match.log`, `plan_arch_mismatch.log`).

## F-235 — CUTLASS 4.8 has no SM80-class CollectiveBuilder, and its sm_120 builder refuses BF16

✅ **Verified: the submodule is CUTLASS 4.8.0 at `dc45f979`**, and
`include/cutlass/gemm/collective/builders/` holds specializations for sm90,
sm100, sm103 and sm120 — and none for SM80 or SM89. The sm_80-class collective
exists only as `collective/sm80_mma_multistage.hpp`. R8 §3 anticipated the
opposite (an old submodule carrying only SM80) and asked for an upgrade in
that case; the upgrade does not apply, and the gap is on the development
machine's own architecture.

✅ **Verified: sm_120's builder rejects the model dtype.**
`CollectiveBuilder<Sm120, OpClassTensorOp, bfloat16_t, ...>` fails to
instantiate with "SM120 TmaWarpSpecialized builder currently only supports
F8F6F4" and "No MMA matches SM120_16x8x32_TN for given data types".

✅ **Verified: the GEMM the anchored models run was already a CUTLASS
collective.** `backend::GemmCandidate` builds `CollectiveMma` with
`MainloopSm80CpAsync` and, on the BF16 profile, the
`SM80_16x8x16_F32BF16BF16F32_TN` tensor-core atom with an FP32 accumulator.
What was missing is that the choice ignored the architecture entirely.

✅ **Verified: five architectures compile and self-check on the CPU.**
`arch::Caps<Arch>::kBf16CollectiveBuilder` — a capability of the toolchain,
established by the probe rather than assumed — selects the builder on sm_90
and sm_100 and the multistage collective on sm_80, sm_89 and sm_120. At
64x128x64 the builder returns 221440 bytes of mainloop storage on sm_90 and
196736 on sm_100, against 73728 for the multistage path.

⚠️ **Inferred, and declared: the builder path is compiled, not priced.** The
solver's `TensorBF16SmemBytes` closed form describes the multistage collective
only, so `TypedGemmCandidate::kPricedCollective` marks which of the two a
candidate is and the compile-time contract is asserted for the priced path.
Enumerating candidates the cost model cannot cost is a cost-model change, not
a backend one.

Evidence: `BACKEND/be2_collective/` (`README.md`, `arch_check.cu`,
`run_arch_check.sh`, `arch_check.tsv`, per-arch build logs), `BACKEND/porting.md`.

## F-236 — Attention spent 127 of 128 threads waiting, and the parallel rewrite is bit-identical

✅ **Verified: the defect.** `AttentionChunkTaskBody` computed the softmax on
lane 0 — three serial scans of the key sequence for the max, the exponential
sum and the normalization — while the other 127 threads sat at the next
barrier. `AttentionPhasedTaskBody::kNormalize` did the same behind
`if (threadIdx.x != 0) return;`, and `RMSNormTaskBody` (and through it
`QKNormTaskBody`) used a shared-memory tree costing log2(threads) barriers per
row.

✅ **Verified: all four now reduce across the warp.** `WarpReduce.cuh` provides
`__shfl_xor_sync` warp reductions and a two-stage CTA reduction; the rounding
points are unchanged, because the exported golden computes softmax in FP32 and
casts the probabilities to the model dtype, so rounding the probability is
required for agreement rather than an artifact. Internals are FP32.

✅ **Verified: bit-identical on the reference cell and exact against PyTorch.**
The gqa2 cell's output hash is unchanged across all four rewrites
(`50f243d42e025b16` for L0.5, L1 and L2 before and after), and the
per-operator comparison against PyTorch gives **0 mismatching elements out of
8192 for each of rmsnorm, qknorm and attention, with `max_abs` exactly 0.0** —
not merely inside the 1.6e-2 tolerance.

Evidence: `BACKEND/operator_check/` (`operator_check.cu`, `generate.py`,
`compare.py`, `operators.tsv`), `BACKEND/coverage.md`.

## F-237 — The release rule's barrier control cannot be made to fail on sm_89, so §8.5 stands

✅ **Verified: the compliant arm and one control behave as required.** Over
grid ∈ {64,128,256} × tiles ∈ {256,1024,4096}, 50 fresh processes per cell:
the role-granularity release (each writer fences, the producing role converges
on `bar.sync 1, 64`, one thread then publishes) passes **450/450**, and the
missing-fence control fails **every round of every cell**.

❌ **Not met: the missing-barrier control passes everywhere.** §5.3 requires
both controls to fail before §8.5 may be rewritten, so B-a is unmet and under
§8.3 the consequences are taken: §8.5 is unchanged, no TaskBody is
specialized, no harness barrier is converted, and the role path is recorded as
undelivered. `TaskRoles` stays declared and unused.

⚠️ **Inferred: latency closes the window, not the absence of the hazard.**
Five constructions were tried, each fixing a real defect in the previous one —
both roles in one CTA (every arm passed: one L1 makes fences unobservable),
cross-CTA publication (the compliant arm failed: the test raced itself), a
two-slot ring with an acknowledgement, the signalling thread owning one
element (F-1's own hazard), and a load-dependent store. In every one, each
writer's device-scope fence plus the global round trip of the flag means the
consumer observes the publication later than the writers' stores land. F-3 is
explicit: a control that passes is not evidence that the ordering it removes
is unnecessary.

⚠️ **Inferred: the experiment belongs on another part.** On sm_90 the two
roles are a warpgroup pair inside one CTA publishing through `mbarrier`, which
is a different scope from a global flag. That, or the cluster scope an sm_120
part offers, is where the control can fail. `BARRIER/run_sm120.sh` carries the
matrix.

Evidence: `BARRIER/` (`README.md`, `barriers.md`, `litmus.cu`,
`run_litmus.py`, `raw/litmus.tsv`, 1350 per-round logs).

## F-238 — Splitting the dialect turns a convention into a grep

✅ **Verified: one dialect became two, and the suite stayed green.** `tmcg`
holds `tile_space`, `event_tensor`, `coupling`, `fused_task_space` and the new
`graph` container; `tmexec` holds `placement`, `implementation` and the new
`plan` container. The module-level `solved_*` attributes moved to `tmexec`
with them, because they are decisions. ctest is **53/53** after the split, and
`tilemega-compile` still runs import → solve → write-back → codegen in one
command.

✅ **Verified: the ownership property is checkable.**
`DIALECT/ownership_check.sh` greps for op construction — `create<...Op>` and
op-name strings — and reports that `lib/Codegen` constructs no `tmexec.*` op
and that `lib/Solver` and the write-back pass construct no `tmcg.*` op. The
first version of that check was wrong in an instructive way: it matched
codegen *reading* `tmexec.solved_grid`, which is exactly what codegen is for.
Reading a decision and making one are different, and the check now
distinguishes them.

⚠️ **Inferred: the containers are defined but not yet emitted.** `tmcg.graph`
and `tmexec.plan` exist as ops with symbol-table regions; no pass wraps the
flat ops in them yet, because doing so touches all 48 `getOps<...>` walks and
every MLIR test at once. The split and the rename are what make the ownership
property real; the nesting is presentation of the same property.

Evidence: `DIALECT/` (`ownership_check.sh`, `rename.md`),
`include/tilemega/Dialect/CouplingGraph/{CGDialect.td,CGOps.td,ExecOps.td}`.

## F-239 — `MIDPOINT_REFINE` costs 4.4x on the reference models, and comparing with it against R7's numbers without it looks like a regression

⚠️ **This entry exists because the measurement was wrong first.** R8's first
A-g run reported gqa2_s4 at L2 0.673 ms against R7's 0.140, and the schedules
were identical (`workers=512 variant_stages=30 task_refs=5256 waits=18906`),
which reads as a 4.4x regression from the round's own backend work. It was not.

✅ **Verified: nothing in R8's backend changes costs time.** Same generated
source, same GPU, same session, one flag set at a time:

| headers | `MIDPOINT_REFINE` | L2 ms |
|---|---|---|
| R8 baseline `bb1902689` | off | 0.1535 |
| R8 head | off | 0.1516 |
| R8 baseline `bb1902689` | on | 0.6789 |
| R8 head | on | 0.6729 |

The head is marginally faster in both columns. Bisecting the rewritten bodies
one header at a time — `WarpReduce.cuh`, `RMSNormTaskBody.h`,
`AttentionChunkTaskBody.h`, `AttentionPhasedTaskBody.h`, `TaskBase.h`,
`TaskResources.h`, `CutlassGemmCandidate.h`, `GemmStageTaskBody.h`,
`ModelHarness.cuh` — moved the number by less than 1% at every step.

✅ **Verified: R7's own binary still reproduces.** `topk/gqa2_s4/bin/selected`
(`c2673ccef2a91c62`), untouched on disk since R7, gives L2 0.1568 ms today,
against the 0.1403 R7 recorded. The machine has not drifted.

✅ **Verified: the switch is the whole difference.** R7's reference binaries
were built with `measure.PROTOCOL` and nothing else; R8's first runner added
`MIDPOINT_REFINE=1`. Selective FP64 recomputation near BF16 midpoints (F-216)
is what the anchored Llama and Qwen3 graphs need for their numerical gate, and
the reference models pass without it. Rebuilding without it returns
gqa2_s4 to L2 0.1556 ms.

⚠️ **Inferred: the cost belongs to the GEMM epilogue, not to the harness.**
The three levels move together (L0.5 0.184 -> 1.033, L1 0.198 -> 1.048,
L2 0.156 -> 0.673), which is what a body-level cost does; a scheduling cost
would move L2 against L1. The switch is not free and should be priced as a
numerical-correctness cost wherever it is enabled — R7 enabled it on every
whole-model run, so the D-c/D-a timings carry it and the D-d reference
timings do not.

✅ **Verified: the switch, priced over the whole decode sweep.** Twelve cells,
both arms, every Plan reused from R7 rather than re-solved (`timing_sweep.py`).
L2 median, refine off -> on:

| seq | gqa2 | mha4 | llama |
|---|---|---|---|
| 1 | 0.137 -> 0.420 (3.06x) | 0.288 -> 0.833 (2.89x) | 4.645 -> 36.40 (7.84x) |
| 4 | 0.159 -> 0.748 (4.71x) | 0.344 -> 1.408 (4.09x) | 5.929 -> 41.10 (6.93x) |
| 16 | 0.214 -> 2.122 (9.91x) | 0.494 -> 4.268 (8.65x) | 6.820 -> 131.2 (19.24x) |
| 128 / 64 | 0.309 -> 4.641 (15.01x) | 0.656 -> 9.585 (14.60x) | 11.74 -> **422.4** (35.98x) |

The cost grows with the work per launch, which is what a per-GEMM-element
recomputation does.

⚠️ **Inferred: this relocates R8 §1's premise.** §1 cites Llama seq 64 at
380.2 / 380.3 / 424.0 ms as evidence that the backend is one to two orders of
magnitude off. Those numbers reproduce on the reworked backend to three digits
(379.5 / 379.6 / 422.4), so §8.4's global-stop condition is not triggered — but
the same cell without the refinement pass is **11.74 ms**, so roughly 97% of
the headline figure is the numerical switch rather than the backend's quality.
The switch is not optional on the anchored models: it is what makes A-a's
Llama graph pass. The caliber and the performance target are therefore coupled,
and R10's depth-aware comparison is a performance item as much as a
correctness one.

Evidence: `BACKEND/timing_sweep/` (`timing_sweep.tsv`, per-round logs for
12 cells x 2 arms), `BACKEND/models/`, `BACKEND/summary.md` §6, the bisect
commands in this entry.

## F-240 — Semantic coupling keys preserve access differences across reuse

✅ **Verified.** The R9 import split retains one immutable semantic model and
instantiates each granularity through `CouplingCache`. At split 1 and split 2,
`module.print` from the original importer, the cold cache and the warm cache
matches byte for byte. Two adversarial pairs retain the same operator kind and
shape but change indexing or `element_reads`; both the normalized signatures
and complete coupling keys differ. Renaming alone leaves the signature equal.
The fixture records 124 hits and 72 misses; these are unit-fixture counts, not
an anchored-model hit-rate claim.

✅ **Verified.** A deliberately incorrect wait cardinality of 999999 is rejected
with exact-expression memoization enabled. Restoring the original attribute
restores verification. Static Llama seq=1 also matches the original import at
both splits after preserving the original dimension-role defaults.

Evidence: `SOLVER_V2/cache_test.log`, `static_import_cache_test.log`,
`test/unit/coupling_cache_test.cpp`; implementation `CouplingCache.cpp` and
`lib/Frontend/Frontend.cpp`. Complete anchored solve timings remain pending.

## F-241 — Exact symbolic fibers survive the Oracle fast paths

✅ **Verified.** Independent enumeration agrees as sets on 384 predecessor and
successor queries, covering Unique, Rectangular and General relations. Another
45 cases check signed floor division and domain holes. The adversarial sets
include a two-dimensional lexicographic counterexample, strided images and a
million-wide sparse hull. The range cannot be replaced by its bounding box.

The implementation derives Unique from `isl_pw_multi_aff_from_map` and permits
Rectangular only after `is_box`. General may enumerate a bounded local scan,
but every emitted point passes the original exact Presburger predicate. Large
sparse scans use ISL point enumeration. Native evaluation of ISL-generated
integer expressions retains domain guards; unsupported operations or overflow
fall back to ISL. No ISL scheduling API determines placement.

Evidence: `SOLVER_V2/oracle_membership_test.log`,
`test/unit/symbolic_oracle_test.cpp`, `SymbolicOracle.cpp`,
`OracleExpression.h`. Unit set equality does not replace the required audits
on all anchored-model search outputs; those are still being collected.

## F-242 — Existing executor windows add order beyond exact CG data edges

✅ **Verified.** In the diagnostic projection, exact data dependencies contain
5136 pairs, while executor wait windows add 1280 pairs. A gqa2 seq=128 top-K
plan using only exact data dependencies was rejected by L-c before GPU
execution: a required event producer followed its consumer on one worker.
This was a plan-admission failure, not an observed numerical failure.

The repaired search separately represents the symbolic execution-order
relation obtained by composing requested events with exact event-group
membership. Its union with the CG data relation supplies the Oracle. The
extra pairs are not relabeled as data dependencies. Independent expansion at
kappa 1/2/4 agrees on 13760/13952/14336 pairs and the resulting queues respect
that order. Existing event and legality semantics are unchanged. This extends
the predecessor set specified in R9 §4.4/§4.6 and is explicitly declared in
`SOLVER_V2/implementation_notes.md`.

✅ **Verified.** Reusing geometry and relation proofs across grids 8 and 16
produces identical worker, slot, start and finish arrays to independent
preparation. A real Llama seq=4 seed retains the exact Level 2 score
5514820.9881833401 ns and residency 4 before and after that optimization.
This is score equivalence, not measured inference latency.

Evidence: `SOLVER_V2/resident_reuse_test.log`, `resident_reuse_score.json`,
`test/unit/plan_skeleton_test.cpp`; implementation `PlanSkeleton.cpp`.

## F-243 — Isolated coordinate evaluations preserve serial search decisions

✅ **Verified.** On a CPU search fixture, jobs=1 and jobs=3 produce the same
26 candidate records, top-five keys and all five final task schedules byte
for byte. Each path records one semantic import. Four independent scheduler
fixtures also preserve exact schedules under process isolation; child Oracle
queries do not mutate parent state, and worker exceptions propagate.

The optional process workers evaluate candidates with all other classes fixed.
Results are consumed in the original order, preserving improvement and tie
rules. Resource probes remain in the parent; real occupancy and full simulation
remain at the final top-K boundary. The default is one job. Primary real-model
runs use three jobs per coordinate sweep and retain the complete domain and
P=3. Stage totals sum worker durations; elapsed total is wall time, so stage
sums can exceed total. A serial legacy / parallel skeleton latency comparison
must disclose this resource difference.

Evidence: `SOLVER_V2/search_isolation_test.log`,
`isolated_evaluation_test.log`, `protocol.json`,
`test/unit/skeleton_search_isolation_test.cpp`. No anchored-model speedup or
G-8 pass is asserted by this fixture.

## F-244 — The reference skeleton plans retain internal bitwise equality

✅ **Verified.** gqa2 and mha4 at seq 4 and 128 each complete all three
shortlisted candidates in ten fresh processes: 120/120 processes have equal
L0.5/L1/L2 hashes and zero internal mismatch counters. All builds explicitly
disable `TILEMEGA_MIDPOINT_REFINE`. CPU golden differences do not decide this
gate. The current raw verifier reports G-3 PASS.

These reference searches use the declared one-shape diagnostic domain and
P=1. They establish G-3 correctness, not full-domain search quality or the
anchored-model G-8 research gate. The latter remains pending. Raw edge dumps
are preserved in deterministic gzip archives with decompressed SHA256 checks;
the verifier reads the raw rows from those archives directly.

Evidence: `SOLVER_V2/reference_correctness.tsv`, the per-process logs and build
commands under `SOLVER_V2/reference/`, and `reference_archive.log`.

## F-245 — Repeated symbolic parsing and ready-set construction are avoidable

✅ **Verified.** A sampled candidate stack reaches `ExactRuntimeDependencies`
through `CouplingRelation::Union` and `isl_map_read_from_str`: the former left
fold repeatedly parses an increasingly large prefix of already composed
edges. `UnionAll` instead parses each input once, combines live ISL maps in a
balanced fold, and serializes the result once. A symbolic equality check covers
35 inputs including duplicate and empty relations; ISL reference counts remain
balanced. The complete preparation-test output matches the preceding version.

✅ **Verified.** A ready tile's candidate set stays fixed because all predecessor
placements are final. Keeping that set across lazy EST retries leaves worker
availability dynamic. Three scheduler fixture digests, makespans, interleaving
fractions and retry counts remain unchanged. With both optimizations, the
26-candidate serial/three-worker search comparison again produces identical
candidate records and all five final schedules.

Primary experiments admit at most four simultaneous skeleton solves, each
with three candidate workers. Admission queue time is recorded before compiler
launch and is not included in solver total. Earlier unfinished, more heavily
concurrent attempts are retained under `before_bulk_union/` and excluded from
completed solver-latency and inference-performance tables. There is no claim
that these source-level savings already establish an anchored speedup.

Evidence: `SOLVER_V2/profiles/repeated_relation_parse.stack.txt`,
`union_algebra_test.log`, `native_union_test.log`,
`schedule_candidate_cache_test.log`, `search_native_cache_test.log`, and
`protocol.json`.

## F-246 — Semantic coupling reuse reduces warm derivation, with cold cost exposed

✅ **Verified.** Separate fresh CPU processes import the Llama and Qwen anchored
exports and compare uncached, cold-cache, and warm-cache derivation at fixed
granularity (split 1 and 2). Every timed module prints byte-identically to the
independent `ImportPlan` result. Expression memoization is disabled for this
comparison, isolating the semantic coupling cache. Observed derive times (ms):

| Model | Split | Uncached | Cold | Warm | Cold hits / misses | Warm hits / misses |
|---|---:|---:|---:|---:|---:|---:|
| Llama | 1 | 1245.38 | 233.508 | 5.77378 | 309 / 45 | 354 / 0 |
| Llama | 2 | 1148.18 | 219.150 | 5.28773 | 411 / 56 | 467 / 0 |
| Qwen3 | 1 | 2058.12 | 241.344 | 11.4686 | 627 / 47 | 674 / 0 |
| Qwen3 | 2 | 2009.18 | 218.520 | 10.5149 | 813 / 58 | 871 / 0 |

These are single diagnostic observations under concurrent CPU work, not
median full-solver latency or GPU speedups. Cold cache also constructs the
edge Oracle in `CouplingCache::Derive`; uncached import only derives the CG.
On the small reference fixture this cold cost exceeds uncached derivation
(194.285 vs 137.831 ms at split 1; 262.411 vs 136.316 ms at split 2).
The next cold-path accounting step is to separate canonical-key construction,
coupling derivation, and eager `OracleFor` proof preparation at that call site;
the current timer includes all three. Real-model full-search timing remains
pending and must retain that work in its total.

Evidence: `SOLVER_V2/cache_derive_{reference,llama,qwen3}.log`, the two anchored
`*.command.json` files, and `test/unit/coupling_cache_test.cpp --benchmark`.

## F-247 — Oracle acceptance follows the measured winner and its actual plan

✅ **Verified.** The gqa2 seq=4 reference's fastest internally equal raw-log
candidate is shortlist rank 2. `audit_winners.py` independently selects it from
the ten-process logs and audits its CG, preserving the original rank-1 audit.
The tool reads grid/residency/κ from that module (512/4/1), and all 330 sampled
physical-fiber comparisons are exactly equal as sets. Its 56 separately
reported `CG_EDGE` rows match the number of semantic coupling operations.

The distinction matters because the original runner audited the simulator's
rank-1 CG and the tool had used grid=SM count, residency=1, κ=1. Those checks
do not identify the final measured plan. The completion runner now performs
winner-aware audits for every real arm, and G-5 checks the selected CG name,
its SHA256, its plan attributes, and the raw set comparisons. Semantic CG
edge categories are reported separately from the physical graph extended with
executor ordering. The complete anchored G-5 gate remains pending.

Evidence: `SOLVER_V2/reference/gqa2_s4/winner_oracle_audit.{log,command.json}`,
its three original per-process measurement directories, `winner_audit_test.log`,
`tools/tilemega-skeleton-audit.cpp`, and `SOLVER_V2/verify.py`.


## F-248 — Qwen seq16 control recovers from pre-output allocation failures

✅ **Verified.** The original ten-process attempt contained four CUDA
`out of memory` exits at `ModelHarness.cuh:2585` before any `E2E_HASH` or
`E2E_TIME`, and six internally equal completed runs. This was not a valid
10/10 control measurement and is excluded as an entire attempt. Its complete
raw logs remain under `SOLVER_V2/legacy_r8_domain/qwen3_s16/failed_attempts/`.

After preserving that attempt, the unchanged generated source was rebuilt
with explicit `TILEMEGA_MIDPOINT_REFINE=0`. All ten new processes have equal
L0.5/L1/L2 hashes and zero internal mismatch counts. Median L0.5/L1/L2 times
are 8.8882315 / 9.112672 / 9.394016 ms. Their exit code 1 reflects the CPU
golden check, not R9's internal-equality gate. The four corresponding skeleton
arms resumed admission after the successful control; their searches and the
full anchored performance gates remain pending.

The runner now records available memory and utilization before each process.
It admits after three consecutive observations with utilization <=5% and
free memory >= fixture bytes + 2048 MiB (5931 MiB for this fixture). The margin
is a stated admission estimate, not a measured CUDA peak. Admission cannot
exclude all external races, and any subsequent failures remain in raw logs.
Only pre-output OOMs may be manually recovered as a whole attempt; numeric
failures and unexplained exits are rejected by the recovery guard.

Evidence: the cell's `resource_recovery.json`, original archived logs, new
`selected.cu.measurement/process_*.{log,command.json}`, `gpu_admission.jsonl`,
`measure.command.json`, and `SOLVER_V2/gpu_admission_test.log`.


## F-249 — A larger isolated candidate batch preserves search decisions

✅ **Verified.** The expanded CPU fixture in
`skeleton_search_isolation_test.cpp` compares serial execution with
`--search-jobs=36`: 116 candidate records, all five selected candidate keys,
and all five final worker/slot/start/end tables are byte-identical. Both
paths import once and record coupling-cache hits. This fixture uses stub
resource limits and establishes process-isolation/selection equivalence,
not GPU residency or anchored performance.

The still queued Llama seq1/k8 arm now uses an additional admission slot and
36 candidate workers. Its compiler bytes, full candidate domain, P=3 and
selection procedure are unchanged; the four active seq4 arms retain their
original three-worker runs. The queue-promotion script freezes and checks the
queue-only parent before cancellation and refuses already admitted solvers.
Original launch metadata is retained. Solve-time comparisons must disclose
these different CPU budgets; anchored throughput remains to be measured.

Evidence: `SOLVER_V2/search_isolation_36_test.log`,
`SOLVER_V2/promote_guard_test.log`, and
`SOLVER_V2/matrix/llama_s1/skeleton-k8/{launch.command.json,queued_launch_history/}`.


## F-250 — Two busy-start control samples receive complete fresh timings

✅ **Verified.** The original Llama seq64 and Qwen seq4 device-before logs
recorded GPU utilization 93% and 100%, with 5157 and 43962 MiB of memory in use.
Their ten processes were internally equal, but a single initial snapshot
cannot establish the duration or cause of any interference. Both complete
original attempts are preserved under their `excluded_attempts/` directories.

The unchanged sources were rebuilt with MIDPOINT_REFINE=0 and remeasured as
whole ten-process sets with memory/idle admission before each process. Both
new sets pass 10/10 internal equality and have zero allocation failures.
New median L0.5/L1/L2 times are **11.785504 / 11.6687035 / 12.5220235 ms**
(Llama seq64) and **9.111864 / 9.5751525 / 8.949024 ms** (Qwen seq4).
The old L2 medians were 15.506999 and 9.320144 ms respectively. This is a
measurement-condition recheck, not a solver or kernel speedup; the difference
is not assigned entirely to contention from the initial snapshots alone.
No slow round was selectively dropped, no numeric failure was retried, and
neither configuration nor an already running search was changed.

Evidence: each cell's `resource_recovery.json`, archived `device_before.log`
and ten-process logs, fresh `selected.cu.measurement/` logs and command
metadata, and `gpu_admission.jsonl`. These fresh control samples replace the
excluded sets in G-8; the full anchored skeleton matrix is still pending.
