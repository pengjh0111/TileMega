# Findings

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
  device code.

## F-35 — A per-edge event graph cannot beat a barrier under a sequential stage loop

- Finding: after F-32, F-33 and skipping waits for CTAs that own nothing in a
  stage, L2 sits at 1.0136× (2-layer GQA, 50 fresh processes) and 1.0155×
  (4-layer MHA, 25 fresh processes) of L1 — a large improvement from 1.182×
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
