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
