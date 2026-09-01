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

## F-12 — CuTe IR can retain dynamic algebra, but execution is unverified

- Finding: checked-in tests explicitly preserve dynamic composition and
  zipped-divide operations when static folding cannot prove them. Dynamic
  right-inverse behavior was not found. The pinned LLVM checkout and system
  MLIR tools were absent, so no test suite was executed.
- Evidence: `experiments/V_F/raw/source_audit.txt` and `environment.txt`.
- Skeleton impact: use pinned CuTe MLIR dialect only as a Phase 3 analysis
  representation, with Presburger/ISL as symbolic-legality authority; do not
  make it a codegen dependency until the dynamic suite runs.
- Confidence: medium for source-level representability, low for runtime/test
  compatibility in the current environment.

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
