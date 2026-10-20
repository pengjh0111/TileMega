# AS-1 port — E-0/E-1 checkpoint, 2026-10-20

Evidence: **verified** means read/executed here; **stated** means a source
assertion; **inferred** means not independently established. This is a partial
delivery: E-0/E-1 pass their scoped C-0/C-1/C-3 checks; the fixed E-2 BERT
encoder block remains unimplemented. C-2 has only one verified executor.

## Source, environment and boundaries

- Initial `origin/tilemega`: `297ac803847e902f954a0852751c2667c549f597`.
  Work is exclusively on `tilemega-ascend`. Explicit pushes to that branch
  are unconfirmed: initial attempts lacked HTTPS credentials and the E-0
  checkpoint attempt timed out after 300 s with no HTTP response. Local commits
  are retained; no protected branch or force push is used.
- Exact original user-message SHA256:
  `132596ba7075142fc33a36318b66948a09d4707d6be32bd063054126b39e44f6`;
  private `tilemega-as1/Prompt/TileMega_AS1_prompt.md`.
- **verified** physical 0/PCI 55:00.0 and 1/PCI D7:00.0 are 950PR;
  each reports 28 AIC / 56 AIV. All compute probes and workloads use physical 0,
  visible logical 0. ACL inventory separately checks both physical mappings.
- **verified** host CANN 9.1.0, driver 25.7.0, BiSheng clang 15.0.5,
  GCC 9.4.0; CATLASS `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`.
- Private venv reads compatible installed torch/torch_npu packages without
  modifying them. CPU torch 2.7.1 export guards were adapted and checked for
  shared-symbol equality, ranges and rejection of derived input dimensions.
  No official weights, tokenizer, datasets or framework packages were downloaded.
- All owned sources, environments, caches, builds and logs remain under
  `/home/d50064638/junhuip`. No shared shell/Git/pip configuration was changed.

## Native build and dependency provenance

- `cmake/AscendAS1.cmake` selects C/C++ semantic core before CUDA discovery.
  No CUDA language/toolkit, CUTLASS device target or NVIDIA test is used.
  Original CUDA runtime/SharedStorage sources are excluded. Their emitter rejects
  calls; pure model/dependency metadata readers in the same file remain reused.
- **verified** pinned LLVM/MLIR 23 commit
  `23a60f15f2fcafcf67b95b0a035053579958b732`, sparse checkout inside private work.
  Built only requested `mlir-tblgen`, `MLIRIR`, `MLIRPass` and their dependencies,
  one compiler process at a time, `nice -n 10`; no LLVM/MLIR test suite.
  The stable private build-tree CMake package is used; no shared installation.
- Private GMP 6.2.1 (including gmpxx), ISL 0.28, NTL 11.5.1, PolyLib 5.22.8,
  Barvinok 0.41.9+ds. `configs/ascend/as1_sources.json` pins URLs/hashes.
  **Difference from repository recipe**: repo.or.cz TLS failed, so the Debian
  upstream Barvinok repack replaces the requested Git SHA; zsolve is disabled.
  PolyLib 6.01 failed on its removed `Smith` declaration; official 5.22.8 works.
  The failed builds and errors are preserved; no numerical library was stubbed.
- **verified** `as1_isl_crosslink` constructs MLIR/Presburger objects and computes
  actual ISL/Barvinok cardinality `[n] -> { n : n > 0 }`; both paths pass.
- `scripts/as1_build_host.sh <component>` reproduces one selected component,
  always parallel=1. It has no `all` mode, candidate sweep or system install.
  `cmake/AscendKernel.cmake` uses native ASC and `--npu-arch=dav-3510`.

## Implemented path and coupling changes

1. `bench/as1_correctness.py` creates seeded BF16 matrices, CPU FP32 and BF16
   references, and a real `torch.export` bridge. Seed is 20261016.
2. `DnnModelPlan.cpp` accepts dense BF16 matrix inputs; existing
   `TorchExportImporter::ImportPlan` builds CG and exact task semantics.
3. `PrepareSymbolicProblem` retains ModelDescription, CTG/ISL projection and
   dependency oracles. Its explicit element span is 128 logical elements,
   independent of NPU threads. Native candidates bypass GPU pricing.
4. `BuildPlanSkeleton` builds the real stage/dependency schedule in fixed,
   unpriced mode; `BindAscendWorkers` restricts owner domains by role.
   Cross-role identity colocation is rejected; group affinity is separate.
5. `MaterializeAscendPlan` enumerates those task domains/oracles, constructs
   queues/events and validates dependency + FIFO acyclicity. Each event has
   exactly one publisher/trigger; aggregate multi-owner counters are rejected.
6. `ImplementationContract::VerifyExactAccess` compares independent logical
   A/B/C tile access sets with exact CTG task-element projection, including tails.
   Logical weights stay NK; physical KN packing/K padding belong to the body.
7. `Codegen/Ascend/AscendCodegen.cpp` inserts the materialized tables into
   `PersistentGemm.cpp.in`. CANN compiles the generated source; ACL launches it.
   TaskRef is 28-byte POD, Metadata 32-byte POD; an actual NPU roundtrip passes.
8. The native executor traverses each worker queue inside one MIX launch.
   `Backend/Ascend/GemmTile.h` calls CATLASS BlockMmad for the supplied tile
   coordinate, never BasicMatmul/DeviceGemm/swizzle. Ordinary K loops stay AIC.

F1/F2/F3/F5/F6/F7/F12/F13/F16 have native paths above; F4 preserves task/wait
meaning in checked POD; F9/F10 CUDA construction is excluded; F14 has an
independent unlocked runner; F15 reuses the importer. F8 native cost lanes and
one shared GM-rate adapter have CPU contracts, but cost fitting/automatic ranking
are **not implemented**. The legacy CUDA ResourceVector is not used to price NPU.
F11 admission enables GEMM, RMSNorm, SiLU and residual Add for the fixed MLP;
all other TaskKinds and unsupported activation mathematics fail before launch.

## Target and synchronization scope

- `configs/targets/ascend_3510.json`: `arch_tag=ascend_3510`, SKU 950PR,
  hardware 28/56 tracks, conservative selected G≤2, residency=1.
  MIX launches G blocks and maps workers AIC(g)=g, AIV(g,lane)=G+2g+lane.
  `msobjdump` verifies `MIX_AIC_MAIN`, ratio 1:2 for the generated executable.
- `Track.h/.cpp`: separate tracks, workers, groups and launch blocks; per-role
  budgets, role/tuple checks, unique ownership, queue bounds and wait-graph audit.
  T-1–T-6/T-16 CPU fixtures pass; native target/whitelist/error checks pass too.
- Actual queried budgets: L1 524288, L0A/B 65536 each, L0C 262144, available
  AIV UB 221184 bytes. CATLASS reserves entire AIC banks, so the native GEMM
  contract accounts conservatively for full banks. Compiler register counts
  are unreported; no fabricated register or occupancy numbers are used.
- `DevicePrimitives.h`: explicit DMA completion (FIX→S/MTE3→S), DSB,
  DCache-bypassed single-owner event controls, consumer cache invalidation,
  bounded 100000-poll waits and shared abort. This is not CUDA acquire/release.
- **verified P-6**: G=2, four role directions, pre-read old cache, reused 128 B /
  4 KiB FP32 payload, 16 epochs, 5/5 fresh processes, zero violations/timeouts.
  E-0 uses task events plus a full-epoch barrier; this does not prove E-1 edges.
- Direct BF16 L0C→UB, UB→L1, SSBuffer control, mode 4 and SIMT remain disabled.
  Current body/plan routes use GM. FP32 EVG direct route is verified in 5/5 fresh processes;
  see `AS1_probe.md`. This does not enable the BF16 TileMega route.
- No calibrated cost fields or automatic performance ranking; native `TopK`,
  GPU occupancy, CUDA target and unsupported TaskKind calls fail explicitly.

## E-0 results (verified)

Tolerance: abs 0.001 + relative 0.0078125 × |reference|, one BF16 relative
spacing with near-zero allowance. No tolerance was increased after measurement.
No missing/nonfinite elements; all output guards remain intact.

| Logical M×N×K | Groups / task owners | FP32 max abs / rel / RMS | BF16 max abs / rel / RMS | Failed elements |
|---|---|---|---|---:|
| 64×128×256 | G1, worker 0 executes four distinct tiles | 0.003905535 / 0.003847520 / 0.0008230355 | 0 / 0 / 0 | 0 / 8192 |
| 33×96×129 | G2, workers 0/1 each execute two tiles | 0.003605962 / 0.003860059 / 0.0005840700 | 0.00048828125 / 0.006578947 / 0.00000867516 | 0 / 3168 |

- C-0 **pass**: real Importer→CTG→Solver→Plan→Codegen→CANN→ACL→output;
  host launch counter=1, per-task owner/coordinate/event trace matches the Plan,
  and metadata ABI roundtrip passes. No host launch per tile.
- C-1 **pass**: both independent CPU FP32 and BF16 graphs above.
  K=129 is zero-padded physically to 144; the logical relation still reads 129.
- C-2 **not met**: only this static-queue executor has device validation.
- C-3 **pass in the stated E-0 scope**: five fresh G2 processes, 16 epochs each,
  64 task executions/process, event epoch=16; all final outputs are bitwise
  identical to the separately verified tail output. No cross-task payload edge
  exists in E-0; E-1a/E-1b still need their own five-process results.

Evidence under `runs/as1/`:
`e0/{aligned/native_g1,tail/native_g2_exact_include,tail/native_g2_e16}/`
contains CG/solved IR, exact task/dependency relations, Plan JSON, generated source
and generation timings. `jobs/e0_{aligned_ctg_solver_g1,tail_ctg_solver_g2}/`
contains numerical JSON, output trace and binary/source SHA256.
`e0/persistent_sync_summary.json` indexes the five independent processes.

## Reproduction (one workload/target at a time)

```bash
source scripts/as1_env.sh
# Use new output directories; existing evidence is never overwritten.
PYTHONPATH="$REPO/python" python3 -m tilemega.bench.as1_correctness \
  prepare-e0 --shape-index 1 --out "$REPO/runs/as1/e0/reproduce_tail"
"$AS1_WORK/build/tilemega/as1_compile_model" \
  "$REPO/runs/as1/e0/reproduce_tail" "$REPO/runs/as1/e0/reproduce_plan" 2
cmake -S "$REPO/runs/as1/e0/reproduce_plan" -B "$AS1_WORK/build/reproduce_tail" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$AS1_WORK/build/reproduce_tail" --target tilemega_as1_generated --parallel 1
python3 python/tilemega/bench/as1_run.py --physical-device 0 \
  --out "$REPO/runs/as1/jobs/reproduce_tail" --timeout-s 90 -- \
  "$AS1_WORK/build/reproduce_tail/tilemega_as1_generated" \
  "$REPO/runs/as1/e0/reproduce_tail" output.bin
PYTHONPATH="$REPO/python" python3 -m tilemega.bench.as1_correctness compare \
  --directory "$REPO/runs/as1/e0/reproduce_tail" \
  --output "$REPO/runs/as1/jobs/reproduce_tail/output.bin" \
  --pipeline-directory "$REPO/runs/as1/e0/reproduce_plan"
```

## Shared-server record and remaining work

- All jobs sequential, physical 0 only, at most two groups, no locks/exclusive
  mode. External load was observed. One earlier probe timed out during
  `aclrtSetDevice/TsdOpen` before launch; retry passed. No kernel timeout/OOM,
  reset, shared configuration change or operation on another user's PID.
- Latest sampled generated-output peak is below 1 GiB; protected dependency
  sources bring private work to about 3 GiB. Manifest cleanup currently retains
  selected binaries, all failure/probe evidence and recent builds; zero eligible
  production artifacts were deleted. `disk_samples.jsonl` records actual samples,
  including shared filesystem free space; raw tensors/binaries are not in Git.
- E-1 scoped results appear below. The native executor uses GM task events;
  a second L1 executor and composite fused-task owner/ack implementation are not
  enabled. Official EVG FP32 direct probes do not enable native BF16 UB handoff.
- E-2 remains fixed to BERT encoder block B1/S16/H256/heads4/FFN1024; rank-three
  entry, LayerNorm/GELU/attention/mask bodies and corresponding importer/Plan
  lowering are still required. No alternate block was substituted.
- `G-INV-AS`, native `G-BUILD-AS`, E-0 `G-PERSIST-AS` verified;
  Ascend T-1–T-6/T-16 and `G-HANDOFF-AS` H-0 pass; full `G-E2E-AS` remains
  partial because E-2 and C-2 are outstanding.
  No torch_npu workload timing or speedup claim yet; optional timing follows
  correctness. E-0 alone is not AS-1 completion.

## E-1 real MLP path (verified)

- Seed 20261016, BF16, B=1/4, H=128, FFN=256, epsilon=1e-5, no bias:
  RMSNorm → GEMM → SiLU → GEMM → residual Add. Real `torch.export` enters
  `BuildDnnModelPlan(materialize_pointwise=true)` and the existing importer.
  Fixed phase batch binds event capacity without redefining symbolic semantics.
- `DnnSemanticLifting.cpp` preserves five exact semantic stages. Two GEMM and
  three AIV independently declared operand/write sets match CTG projections.
  Solver/Plan produce 10 tasks at B1 and 16 at B4; dependency+FIFO graph is acyclic.
- `AscendGraphCodegen.cpp` and `PersistentGraph.cpp.in` consume those tables.
  Native Stage ABI is 36-byte POD; every stage field, Metadata, task ownership,
  epoch count and single-owner event is checked after actual NPU execution.
  One MIX(1:2) launch executes all stages, including empty-lane epoch ack.
- `AivBody.h`: FP32 vector arithmetic with BF16 load/store, UB 6656 bytes.
  RMSNorm, SiLU and Add each pass 1x128 and 4x129 body probes against independent
  FP32/BF16 references, with zero BF16 differences and intact output guards.
- Same-group placement uses G1. Forced G2 puts the up GEMM on AIC0 and SiLU on
  group1 AIV0; down GEMM on AIC1 and residual Add on group0 AIV0. The B4 Plan
  contains 18 cross-group and 24 same-group dependency edges, each with a GM
  HandoffContract, allocation size, lifetime and uncalibrated cost status.
- B4: 5/5 fresh processes for **each** placement, 16 epochs, 256 task executions
  per process; all outputs bitwise equal to CPU BF16 and to the other placement.
  B1: both placements pass with 10 tasks/one epoch. No timeout/OOM in these jobs.
- CPU BF16 errors are exactly zero. Full FP32 diagnostic errors:

| Batch | max abs | max rel | RMS | nonfinite |
|---|---:|---:|---:|---:|
| 1 | 0.00568891 | 1.27335 | 0.00174132 | 0 |
| 4 | 0.00752246 | 3.11111 | 0.00176185 | 0 |

- Full FP32 removes BF16 rounding points; cancellation explains large relative
  errors near zero. The initial B4 atol=.004/rtol=.015625 check failed 1/512;
  that receipt is retained, never relabeled. Same-graph tolerance is now the
  stricter .001/.0078125, with **zero actual error**. A separately precomputed
  conservative rounding envelope (|W| error propagation, dot gamma_K, SiLU
  Lipschitz 1.5, BF16 unit roundoff) is diagnostic; its B4 bounds are .393–.532,
  and are not a claim of fixed-tolerance FP32 equivalence.
- Raw index: `runs/as1/e1/e1_summary.json`, same/cross-group sync summaries,
  `b{1,4}_bounded_reference/native_*`, and jobs `e1_b*`. Selected B4 binaries:
  G1 `463cd6941bdb2cfb9c2f07258b321b62dced1c7d1a55bccae3b5067a20e33aba`;
  G2 `e9f2582facc92cb8e0b8d5b763ca304482a1af50745d8f68bc1d818bfd142276`.
- Reproduce one case: `as1_correctness prepare-e1 --batch 4 --out <new>`;
  `as1_compile_model <prepared> <new-plan> 2 16 cross_group`; configure/build
  only `tilemega_as1_generated`, then the unlocked wrapper and `compare`.
  Existing argument forms and private environment are the same as E-0 above.
- Latest Ascend host contracts pass; unsupported standalone ReLU is rejected
  during import. E-0 tail generated source remains byte-identical to the
  previously verified binary, avoiding an unnecessary rebuild/device test.
