# AS-1 port — E-0 checkpoint, 2026-10-20

Evidence: **verified** means read/executed here; **stated** means a source
assertion; **inferred** means not independently established. This is a partial
delivery: E-0 passes; E-1 and the fixed E-2 BERT encoder block are not implemented.

## Source, environment and boundaries

- Initial `origin/tilemega`: `297ac803847e902f954a0852751c2667c549f597`.
  Work is exclusively on `tilemega-ascend`. Explicit pushes to that branch
  currently fail with missing HTTPS credentials; local commits are retained.
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
F11 admission currently enables only GEMM, and rejects every unsupported body.

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
- E-1 remains unimplemented: AIV RMSNorm/SiLU/Add bodies, importer stage binding,
  general multi-stage codegen, same/cross-group numerical tests and C-3.
  Current codegen rejects these graphs before NPU launch.
- E-2 remains fixed to BERT encoder block B1/S16/H256/heads4/FFN1024; rank-three
  entry, LayerNorm/GELU/attention/mask bodies and corresponding importer/Plan
  lowering are still required. No alternate block was substituted.
- `G-INV-AS`, native `G-BUILD-AS`, E-0 `G-PERSIST-AS` verified;
  `G-TRACK-AS`, `G-HANDOFF-AS`, full `G-E2E-AS` remain partial/pending.
  No torch_npu workload timing or speedup claim yet; optional timing follows
  correctness. E-0 alone is not AS-1 completion.
