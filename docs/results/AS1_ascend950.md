# AS-1 port — E-0/E-1/E-2 checkpoint, 2026-10-20

Evidence: **verified** means read/executed here; **stated** means a source
assertion; **inferred** means not independently established. E-0/E-1 and the
fixed bias-free E-2 BERT encoder block pass their scoped end-to-end checks.
E-2 also verifies L1/L2 equivalence. Remaining capabilities are listed below.

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
  each reports 28 AIC / 56 AIV. E-0/E-1 use physical 0; final E-2 uses physical 1.
  Each child sees logical 0 only. ACL verifies both physical mappings.
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
F11 admission enables the fixed GEMM, RMSNorm, SiLU, Add, LayerNorm, Erf GELU
and masked encoder attention bodies. Other tasks/math fail before launch.

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
  Physical 1 repeats the same four-direction test in 5/5 fresh processes.
  Workload edges have separate E-1/E-2 receipts, not just this litmus.
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

## Reproduction and shared-server record

Source `scripts/as1_env.sh`; run `as1_correctness prepare-e2 --out <new>`;
`as1_compile_model <prepared> <new-plan> 2 4 stage_barrier` (omit the final
argument for task events). Configure `<new-plan>` with CMake in private build,
build only `tilemega_as1_generated --parallel 1`, execute via unlocked
`as1_run.py --physical-device 1 --timeout-s 90 --out <new-job> -- <binary>
<prepared> <new-job>/output.bin`, then `as1_correctness compare --directory
<prepared> --output <new-job>/output.bin --pipeline-directory <new-plan>`.
Use the same commands with prepare-e0/prepare-e1 for those fixed graphs.

- All project NPU jobs are sequential, single-card, at most two groups. No device
  locks, exclusive mode, HCCL, reset or action on non-project processes.
  External jobs were observed on both cards; timing is contended.
- Two timeouts are preserved: earlier physical-0 initialization before launch;
  experimental scalar-PV attention on physical 0 timed out after 90 s inside
  `aclrtSynchronizeStream`. Wrapper ended only its tracked process group.
  Device health stayed OK; immediate device termination is not claimed.
  Further E-2 work moved to physical 1 and replaced that experimental body.
- CPU target test initially rejected duplicate TaskKind-4 implementations;
  binding now selects by explicit impl_id and tests distinguish SiLU/Erf GELU.
  The abort produced no repository core file; subsequent own shells disable cores.
- Sampled generated peak is about 1 GiB; private work including protected
  dependencies about 3 GiB, on a shared partition with about 13.4 TB free.
  Manifest dry-run/apply retains all selected and failure evidence; zero eligible
  production files deleted. Exact samples: `runs/as1/disk_samples.jsonl`.
  Raw tensors, binaries, traces, generated sources and caches stay outside Git.

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
- Latest Ascend host contracts pass; unsupported standalone ReLU is rejected
  during import. E-0 tail generated source remains byte-identical to the
  previously verified binary, avoiding an unnecessary rebuild/device test.

## E-2 real BERT encoder block (verified)

- Fixed B1/S16/H256, four heads of 64, FFN1024, no linear biases, affine
  LayerNorm epsilon=1e-12, Erf GELU, dropout=0. Explicit int64 key-padding mask
  invalidates the final three keys. This is a locally constructed **bias-free
  encoder block variant**, not a claim to support a pretrained/HF BERT model.
- `DnnModelPlan.cpp` imports actual rank-three BF16 input, QKV packing, attention
  and mask structure. No embedding/weights download. LayerNorm owns one row;
  attention owns one query/head. Independent ISL operand/write contracts match
  all four GEMM and six AIV stages. CTG math is preserved.
- `AscendEncoderCodegen.cpp` / `PersistentEncoder.cpp.in` consume the solved
  Plan: 10 stages, 156 tasks, 1224 exact GM dependencies, G2/6 workers. The
  event executor allocates 1,922,340 bytes. Metadata/Stage ABI, every task owner,
  coordinate, event, buffer guard and one-launch receipt are checked on device.
- `EncoderVectorBody.h`: 16 KiB UB; FP32 LayerNorm/Erf GELU/Add, BF16 I/O.
  Six regular/tail vector probes pass. Attention uses SIMD QK/softmax and
  register SIMD PV, FP32 opmath and BF16 output. S1/S13/S16 and all-masked cases
  pass; tail max BF16 abs error 3.8147e-6. Earlier vector/scalar PV failures
  remain archived; their exact cause is not generalized to unsupported API claims.
- C-0/C-1/C-3 **pass**: Task Event executor 5/5 fresh physical-1 processes,
  four epochs/624 task executions each, one launch per process. Final output is
  bitwise equal to independent CPU BF16 MATH-SDPA reference, zero failed elements.
  Ten stage references pass; stage0 max abs 1.19209e-7, stage7 7.62939e-6,
  other stages exactly equal. Output SHA256:
  `5b43ac760d9c09150ba57a4615d1f010f8fbf7458ed93bf0492ac0b74a18539e`.
- Full FP32 graph is diagnostic: max abs .0274086, max rel 2.24528, RMS .00439784,
  165/4096 exceed the predeclared BF16 comparison tolerance .00390625/.0078125.
  This failure is retained; no claim of fixed-tolerance FP32 equivalence.
- C-2 **pass for this encoder**: real L1 stage barriers versus L2 task events,
  each 5/5 fresh processes, same four epochs and bitwise final output. L1 has
  per-worker/per-stage GM acknowledgements and one AIC0 release; empty lanes
  acknowledge. Static preflight rejects same/later-stage dependencies and FIFO
  reorder. Extra barrier control is 35,840 bytes. Task events still publish once.
  E-0/E-1 retain only their original executor; no blanket C-2 claim.
- Evidence: `runs/as1/e2/e2_summary.json`, `bert_b1_s16_refs/native_*`,
  `jobs/e2_bert_*`, `probes/encoder_{vectors,attention_reg}_summary.json`,
  `probes/gm_visibility_card1_summary.json`. E-0/E-1 host re-generation remains
  byte-identical to selected kernels; no CUDA regression or redundant NPU run.

## Remaining scope and gates

- Fixed-workload G-BUILD/G-PERSIST/G-HANDOFF H-0 and E-0/E-1/E-2 C-0/C-1/C-3
  are verified; E-2 C-2 is verified. Target costs remain uncalibrated and fixed;
  no auto-optimality claim. Unmeasured lanes prohibit performance ranking.
- Native composite **single Task** GEMM+activation owner/participant protocol is
  not implemented: current MIX kernel coordinates separate AIC/AIV Tasks.
  WorkerMap composite CPU checks and official EVG probes do not prove that body.
- BF16 direct UB, UB→L1, mode4, SSBuffer, SIMT, split-K combine and other model
  shapes/tasks remain disabled. P-4 full allocation limits and P-10 cycle/ns
  calibration are unverified. These prevent claiming every AS-1 work item done.
- HTTPS push still lacks repository credentials; local checkpoint commits remain
  intact. No force push or credential/global-configuration modification.
- torch_npu end-to-end timing is the next limited experiment; no speedup claim.
