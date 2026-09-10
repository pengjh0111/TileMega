# Interval Selection and Shared Runtime Progress

This is a progress checkpoint, not B1 completion or end-to-end GPU acceptance.

Latest evidence is [runtime_result.md](runtime_result.md): production RoPE/KV
400/400 and GEMM calibration chains400/400 pass. RMS prediction/sign fails;
that local gate supersedes the earlier pending-experiment statements below.

## Verified CPU Work

`lib/Solver/FusedRuntimeProjection.cpp:14` composes the replacement-to-phase
map on both sides of runtime C. It removes the selected internal edge before
composition, rebuilds event groups, and counts the deduplicated worker/event
image. It does not subtract a second fixed fused-edge rebate. Synthetic
coverage: 24 symbolic/concrete cases and 16 rejection branches, zero ISL
references. Complete producer coverage is required; prefix-only KV consumers
retain their work without invoking a nonexistent producer phase.

`lib/Solver/FusionIntervalDP.cpp:16` implements adjacent interval transitions
inside a pinned-residency outer loop. Fusion choices survive until terminal
evaluation because event deduplication is not an additive per-edge quantity.
The current domain fixes GEMM tile/split and attention choices; it is not yet
the full joint shape/split/chunk/fusion solver. All terminal patterns are kept,
so this implementation can grow exponentially with fusion candidates.
Resources use phase max registers and max scratch plus intermediate storage.
The production probe uses the archived whole-kernel register maximum as an
explicit input bound, not a measurement of a new fused kernel.

`intervals_first/` records two BF16 models at seq 4/128, past 3. gqa2 has
4 alternatives per shape; mha4 has 16. All four empty-fusion baselines retain
the original Evaluate double bits. Selected rows:

| Model / seq | Selected pairs | Task refs | Wait entries | event_ns | total_ns |
|---|---:|---:|---:|---:|---:|
| gqa2 / 4 | 2 | 224 | 516 | 81147.146742347162 | 371357.60640959791 |
| gqa2 / 128 | 2 | 5184 | 12756 | 108324.78802778097 | 413239.97158688481 |
| mha4 / 4 | 4 | 528 | 1108 | 162668.93669298591 | 743089.85602748732 |
| mha4 / 128 | 4 | 12432 | 26020 | 225553.12125813717 | 838012.52489494742 |

These selected pairs are RoPE to KV append, not GEMM fusion acceptance cases.
For one gqa2/seq4 pair, kappa1 event_ns changes from
86894.754072067124 to 84020.950407207143, delta -2873.8036648599809 ns.
This verifies a nonzero projected fusion price; its GPU sign gate is open.
The runtime producer fanout is 2 even where logical element fanout is 1:
the CTA ownership projection must happen before charging recomputation.

`lib/Dialect/CouplingGraph/FusionPass.cpp` now writes a selected disjoint
batch transactionally. Two-model testing replaces 6 tasks and checks 122
edge identities, with 4 rejected requests leaving the original untouched.
The existing single-pair suite retains 1890 edge checks and 16 rejected cases.

## Verified GPU Bodies

`GemmStageTaskBody.h:439` adds an independently switched shared destination.
It reuses CUTLASS's mainloop and thread epilogue operation, preserving BF16
rounding, but stores by CuTe accumulator coordinates into shared memory.
CUTLASS's existing global-store epilogue cannot accept a shared pointer.
`FusedGemmTaskBody.h` supplies actual GEMM/add and GEMM/RMSNorm bodies;
the latter runs one consumer row per task and recomputes the full producer
tile for each row. The shared budget is max scratch plus the intermediate.

`shared_body_first/`: BF16 50/50 fresh processes and FP32 50/50, rotated by
dtype each round. Each process runs 24 cases, including M tails, N tails,
multiple producer M tiles and beta 0/1. Both fused outputs are bit-identical
to separate GPU bodies. The global intermediate remains at its sentinel.
This is 1200 body cases per dtype, not 50 end-to-end model processes.

| Dtype | Threads | Shared bytes | Add registers | Norm registers | Add/norm CTA per SM |
|---|---:|---:|---:|---:|---:|
| BF16 | 128 | 23552 | 94 | 92 | 4 / 4 |
| FP32 | 256 | 47488 | 95 | 96 | 2 / 2 |

The ptxas logs are retained. No spill stores or loads in these body kernels.
These register counts must not replace megakernel register measurements.
No performance comparison or win/loss conclusion is made here.

`ModelHarness.cuh` dispatches the two new stage kinds behind
`TILEMEGA_FUSION_RUNTIME=1` (default 0), includes their storage in TaskSmem,
and checks unsplit accumulation and full-row norm tiles on the host.
The enabled production harness compiles. Production lowering now consumes
the verified replacement CG, phase ownership, and exact task dependency table.
`Lower` verifies CG before entering the fused branch. Missing provenance or
replacement dependencies are rejected rather than replaced with kAll.

## Production Lowering Follow-up

The selected RoPE/KV graphs now lower to actual 28/56-stage models. The host
binds exact symbolic task/dependency descriptors, checks their task domains,
and builds per-task waits with existing deduplication and lifting. Fine-event
waits use exact predecessors. Resident-only I3 remains enforced.
`ProjectWrittenFusionQueues` independently verifies the replacement CG edges
are covered after logical-to-runtime projection; writeback event prices match
the selected DP price bits. gqa2 single-process probes at seq4/128, past3,
and seq4/past512 pass all levels. These probes alone are not a race claim.
The 400-process state-rotated matrix is recorded in `selected_runtime_second/`.

Two-stage GEMM/add and GEMM/RMS chains use dimensions taken from the exported
projection and the same importer, FusionPass, lowering and ModelHarness.
They are calibration subgraphs, not complete decoder measurements. Nonconstant
fixed-seed BF16 inputs and PyTorch references are generated independently.
`chain_runtime_first/` is the corresponding execution record.

Physical tail bytes initially understated allocated shared memory: seq4
GEMM/add tile32x128 predicted 16384 B, while the body allocates 23552 B.
The resource API now accepts the backend's full intermediate allocation,
while physical R/W and traffic retain their predicated domains. Full-pattern
ptxas register evidence can override phase maxima in the interval DP; missing
evidence is rejected when that tier-3 mode is requested. Separate/fused add
L2 register counts are 112/148, so phase maxima are not a measured fused bound.

The first production runner stopped before its GPU matrix because new Add
dispatch indexed the void-pointer invocation table without a typed cast.
The compiler error is retained in `selected_runtime_first/`; the second run
reuses only verified selected MLIR and rebuilds every binary. No numerical
failure was hidden by this restart. Linking exact host dependencies also
requires the existing ISL/LLVM support libraries; cudart alone is insufficient.

## Corrections and Open Work

- Runtime access composition initially left CG seq/past parameters unbound,
  rejecting a valid concrete seq4/past3 pair. Concrete runtime accesses now
  bind ModelDescription aliases before composition; the logical path stays
  symbolic. No parameter-domain relaxation was used.
- Fiber sums initially aligned only relation parameters, not polynomial
  parameters. Both spaces now align; 63 unit checks and 2 rejection cases
  pass with zero ISL references. Constant arithmetic numerators retain their
  original domain semantics; coordinate-dependent quantities use SumAlong.
- The new CMake GPU test initially inherited sm52 and was rejected by the
  capability check. It now explicitly targets sm89, like the existing BF16
  compile contract. No capability check was weakened.
- Full portable build, CTest 43/43, policy, and the CTest five-target audit
  passed at this checkpoint. These do not imply GPU lowering acceptance.
- Open internal work: joint implementation-domain interval search,
  completed two-edge BF16 calibration and measured register/resource evidence,
  paired steady-state timing, model-vs-measurement sign gates, sm120 manifest.
  These are implementation gaps, not external blockers. B2/B3 local negative
  gates do not stop this work.
