# MX-7: SM90 / MetaX TP implementation audit

Evidence: **verified** = source or local execution checked; **stated** = remote report;
**inferred** = interpretation. The original 30 external baseline cells stay frozen;
no baseline measurement is part of this audit or optimization.

**verified** Same-geometry TP4 B16 E2E P-PAIR:12.526515→3.081392ms,
**4.065×** over the original and **1.092×** over the frozen SGLang baseline.
Both arms use the same SDK host path and group; generated CU bytes are identical.
Only the included shared-AR epilogue changes. This is development control data;
new final pins must still come from the updated measured-cost/top-4 selection.

The original eight-column ownership followed MX-7§6.4. The migration mismatch
is its packet-to-lane granularity relative to current SM90, rather than a missing
shard, collective, or launch. The user's subsequent SM90 comparison/optimization
request motivates the local epilogue change; wire protocol and arithmetic remain fixed.

## Sources

- **verified** MetaX starting HEAD `a7c4a7968a787a5bf5a36fbbfc6532ae1de7a176`.
- **verified** fetched `origin/tilemega` at `213a3884a4e14629ca99f63755d501c65a7d8f53`.
  Its `docs/results/SM90R5_tp_final.csv` and `SM90R5_h800.md` contain the complete
  H800 TP results; the older `origin/sm90r5` tip lacks these final result files.
- **verified** both refs have identical `Sm90DenseEpilogue.h`, SHA256
  `90f9f68f908aa6d21b077b1f2822c1e43b128635e7f82d861d052d9ec738099b`.
  `f0b1a95c7` changed the original per-item AR to two passes.
- Remote source/result snapshots and hashes: `runs/mx7/sm90_comparison/REFERENCE.json`.
  No SM90 commit is merged, and no H800 measurement is rerun here.

## Pipeline correspondence

| Layer | Current SM90 | MetaX | Finding |
|---|---|---|---|
| Shards | qkv/gate_up/head by N; o/down by K | Same recipes and per-rank KV | **verified** No missing shard or extra unsharded GEMM |
| Execution | One library, one persistent grid/rank; prompt and decode ranges | Same, two asynchronous range launches/rank | **verified** No per-token host launch or host AR |
| Prompt | All steps execute AR/head exchange; suppress prompt token writes | Same | **verified** Extra prompt heads affect TTFT |
| qkv handoff | All ten selected pins use per-KV-group readiness (`group_wait=true`) | V2 grid boundary | **verified** Finer handoff is not ported by MX-7 |
| CTA roles |160threads:32 producers +128 consumers |512 compute threads; thread0 also barrier role | **verified** Platform-specific task-body difference |
| AR | BF16 partials, ascending-rank FP32 sum, BF16 sum then residual | Same rounding and rank order | **verified** No changed arithmetic contract |
| AR schedule | Send all thread packets, then receive; recompute packets; noinline helpers | Resource-approved per-item send/receive, inline polling | **verified** Implementation difference; see qualification below |
| Shared epilogue | One two-column LL packet per thread/item | Eight columns and four LL packets per thread/item | **verified** MetaX checks four packets serially |
| Peer visibility | CUDA strong system-scope accesses, ordinary device allocation | C0-selected fine allocation, volatile AS1 store, `__ldcg` load | **verified** Deliberate platform adaptation; original LL cohorts pass 5/5 |
| Head | Fence/flag exchange | LL8 candidate exchange | **verified** Explicit MX-7 design; original units/litmus pass 5/5 |
| Residual prefetch | Async producer path and consumer epilogue | V2 GEMM Early/Finish path | **verified** No separate communication kernel |
| Solver | Analytic stage terms plus measured communication | Measured rank stages plus measured communication, top-4 E2E | **verified** Both include 56 ARs and one exchange/step |
| Stream creation | `torch.cuda.Stream` | Approved SDK-owned nonblocking stream; lifetime correction below | **verified** Initialization occurs outside TPOT timing |

## What explains the current gap

**verified** For N=2048, dense legality allows WN<=2 for TN16 and WN=1 for
TN32/64/128 (`DecodeGeometryV2.h:89`). Thus every legal o/down instance uses the
shared epilogue and has `kResidualVectors=1`. Restoring two passes cannot remove
inter-item waits in these shapes. The source difference is real, but it is not
an established explanation of B16 performance.

**verified** Current measured AR increments, microseconds/stage:

| TP | B | o TN16 | o TN32 | down TN16 | down TN32 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 8.860 | 5.870 | 9.860 | 6.640 |
| 2 | 16 | 64.860 | 70.030 | 63.330 | 69.820 |
| 4 | 1 | 23.280 | 13.220 | 25.165 | 12.940 |
| 4 | 16 | 164.450 | 179.220 | 164.210 | 183.515 |

**inferred** Summing the selected communication keys over 28 layers attributes
3.771 ms of TP2 B16's 7.246 ms to AR; TP4 B8 attributes 4.232 ms of 6.210 ms.
These are measured-cost estimates, not stage trace measurements.

**stated** The H800 report records TP2/TP4 B16 TPOT 1.5844/1.5283 ms and AR
increments of 1.888/4.960 us at TN16. Its selected-plan model under-predicts and
can miss faster legal tiles. **verified** MetaX's completed candidate predictions
closely track E2E; its currently large AR constants are charged by the solver.
A missing communication term does not explain the MetaX gap.

**verified** MetaX's large-write probe reports about 9.4 GB/s to one peer and
19.1 GB/s aggregate to three peers. H800's reported bandwidth probe uses wider
stores, so these figures do not isolate physical interconnect capability.
Fine-memory throughput and serialized receive waits remain separate hypotheses.

## Controlled implementation experiments

Experimental sources remain copied under `runs/mx7/sm90_comparison/`.
The qualified pair-lane shared epilogue is committed at `8b32ec8f7`; production
peer primitives, TP1 and MoE remain unchanged. The eight-plan frozen gate passes8/8.

- `baseline`: current production body.
- `two_pass`: send all items before receive, recomputing BF16 values on receive.
- `packet_batch`: issue independent first reads for each item's existing four
  LL8 packets before testing epochs. A stale packet uses the original scalar
  await. Every read/store remains eight bytes, allocation and epoch rules are
  unchanged, and arithmetic still visits source ranks in ascending order.

**inferred** `packet_batch` may overlap independent receive latencies without
changing the LL protocol. It must pass ordered-BF16 AR tests in five fresh
processes at TP2/TP4, resources, full C-1/C-2/litmus and E2E before enabling.
Measurements and decisions will be appended after qualification. Baselines are reused.

## First measurements and compiler-level serialization

**verified** TP2 B16, same {4,5} group, three fresh processes per variant:

| AR increment (us) | baseline | two_pass | packet_batch |
|---|---:|---:|---:|
| o TN32 | 70.020 | 70.000 | 70.090 |
| down TN32 | 69.840 | 69.790 | 69.830 |

Neither source-loop change wins; both stay disabled. `packet_batch` AR units
pass 5/5 fresh processes at TP2 and TP4, nine legal geometries/process,
zero bit mismatches, stack/static=0 and actual resident=1.

**verified** Isolated compiler probes locate a previously missed difference:

| Source primitive | Generated instructions for four accesses |
|---|---|
| volatile AS1 u64 stores | Four `STG_B64`, each immediately followed by `ARRIVE 64` |
| `__ldcg` u64 reads | Four `LDG_B64`, each immediately followed by `ARRIVE 64` |
| relaxed system `__maca_atomic_store` | Four separate `STG_B64`, no per-store wait |
| relaxed system `__maca_atomic_load` | Four separate `LDG_B64` issued before waits for their results |

This also appears in the real AR comparison kernel, not just the isolated
probe. Raw MIR: `runs/mx7/sm90_comparison/primitive_codegen.mir.log` and
`packet_batch/tp4_B16/mir.log`. **inferred** Compiler-enforced serialization
explains why the two C++-level changes above cannot expose overlap.

**verified** The SDK declares the pure system atomic store in
`cuda/std/detail/libcxx/include/support/atomic/atomic_cuda_generated.h:1936`.
The earlier C0 write dispatch covered volatile store and atomic exchange,
but omitted this pure store. It belongs to the prompt's C0 system-atomic
candidate domain; it is now store method2. Exchange is a different operation.

The `system_relaxed` experiment retains fine allocation, LL8, epoch checks,
ascending-rank BF16 arithmetic and the existing head handshake. It uses relaxed
system atomic loads/stores and independent first reads. No fence is added.
**verified** Its first TP2 P1024 cohort passes 5/5 fresh processes, each2000 epochs,
104 CTAs, zero tears/timeouts; median roundtrip7.549us. Large-packet, all-to-all,
AR and whole-model qualification are pending. Production remains unchanged.

## Primitive-only result and the next task-body comparison

**verified** The relaxed-system primitive with fine allocation passes both TP2
packet sizes and B1/B16 AR units, each5/5 fresh processes. However its three-process
TP2 B16 AR increments remain70.550us(o TN32) and70.340us(down TN32), compared
with70.020/69.840us for the original. Removing the per-access compiler waits
alone is insufficient. This candidate is not promoted. Four-card checks continue.

**inferred** The remaining difference worth isolating is the shared epilogue's
packet ownership: SM90 assigns one adjacent column pair per lane; MetaX assigns
four packets per lane with a32-byte inter-lane stride for each instruction.
The `pair_lanes` prototype follows SM90's column-pair ownership, keeps the
original qualified peer primitives and unchanged rank-sum/rounding order, and
loads the matching residual pair. Geometry, LL8 layout, epochs and barriers
are unchanged. Its ordered-BF16 units and same-group stage comparisons are queued.

## Pair-lane result (qualification in progress)

**verified** With the original peer primitives, one column pair/lane reduces
TP2 B16's three-process AR increments as follows:

| stage | TN | original us | pair_lanes us |
|---|---:|---:|---:|
| o | 16 | 64.840 | 2.870 |
| o | 32 | 70.020 | 4.310 |
| down | 16 | 63.230 | 2.950 |
| down | 32 | 69.840 | 4.060 |

All nine legal AR geometries pass bitwise against the ordered BF16 reference
in5/5 fresh TP2 B16 processes, with resident1 and zero stack/static memory.
**inferred** This controlled result identifies packet-to-lane assignment and
coalescing as the important task-body mismatch. It does not establish an E2E
speedup. TP2 B1, TP4 and complete-model gates must pass before promotion.

**verified** The separately approved SDK stream adapter passes5/5 fresh
initialization/event/copy/explicit-destroy processes; six focused CPU lifecycle
checks pass (full focused CPU suite56). It is integrated only into the TP engine.
The edit acquired GPUs1–5's existing locks between payloads, preserving source
identity during every in-flight check. Full-model qualification continues.

## Complete-model acceptance of the local mapping change

- **verified** TP2/TP4 × B1/B16 ordered-BF16 AR units: each5/5 fresh processes,
  nine legal geometries/process, zero bit mismatches; all-rank resident1, stack/static0.
- **verified** Full TP4 B16 C-1:8192 positions, ratio1.0, maximum gap0.171875;
  five fresh1024-token C-2 processes pass. Tokens also match the old same-geometry
  implementation byte for byte. Full-model resources:233 MTreg, stack/static0, resident1.
- **verified** Same-card-group A/B/A/B, two fresh processes/arm, warm1+timed2:
  old12.537724/12.515307ms; new3.081912/3.080873ms. No trace/DIAG/watchdog.
  `MX7_tp_optimization_c550.csv` binds the measurements, pin and binaries.
- **verified** The first SDK full-model litmus produced correct tokens but exited−6
  when tensor storage cleanup tried to record an event after stream destruction.
  That failed exit is retained in the correctness CSV. The corrected engine
  releases pinned copy storage inside generation, returns ordinary CPU storage,
  and frees GPU storage before destroying streams. Four-card pinned-lifetime
  probes and a complete new B16 injected-delay cohort both pass5/5 fresh processes.
- **verified** `5636b54ae` contains the lifetime correction; no AR fence, number
  format, tolerance, HF reference or guard policy changed.
- **stated** External baselines remain the original frozen30 passing cells.
  Communication calibration is on C0's primary{1,3} and TP4{1,2,3,5}; the earlier
  {4,5} microbenchmarks are diagnostics. New ten-scope solver selection continues.

## First reselected optimized scope

**verified** TP2 B16 completes source-bound top4/1% selection using eligible cost
SHA256 `2927cf91afbc8589f8706d50699bc837fa7a6898978ed0fefed6fcab2c906f6e`. Final TPOT is3.561300ms (fresh processes3.560129/3.562471ms); C-1 and5/5 fresh1024-token C-2 checks pass.
Actual resources are234MTreg, stack/static0 and resident1 on both ranks.
This is a solver-selected result, separate from the TP4 same-geometry control.
The other nine scopes and selected-pin delay/trace gates remain in progress.

**verified** The reselected TP2 B16 injected-delay build also passes5/5 fresh
1024-token processes against its ordinary final binary. Production and the
unit-qualified prototype match in all319 frozen include files after removing
only the added mapping comment (`pair_lanes/PRODUCTION_UNIT_EQUIVALENCE.json`).

**verified** TP4 B16 re-selection also completes: final TPOT3.071796ms,
all four candidates pass C-1/two-process C-2, and the selected rebuild passes
C-1/five-process C-2. Both fresh timing processes agree with fresh C-2 in all
12 warm/timed requests. Selected-pin delay qualification and trace are queued;
the remaining eight scopes continue with the same frozen baseline matrix.

**verified** Four timed requests per reselected B16 scope give median wall
E2E minus CUDA TTFT minus rank0 CUDA decode event time of0.741us/step(TP2)
and1.412us/step(TP4). Per-rank prompt/decode event medians are close.
**inferred** The remaining time is predominantly device execution and peer
waiting; the persistent two-launch pipeline is preserved. Final trace will
separate stage costs. The source metrics are each scope final_timing references.

## Reselected TP4 B16 trace attribution

**verified** Rank0 mean of28 per-layer medians (us), original/new selected
geometries respectively: qkv14.426/14.636; attention42.594/42.614;
o+AR166.449/15.497; gate_up16.302/16.791; down+AR202.020/19.777.
**verified** Head per-step213.0/137.5us uses identical TN32/WN8/D3/Pw8
coordinates. Its median inter-rank duration spread falls72.09→2.06us.
**inferred** Reduced upstream arrival skew contributes; this is not an isolated
head-body speedup. Maximum per-step inter-rank differences fall from
83.42/98.60us(o/down) to9.28/3.94us. These traces are diagnostics; the
same-geometry, uninstrumented E2E P-PAIR supplies the enabling evidence.
**inferred** Attention V2 is now the largest remaining per-layer component.
MX-7§10.1 excludes attention V3, so this opportunity is recorded without
expanding the implementation or candidate domain. Complete trace rows and
resource checks are appended to their existing CSVs.

## Remaining pipeline differences from current SM90 selected pins

**verified** All ten H800 TP pins set `group_wait=true`, `weight_stream=false`.
TP4 B16 additionally selects cluster2/qkv multicast; the other nine use cluster1.
SM90 has160threads(32 producers/128 consumers), TMA rings and separate KV-group
release/acquire at qkv→attention (`Sm90DenseProgram.cuh:679`,
`Codegen.cpp:2514`). C550 retains512-thread V2, grid stage boundaries and
32-key attention blocks. The coarse persistent launch/sharding pipeline is
the same; the intra-CTA and qkv handoff implementations are not identical.
Cross-stage weight streaming is not an enabled H800 advantage in these pins.
**inferred** Group readiness and attention body changes are future opportunities,
not measured causes of the remaining C550 gap. They require a separate design
from the current MX-7§6–§10 protocol; neither is implemented here. Source/pin
snapshots: `runs/mx7/sm90_comparison/sm90_reference/pipeline/REFERENCE.json`.

**verified** TP2 B1 re-selection completes at2.252859ms(final two-process median),
with C-1/five-process C-2 passing. Its four candidates are2.254064/2.259908/
2.248423/2.250167ms; the1% rule retains predicted rank1. TP4 B1 and the
remaining six middle-batch scopes continue; B1 selected-pin delay checks remain.

**verified** Selected TP2 B1 injected-delay qualification completes5/5 fresh
1024-token processes, exact against its ordinary selected final binary. The
two B16 cohorts also remain5/5; only selected TP4 B1 delay qualification waits.
