# AS-1 probes — 2026-10-20 (in progress)

`verified`: read/executed here; `stated`: source assertion;
`inferred`: not independently established. E-0/E-1 and the fixed bias-free E-2 encoder pass the full chain.

## Fixed platform and build

- **verified**: physical 0/PCI 55:00.0 and physical 1/PCI D7:00.0 are
  Ascend950PR, runtime SKU `Ascend950PR_9579`, each 28 AIC / 56 AIV.
  Separate ACL inventory processes confirm visible logical 0 maps to the
  physical ID selected with child-only `ASCEND_RT_VISIBLE_DEVICES`.
- **verified**: early probes select physical 0; final E-2 probes select physical 1. CANN 9.1.0,
  driver 25.7.0, BiSheng clang 15.0.5 (2026-06-11).
- **verified**: CATLASS official GitCode checkout
  `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`, unchanged.
  Native CMake uses `find_package(ASC)` and `LANGUAGES ASC CXX`,
  `CATLASS_ARCH=3510`, `--npu-arch=dav-3510`. CPU compiler concurrency is 1.
- Reproduction: `tools/probes/ascend950/README.md`; actual compiler commands:
  private `build/probes/compile_commands.json` and `logs/*build*.log`.
  Generated sample copies remove reset teardown and request at most two groups.
  There is no device lock, exclusive mode, CUDA build or NVIDIA test.

## Probe matrix

| Probe | Observed scope and current conclusion |
|---|---|
| P-1 | **verified** explicit `__cube__` and `__vector__` compile, run and write expected IDs. `msobjdump` confirms AIC/AIV metadata. MIX compiles and runs below. Native `--cce-res-usage` is accepted; numeric resource-report extraction remains open. |
| P-2 | **verified** G=1/2/4 topology: AIC block g, AIV block 2g+lane, subblock lane; both report block count G. Global worker IDs are g and G+2g+lane. 3G workers, G launch blocks. G=2 is also exercised with interdependent workers in P-6; use two groups initially. |
| P-3 | **verified** G=2, 6 workers × 16 epochs, one host launch and distinct per-worker records. G=1/4 also pass. This is a persistence probe, not the final GEMM plan. |
| P-4 | **verified** AIV entry touches 217088 bytes (212 KiB) UB and returns both boundary values. ACL available UB=221184, platform table=253952. L1/L0 budgets are queried, not allocation-limit proofs. A 1 MiB UB program also compiles (AIV ELF entry retained), so compilation alone is not a capacity proof. It is never executed and host budget preflight rejects it. Remaining buffer diagnostics pending. |
| P-5 | **verified scoped reuse** official EVG GM mode-2 flags 0/1, 16 tiles/group and reverse acknowledgement at 15, 5/5 fresh processes. Evidence: `probes/evg_flag_reuse_summary.json`. Mode 4 remains disabled. |
| P-6 | **verified** four cross-group directions, 16 epochs/direction, 128 B and 4 KiB reused payloads, stale-cache pre-read variant, 5/5 fresh processes, zero mismatches/timeouts. Single-owner monotone events only; not a multi-producer atomic counter proof. |
| P-7 | **verified** GM path via actual Cube output and AIV DMA in P-6. **verified FP32 L0C→UB** EVG Add, 16 tiles/group, 5/5 fresh processes (`evg_ub_reuse_summary.json`). BF16 direct route, UB→L1 and NZ output remain disabled/unverified. BF16 ND tails pass the native GEMM body. |
| P-8 | **verified** safe copies of target 43 at 64×128×256, 50 at 1×96×129, 64 Add GM at 64×128×256 report `Compare success`. UB example at 64×8192×32 also reports `Compare success` in 5 fresh processes. |
| P-9 | **not run** no calibrated performance numbers; current timings include startup and external load. Fixed implementation selection only; no invented target rates. |
| P-10 | **verified** unlocked wrapper timeout/owned-process isolation (T-14), one visible device, raw `GetSystemCycle` records. Cycle-to-ns calibration, device trap and profiling not established. |
| P-11 | **verified scoped SIMD** LayerNorm/Erf GELU/Add regular/tail and masked attention S1/S13/S16/all-masked pass; SIMD PV, no SIMT. |

The upstream CATLASS basic GEMM executable's metadata is **verified**
`AIC_ROLLBACK` with MIX ratio 1:2. Its Cube arithmetic is not evidence of a
native AIC-only entry. The separate explicit entry probe establishes that.

## GM message-passing protocol and evidence

Code: `Target/Ascend/DevicePrimitives.h`, `Backend/Ascend/GemmTile.h`,
`tools/probes/ascend950/gm_visibility.cpp` (headers under `include/tilemega`).

1. Consumer scalar-reads previous payload, then publishes `preloaded[epoch]`.
2. Producer awaits previous acknowledgement and current pre-read. AIC uses
   CATLASS BlockMmad on identity A and changing FP32 B; AIV uses DMA via UB.
3. Producer explicitly completes FIX→S or MTE3→S, executes DSB and publishes
   ready with installed `WriteGmByPassDCache`; control reads bypass DCache too.
4. Consumer observes ready, executes DSB plus data-cache clean/invalidate,
   verifies every payload bit, and publishes acknowledgement before reuse.
5. Polls are bounded; abort is shared and terminates the protocol on failure.

**verified measured scope**: single publisher/event, G=2, FP32 payload,
32/1024 elements, 16 epochs, actual AIC/AIV DMA, all four role directions.
No CUDA release/acquire assumption, scalar atomic assumption, or claim about
unmeasured buffer sizes/epoch ranges is made. Aggregated multi-owner trigger
counts must be rejected until separately implemented and verified.

| Job | PID | Seconds including startup | Mismatches |
|---|---:|---:|---:|
| gm_visibility_all_1024_1 | 3173533 | 4.074 | 0 |
| gm_visibility_all_32_2 | 3220558 | 4.273 | 0 |
| gm_visibility_all_1024_3 | 3264851 | 4.224 | 0 |
| gm_visibility_all_32_4 | 3340935 | 4.071 | 0 |
| gm_visibility_all_1024_5 | 3406097 | 4.071 | 0 |

All five used executable SHA256
`de4f9473d136c6f5edf756d9229455a0e77c172f75281f5d24bc6504b1ce703e`.
Receipts: `runs/as1/probes/gm_visibility_summary.json` and each job's
`job.json`, `stdout.log`, snapshots. These are not performance measurements.

## Failure and shared-server record

- **verified** `entry_cube` timed out after 30.114 s, exit 124. Its saved
  stack is in `aclrtSetDevice → TsdOpen → WaitRsp`; kernel launch was not reached.
  Device 0 subsequently reported healthy, zero utilization, no residual process.
  One retry with a 90 s host limit passed in 14.492 s; the vector entry passed
  in 14.537 s. Both used binary SHA256
  `c76c9044e9ecd41a16ef56fdb99cb0260eec8ab12eb89c072df1496c18679370`.
- **verified** all jobs were sequential, single-device, bounded and unlocked.
  External processes were observed; only our timed-out process group was stopped.
  No OOM, device reset, exclusive setting or operation on another job occurred.
- **verified** last disk sample: runs 2592768 bytes, build 139161600,
  cache 115359744, tmp 180224; protected dependencies included, private work
  totals 2405335040 bytes. Filesystem free 13507942952960 bytes.
  Manifest dry-run found zero removable files; failed-init evidence is protected.

## Gate and downstream restrictions

G-PROBE-AS is **partial**: native entries, MIX IDs/persistence and scoped GM
visibility are verified. P-4 remaining resource diagnostics, BF16 P-7 direct
routes and P-10 timer conversion are pending; P-9 component-rate calibration remains unrun; limited contended E-2 timing is separate.
Use G≤2, residency=1, fixed candidates, single-owner GM events initially.
Direct routes, mode 4, SSBuffer messaging, SIMT and performance auto-ranking
remain off. E-0 CTG/Solver/Plan/device execution is verified in `AS1_ascend950.md`.
Native AIV bodies and fixed E-1/E-2 pass; native single-task composite integration remains unimplemented.

### E-1 AIV bodies and exact executor follow-up (verified)

`AivBody.h`/`as1_aiv_body` use 6656 bytes UB, FP32 arithmetic, BF16 load/store.
RMSNorm/SiLU/Add each pass 1x128 and 4x129 independently; all six outputs are
bitwise equal to CPU BF16, within the fixed FP32 body tolerance, guards intact.
`runs/as1/probes/aiv_bodies_summary.json` records binary hash and exact errors.
These body probes are separate from real five-stage CTG/Solver E-1 validation:
G1 and forced G2 each pass 5/5 new processes, 16 epochs; B1 also passes both.
Native BF16 direct-UB, mode4 and SSBuffer remain disabled. See
`AS1_ascend950.md` for the initial full-FP32 check failure and exact scope.

## Encoder and second-card additions (verified)

- Physical 1 MIX G2/16 epochs gives 96 unique topology records; ID mapping
  matches physical 0. `jobs/topology_card1_g2_e16/`.
- Physical 1 GM visibility repeats all four AIC/AIV directions, pre-read cache,
  32/1024 floats, 16 epochs, 5/5 fresh processes, zero mismatches/timeouts.
  `probes/gm_visibility_card1_summary.json`; no device locks or empty-card gate.
- Six LayerNorm/Erf-GELU/Add regular/tail probes pass. Erf is explicit CANN
  `<lib/math/erf.h>`; default tanh-approximate GELU is not substituted.
- Attention S1/S13/S16 plus fully masked case pass using register SIMD PV.
  S16 BF16 output is bitwise CPU-equal; S13 max abs 3.8147e-6.
  `probes/encoder_attention_reg_summary.json`. Prior vector PV numerical failures
  and scalar PV 90 s stream-sync timeout on physical 0 are protected evidence.
  Host stopped only its own process group, no reset; device-work stop latency unknown.
- Final E-2 L2 and L1 executors each pass 5/5 fresh processes on physical 1,
  four epochs each, one launch and bitwise-identical final output. This proves
  only the fixed ten-stage encoder barrier scope, not arbitrary barrier graphs.
  `runs/as1/e2/e2_summary.json` indexes every PID, binary hash and output hash.

## R-1–R-11 capability evidence

| Requirement | Current installed 950PR scope |
|---|---|
| R-1 | **verified** both physical IDs/SKU, 28 AIC/56 AIV per card; selected launch G≤2, not 84 blocks. |
| R-2 | **verified** API bank budgets and 212 KiB UB touch; full AIC-bank allocation limits/register/SSBuffer budgets unverified. |
| R-3 | **verified** MIX(1:2), block/subblock IDs, G-block launch, batch mode and multiple tasks/epochs in one launch. |
| R-4 | **verified** official EVG mode2 reuse, flags 0/1 and reverse ack 15, 5/5; mode4 disabled. |
| R-5 | **verified** bounded single-owner bypass-DCache GM protocol with DMA completion/cache invalidation, not general atomic release/acquire. Scalar atomic-return semantics unverified. |
| R-6 | **verified** GM C/V handoff and official FP32 L0C→UB; native BF16 direct and UB→L1 disabled pending bodies/probes. |
| R-7 | **verified** CATLASS BF16 ND output/stride/tails and official FP32 UB epilogue; no general NZ/direct-layout support claim. |
| R-8 | **unverified** component sustained throughput/overlap; only contended full-encoder timings, no target cost calibration. |
| R-9 | **verified** native ASC/BiSheng/ACL and pinned CATLASS 3510; actual compiler lines retained. |
| R-10 | **verified** raw cycle reads and owned-process timeout; cycle/ns conversion, Trap and profiler remain unverified. |
| R-11 | **verified** SIMD reductions, Erf GELU, mask/softmax and register PV for fixed encoder; SIMT/gather/RoPE unimplemented. |
