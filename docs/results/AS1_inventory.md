# AS-1 inventory — 2026-10-20

Evidence: `verified` means read or executed in this session; `stated` means a
source assertion; `inferred` means a conclusion awaiting its own probe.

## Source and scope

- **verified**: origin `https://github.com/pengjh0111/TileMega.git`;
  initial `origin/tilemega` and HEAD `297ac803847e902f954a0852751c2667c549f597`.
  A fresh checkout was clean; neither local nor remote `tilemega-ascend` existed.
  Work proceeds on the new `tilemega-ascend` branch. Pushes name that ref explicitly.
- **verified**: exact initial user-message bytes retained at
  `/home/d50064638/junhuip/tilemega-as1/Prompt/TileMega_AS1_prompt.md`.
  SHA256 `132596ba7075142fc33a36318b66948a09d4707d6be32bd063054126b39e44f6`.
- **verified**: read root AGENTS/CLAUDE, README, TOOLS, DEPENDENCIES,
  BUILD_MLIR and relevant skeleton Plan/Codegen sections. There are no nested
  AGENTS/CLAUDE files in TileMega's implementation directories.
- User's AS-1 instructions override the README/TOOLS NVIDIA device lock,
  CUDA dependency, checkpoint download, and speedup acceptance conventions.
  CTG/ISL semantics, evidence and commit conventions still apply.
- No CUDA build, NVIDIA test, model-weight download, global configuration
  change, device lock, exclusive mode or operation on another job was performed.

## Directories and disk

- **verified**: only project-owned paths created under
  `/home/d50064638/junhuip/TileMega` and `tilemega-as1`.
  The directories did not exist before this work.
- **verified**: ext4 `/dev/mapper/ubuntu--vg-ubuntu--lv`, mounted at `/`.
  `disk_baseline.txt`: total 30,595,029,204,992 bytes; available
  13,527,906,037,760 bytes; 328,125,833 available inodes. No quota executable.
- **verified**: pre-CP-1 generated outputs: runs 868,352 bytes; build 126,976;
  cache 107,044,864; tmp 57,344 (allocated bytes; see the raw snapshots).
  Filesystem availability changed with shared workload; this is not project usage.
- Policy: stop substantial generation below 15% free space or 5% free inodes;
  10 GiB generated-output soft budget; compile concurrency 1; no candidate sweeps.
  `configs/bench/as1_disk_policy.json` records the thresholds.
- `scripts/as1_cleanup.py`: default dry-run, manifest/file-identity checks,
  path/symlink/active-PID checks, selected/evidence protection, latest two builds.
  It unlinks registered individual files, never recursively removes directories.
- **verified T-15**: 10 boundary cases, one intended fixture removed, nine
  protected survivors. CP-1 dry-run/apply removed 0 files / released 0 bytes.

## Device and software

| Property | Result and evidence |
|---|---|
| Physical cards | **verified** 0 / PCI 55:00.0, 1 / PCI D7:00.0, both Ascend950PR |
| Exact runtime SKU | **verified** `Ascend950PR_9579` on each card |
| Visibility | **verified** separate processes with `ASCEND_RT_VISIBLE_DEVICES=0` / `1`; each reports count=1, logical=0, correct physical ID via `rtGetDevicePhyIdByIndex` |
| Runtime cores | **verified** each selected card reports AIC=28, AIV=56 via ACL |
| Launch groups | **unverified**; 28 is a runtime core count, not a verified MIX residency limit |
| CANN / driver | **verified** CANN 9.1.0, driver/npu-smi 25.7.0 |
| BiSheng | **verified** clang 15.0.5 build 2026-06-11; native compiler help retained |
| CMake / host | **verified** CMake 4.4.3, GCC host; 256 logical CPUs, ~503 GiB RAM |
| ACL UB budget | **verified** 221,184 bytes; use this conservative available budget initially |
| Platform table | **verified query**, allocation not yet tested: L1=524,288; L0A/B=65,536 each; L0C=262,144; UB=253,952; L2=134,217,728 bytes |
| UB discrepancy | **inferred** reserved space may explain the 32 KiB difference; not an allocation proof |
| Default Python | **verified** 3.11.14, torch 2.7.1+cpu; no torch_npu/numpy |
| Private NPU venv | **verified** `tilemega-as1/envs/as1-npu`, read-only system packages from `/data/miniconda3/envs/dev`; torch 2.7.1+cpu, torch_npu 2.7.1.post7.dev20260707, numpy 2.4.3 |
| torch_npu sanity | **verified** physical 0, one visible NPU, 128 BF16 additions exactly match CPU; this is not a TileMega test |
| CATLASS | **verified checkout** official `https://gitcode.com/cann/catlass.git`, SHA `2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52`, CANN Open Software License v2; no submodules initialized |
| CATLASS compatibility | **stated** local README requires CANN >=9.0 for 950 C++; sample compilation/execution remains an AS-P gate |
| Native build interface | **verified code reading** `find_package(ASC)` / `LANGUAGES ASC CXX`, `CATLASS_ARCH=3510`, `--npu-arch=dav-3510`; not `LANGUAGE ASCEND` |
| CTG dependency | **verified** TileMega requires pinned LLVM/MLIR 23.0.0git commit `23a60f15f2fcafcf67b95b0a035053579958b732`; development package not yet found; read-only user-directory search ongoing |

`scripts/as1_env.sh` is copied to the private work scripts directory and sourced
explicitly. It redirects Python/pip/Torch/CANN temporary files, logs and caches;
no shell rc or shared package is changed. CANN path controls follow the official
[9.1 cache reference](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0006.html)
and [work reference](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0007.html).

## Current coupling audit (all locations verified)

| ID / original assumption | Current location | Ascend consequence / work / test |
|---|---|---|
| F1 homogeneous SM workers | `Runtime/Launcher.h:8`, `lib/Solver/FlowPreparation.cpp:395`, `SkeletonSearch.cpp:505` | Split groups/tracks/launch; AS-R; T-1/2 |
| F2 whole-grid Home/Spread | `lib/Solver/PlanSkeleton.cpp:198`, `:205` | Role-restricted domains; AS-R; T-3/4 |
| F3 blockIdx queue owner | `Codegen/tasks/ModelHarness.cuh:2108`, `ModelRuntime.h:782` | Explicit WorkerMap; AS-D; T-10 |
| F4 POD task/params | `Codegen/tasks/ModelRuntime.h:472`, `:768` | Preserve dependency/count semantics, new checked POD ABI; AS-B/D |
| F5 SM/CTA target | `Target/TargetSpec.h:68`, `:94` | Add role budgets/capabilities; AS-R/B; T-2/5 |
| F6 smem/architecture legality | `lib/Solver/BackendCostQuery.cpp:16`, `Solver/BackendCostQuery.h` | Per-role buffer and arch-tag legality; AS-R; T-3/5 |
| F7 implementation contract | `Solver/ImplementationContract.h:34` | Keep operand math; attach role/layout/flags; AS-R/T; T-11/13 |
| F8 nine CUDA cost lanes | `Solver/CostModel.h:90`, `CoResidency.h:20` | AIC/AIV lanes, one shared GM budget; AS-R/C; T-6 |
| F9 CUDA build and GEMM ABI | `CMakeLists.txt:17`, `tools/commands/compile.cpp:2475`, `Codegen/tasks/GemmStageTaskBody.h:549` | Separate native build/body; AS-B/T; T-10/11 |
| F10 SharedStorage/PTX | `Codegen/tasks/ModelHarness.cuh:285`, `EventSync.cuh:47`, `executor/Async.cuh` | Separate L0/L1/UB; exclude CUDA primitive path; AS-D; T-7/8 |
| F11 26 task kinds | `Codegen/tasks/TaskBase.h:72`, `ModelHarness.cuh:697` | Explicit small support set, reject rest; AS-T; T-11/13 |
| F12 monotone event epochs | `Codegen/tasks/ModelHarness.cuh:1206`, `EventSync.cuh:45` | Preserve counts; independently prove DMA/cache ordering; AS-P/D; T-7/8 |
| F13 ptxas/occupancy probe | `lib/Solver/BackendCostQuery.cpp:47`, `tools/commands/compile.cpp:142` | Native compiler receipts; AS-B/C |
| F14 NVIDIA guard | `python/tilemega/bench/gpu_guard.py:34`, `serving/measurement_policy.py:25` | Independent unlocked `bench/as1_run.py`; AS-S; T-14 |
| F15 importers | `python/tilemega/dnn/export.py:23`, `:46`; `Solver/ModelDescription.h:86` | Reuse DNN semantics with tiny local BERT; AS-V; T-12 |
| F16 CUDA launcher/ABI | `Target/ArchDispatch.h:8`, `Codegen/tasks/ModelHarness.cuh:5209`, `executor/ServingLaunch.cuh` | Native ACL launch and dispatch; AS-B/D; T-10/13 |

Header locations are under `include/tilemega/`. Full matches, including additional
CUDA includes, atomics, compiler and Python paths, are indexed in `code_audit.json`
and `coupling_audit.txt`. No mass textual replacement has been made.

## Fixed workloads and execution record

- `configs/ascend/as1_workloads.json`: seed 20261016, BF16/FP32 accumulation;
  E-0 64x128x256 and 33x96x129; E-1 RMSNorm→GEMM→SiLU→GEMM→residual,
  B=1/4; **E-2 fixed to BERT encoder block**, B1 S16 H256 heads4 FFN1024.
  Dropout disabled, explicit attention mask, random parameters only. All E-0/1/2
  are `not_run`; no replacement of the CTG/Solver chain is accepted.
- **verified T-14**: timeout returns 124, stops its own descendant, leaves the
  separately started sentinel alive. No device kernel launched by that check.
- **verified** ACL inventory: physical 0, 2.428 s; physical 1, 2.218 s;
  zero kernels. torch_npu sanity: physical 0, 10.339 s including Python startup.
  All ran sequentially. Other jobs were observed on physical 1 and, later, 0;
  functionality was permitted. Timing is not a performance measurement.
- The wrapper's first HBM parser picked the memory column (0/0); fixed after
  comparing the preserved raw snapshot. Later receipts parse 128 GiB HBM
  correctly. Original evidence is retained rather than rewritten.
- No device job timed out/OOMed. The deliberate **CPU-only** timeout check is
  separate. No cross-core synchronization claim has been made.

## Raw evidence index and gate

`runs/as1/inventory/{system.txt,npu.txt,software.json,baseline.txt,acl_device_0.json,
acl_device_1.json,code_audit.json,coupling_audit.txt,disk_baseline.txt,disk_after_cp1.txt}`;
`runs/as1/jobs/{acl_inventory_0,acl_inventory_1,torch_npu_environment,runner_timeout_*}`;
`runs/as1/{manifest.jsonl,cleanup_history.jsonl,cleanup_*_cp1.json}`;
private `build/acl_inventory/` retains all compiler/linker failures and the
working binary. Raw outputs and generated files are ignored, never staged.

**G-INV-AS: verified.** Native GEMM/MIX/persistence/synchronization are not yet
verified; their capability bits remain disabled. Next: native compiler/topology
probes, then role-aware Track/Plan contracts and CTG build dependency resolution.
