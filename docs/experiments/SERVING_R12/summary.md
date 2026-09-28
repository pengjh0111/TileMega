# R12: plan-directed serving data flow

Baseline: `2dfea08f19f662a1670da0d11f680fe711d94666` (`origin/tilemega` at start). Prompt: `/root/Prompt/TileMega_R12_prompt.md`, SHA256 `bbe780cb14874f2c7f184a2909f84f244e2f6cc72fe5ae8db56a95577cde3acc`. Final HEAD and commit count: pending final acceptance.

The four-cell end-to-end acceptance has **not yet run**. Development measurements below do not establish G-1 through G-5. The GPU currently has roughly 36 GiB allocated by a process not listed by the compute-app query; serving model creation failed with CUDA OOM and the serving calibration marked its base BF16 segment uncalibrated. Evidence: `/root/r12_work/s1_pages_b16.log`, `/root/r12_work/cli_calibrate2/commands/calibrate/stderr.txt`. No performance comparison will use those contaminated attempts.

| ID | Code and commit | State |
|---|---|---|
| FX-1…7 | `GemmStageTaskBody.h`, `ServingPages.cuh`, `LastArriverTaskBody.h`, `ServingPruning.h`; seven commits `6e52f53e8`…`ef936a6c5` | Code complete; directed unit checks passed; long-sequence C-2 pending |
| PL-1 | `weights.py:122`, `PagedGemmTaskBody.h`, `ServingPages.cpp`; `735cd9f3f`, `a28bda80a` | Tile packing and whole-page loader implemented; E-1 measured |
| PL-2 | `PagedGemmTaskBody.h:28`; `735cd9f3f` | Four activation slots implemented; small GEMM test passed |
| PL-3 | `PagedAttentionTaskBody.h:6`, `PageLayout.h`; `342c982f8` | Q-row workspace and KV co-page implemented; small attention test passed |
| PL-4 | `ServingPages.cuh:23`; `a28bda80a` | Second page-stream cursor and L2 lookahead implemented; S-1b pending |
| SY-2 | `EventSync.cuh:41`, `ServingSyncCalibration.cu`; `0ec3de432`, `a28bda80a` | V3 acquire/release path implemented and calibrated in a development target; 50-process check pending |
| MS-1 | `ModelPlan.cpp:926`, `ServingEpilogue.h`, `weights.py`; `6cb731bd0` | Deferred norm implemented; generated decode has zero `kRMSNorm` stages; C-1 pending |
| MS-2 | `HandoffPass.cpp`, `ServingPages.cuh:205`, `ModelHarness.cuh`; `a28bda80a` | Last-arriver merge/combine/argmax implemented with elided queue stages; C-2 pending |
| DP-1 | `ServingSemanticLifting.cpp`, `ServingTaskIndex.h`; `342c982f8` | Decode attention and merge use group-major task order; host task-index test passed |
| DP-2 | `Frontend.cpp:1027`, `PagedGemmTaskBody.h:191`; `61d637fec` | ISL-proved phase maps and device gate implemented; 47 selected Llama B1 edges in generated code; mode comparison pending |
| SL-1…2 | `SkeletonSearch.cpp`, `ServingPagedCalibration.cu`; `61d637fec` | Seed uses actual resident limit and paged body fit exists; final uncontaminated calibration pending |
| SL-3 | `StageFlowModel.cpp`, `FluidExecutionSimulator.cpp`, `FlowPreparation.cpp`; `61d637fec`, `6099ef26d`, `ae4b4ac7c` | Page-plus-lookahead capacity, inline reducer on final producer, ISL-proved phase release in outer search implemented; generated-plan mode comparison pending |
| SL-4…5 | `cli.py`; `c0842b3c4` | Pages/L2 alternating measurement selection implemented; model fidelity pending final plans |
| RL-1 | `ServingRuntime.cuh:409`, `ServingPages.cuh:585`, `engine.py`; `c0842b3c4` | Device loop and separate lag table implemented; 256-step comparison pending |
| EV-3 | — | Not run; four cells, C-1/C-2, 50 processes and paired vLLM pending |

E-1 decides whether row-strided access or loader instructions dominated the R11 slowdown. At 128 CTA with TN=128, TK=64, the R11 row loader took 0.2284 ms, R12 tile loader 0.0696 ms with `evict_first` and 0.0649 ms without, and the old collective 0.1024–0.1074 ms. A matched 512-MiB stream test gave 856.5 GB/s for row segments and 902.4 GB/s for contiguous pages (+5.4%). The R12 default leaves `evict_first` off. Evidence: `/root/r12_work/e1/{row,tile,noevict}.run.log`, `/root/r12_work/paged_repeat.log`.

CPU and small-shape validation: `stage_flow`, `flow_runtime_release`, and `serving_task_index` passed (3/3); `serving_epilogue`, `serving_attention_cases`, `paged_gemm`, and `last_arriver` passed (4/4). The K-phase flow test proves that a first phase can start consumer work early while publication waits for the last phase; the fluid test proves an inline reducer leaves its former queue slot free. Exact ISL phase projection proved and priced 64 Llama and 112 Qwen3 decode edges in the outer flow search. Evidence: `/root/r12_work/cpu_tests.log`, `/root/r12_work/current_tests.log`, `/root/r12_work/inline_test.log`, `/root/r12_work/phase_import_{full,qwen}.log`. A generated Llama B1 paged plan compiled for sm_80/89/90/100/120 (5/5). These are compile checks, not execution on those GPUs. Evidence: `/root/r12_work/arch_compile/report.json`. SASS auditing including both loop kernels found zero FP64 instructions (`/root/r12_work/arch_loop_sass.log`).

Quick experiments S-1, S-1b and S-2: pending uncontaminated GPU execution. EV-3 G-1…G-5: **not evaluated**. The two generated Llama B1 candidate libraries are development artifacts, not accepted plans; they were compiled while external GPU occupancy made their top-3 timings unusable.

| Gate | Criterion | Current result |
|---|---|---|
| G-1 | Four paired speedups exceed R10-C and R11 by the three-run range | Not evaluated |
| G-2 | C-1 teacher-forced gap in four cells | Not evaluated |
| G-3 | C-2 timing, L1/L2, K-phase and loop/step identity | Not evaluated |
| G-4 | 50/50 fresh-process Llama B16 protocol check | Not evaluated |
| G-5 | TileMega/vLLM throughput ≥1 in all four cells | Not evaluated |

Deviations and incomplete work: commits group tightly coupled changes instead of one commit per every PL/SL subitem, because shared executor and ABI definitions changed together. Outer-search phase release is now derived by an exact relation proof; failed proofs retain a complete task edge. No uncontaminated end-to-end speed claim is made.

Next: rerun serving calibration when the GPU is free; perform S-1/S-1b/S-2, C-1/C-2 and the 50-process check; then run the four-cell paired EV-3 once and record the exact G-1…G-5 results here. If a gate fails, locate it to the responsible code or calibrated constant before selecting the next change.
