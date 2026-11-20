# TileMega single-GPU A800 experiment

## Revisions and environment

**Verified.** The paper audit base is `7425a5242580af16a7ea43be742e313a6eedb351`. The inherited experimental parent is `cc62bb703de5e8f4f0b03b21c4e58f04ef4ce0b3`. The A800 branch is independent. CUTLASS in the inherited worktree matches the recorded gitlink `dc45f979ae336a235da1676b311f35efeb30149a`.

**Verified.** The Paracloud `n46h` deployment uses Slurm 24.11.6 and Apptainer 1.4.5. Login-node checks confirmed CUDA compiler 12.8.93, Codex CLI 0.170.0, the HTTPS proxy, and an empty CUDA mask outside Slurm. The workspace launcher injects the driver through Apptainer only in a GPU allocation. Slurm assigns memory from its per-GPU policy.

## A800 evidence

**Verified.** Device probe `343820` completed with strict host process ownership, 31 samples at one-second intervals, and CUDA/NVML identity agreement. The allocated A800-SXM4-80GB has compute capability 8.0, 108 SMs, 40 MiB L2, 65,536 registers per SM, 166,912 bytes of opt-in shared memory per CTA, and MIG disabled. The NVIDIA driver was 580.105.08. The raw receipt and uncalibrated target are in `runs/a800/probes/343820/`.

**Verified.** A later Slurm allocation on the same node exposed a different UUID. The pinned calibration gate rejected it in one second. New diagnostic cohorts record their device within the same allocation. The result collector requires the exact cohort UUID.

**Verified.** Bounded job `343865` ran the first SM80 native transport calibration process and completed in 76 seconds, including both guarded admission windows. Diagnostic medians include 1,658.28 GB/s for the tested HBM transfer, 2,218.77 GB/s for the tested L2-resident transfer, and 12.8805 GB/s for the tested per-CTA swizzled activation copy. These are single-process component measurements. The command, binary hash, and receipts are in `runs/a800/calibration/343865/`.

**Verified.** The first full calibration attempt `343873` reached the MoE epilogue and caught an illegal memory access. The older calibration fixture omitted the current shared routing state. The fixture now supplies the routing layout consumed by the production epilogue and MoE body. It derives the maximum feasible ring depth from actual shared-memory limits.

**Verified.** Job `343908` completed all five fresh calibration processes on one A800 in 238 seconds. The strict guard accepted every process. The calibrated profile contains 719 component rows and is saved as `configs/targets/sm_80_a800_sxm4_80gb.json`. Its HBM, L2 and swizzled per-CTA activation-copy medians are 1,673.50, 2,215.99 and 12.88 GB/s. The report and raw results are in `runs/a800/calibration/343908/`.

**Verified.** Job `343941` repeated the expanded native epilogue and MoE shape calibration in five fresh processes on the same A800. The 1,009 aggregated rows are in `runs/a800/calibration/343941/native-constants.csv`. The measured target for the expanded shape set is `configs/targets/sm_80_a800_sxm4_80gb_expanded.json`, SHA-256 `dacb24e0da7e18787ecf98ff33a67926dffb00dc164c608dea53ad3e9eccf468`. Existing Llama search receipts refer to the original immutable target.

**Verified.** The A800 driver reported one resident CTA per SM for both inherited Llama decode binaries in job `343889`. The B1/B16 binaries each launch 108 CTAs and request 128,000/129,536 bytes of shared memory. CPU compilation and disassembly accepted all 66 MMA slice probes after updating the probe to call the current RMS-accumulating body. A CPU-only additive price check materialized the Llama B1 CTG using the measured A800 target.

**Verified.** The staged TileMega Python environment loaded PyTorch 2.13.0+cu126 on the login node and reproduced the Llama-3.2-1B shared checkpoint digest `5a53ddca4daa1421acb73fc07e8d328fb7d43f4258d1b6b1bdc0d18be2748c9d`. Native pilot `343916` was rejected during initialization when the non-dumpable worker became invisible under the cluster's `hidepid` mount. The A800 guard retains visible worker identity and disables piped dumps with the Linux one-byte core limit. The original ownership threshold remains in force.

## Measurement protocol

The baseline matrix contains 160 cells, each requiring three independently accepted processes. Each process uses one of the frozen prompt batches, TP1, BF16 weights and KV, and a shared seed-zero checkpoint. N=1 and N=1024 each use one warmup and three formal request-wall repetitions. Process TPOT is `(median(T1024) - median(T1)) / 1023`. The final cell uses the median of process TPOTs.

The native pilot retains separate Llama-3.2-1B prefill and decode artifacts at B1 and B16. Formal timing requires a measured solver selection and five fresh correctness receipts for the exact artifact pair. The final native matrix has 40 cells with three or five independent timing processes per cell.

Pipeline selection uses the guarded linked-cubin occupancy probe and mirrored end-to-end shortlist measurements. Formal timing uses fresh processes. The A800 target requires its own measured constants, and CPU build evidence from A100 is retained under its original run directory.

## Status

The workspace archive and the inherited experiment archive passed SHA-256 verification. All four Python environments passed dependency checks and imported the pinned Torch CUDA runtime on the login node. A separate clean SM80 compiler build linked all 172 Ninja targets in `build-a800/`. The login-node CUDA link stub was sufficient to verify its CLI help. The raw per-attempt output, Slurm receipts and thermal telemetry are under `runs/a800/` and the workspace `logs/`. The two result CSVs are generated from accepted attempts. Speedups require final measurements from both sides of the comparison.


**Verified.** Native Llama B1 completed the 1,024-token step/loop and Hugging Face teacher-forced checks on the main A800 in job `343927`. All positions were within the accepted gap threshold and the maximum logit gap was 0.015625. The initial B16 attempt completed both model checks, but the strict guard rejected a split NVML memory/owner snapshot while the HF child exited. B16 retry `343964` passed the guard and both model checks. It covered all 16,384 positions with maximum logit gap 0.03125. Slurm gave that diagnostic a second A800 UUID. The initial result tables were scoped to the main calibrated UUID.

**Verified.** TorchCompile Llama B1/B16 has three accepted processes per cell on the same UUID. The B1 median TPOT is 4.595 ms and the B16 median TPOT is 8.905 ms. The table includes T1, T1024, throughput, configuration and both input and measurement hashes. The compiler used StaticCache, SDPA, reduce-overhead and two recorded graphs.

**Verified.** The guarded Llama B1 pilots for vLLM (`343956`), SGLang (`343958`) and TensorRT-LLM (`343960`) each completed one 1,024-token timing process on the same UUID and shared checkpoint. Their process TPOTs were 3.252, 2.600 and 2.490 ms respectively. SGLang recorded FlashInfer. TensorRT-LLM recorded its PyTorch backend, TRTLLM attention, CUTLASS MoE support and CUDA Graph capture. These cells remain incomplete until three independent receipts exist.

**Verified.** Qwen3-30B-A3B prefill was exported on the login node from the existing configuration with metadata-only weights, TP1, prompt length 64 and capacity 1,088. The bridge preserves the custom MoE operation and contains 4,957 tasks and 6,060 couplings. The export and its hashes are under `runs/a800/build/qwen30b-prefill-export/`. The A800 prefill artifact and paired runtime still need correctness evidence.

**Verified.** Frozen-input checking and SM80 additive ranking emitted decode code for all eight models at B1 and B16 on the login node. The 16 codegen receipts are under `runs/a800/build/cost-shape-coverage/`. The independent Qwen3-0.6B prefill and native decode pair links for SM80 and passed the correctness-run planner. Its GPU pilot is `344029`.

**Inferred.** The Qwen3-30B-A3B B1 pair uses compatible expert packing recipes across the two phases. The linked decode buffer ABI and prefill plan estimate 58.9 GiB before CUDA context, event and queue storage. The recipes deduplicate approximately 55.4 GiB across phases. The CPU estimate and artifacts are in `runs/a800/build/prefill/qwen30b-packed-b1/`. The paired deployment still needs a guarded on-device measurement.

## Queue

The ordinary `gpu_a800` Slurm dependency chain allows one active allocation at a time. Each allocation and child process has a finite deadline. The single-card tail and stage IDs are in the workspace `logs/`.

| Slurm job | Scope |
|---|---|
| `343965` | Llama B1 measured pipeline selection |
| `343976` | TorchCompile B1 first round on five additional dense models |
| `343986`, `343992`, `344005` | vLLM, SGLang and TensorRT-LLM Llama B1/B16 repeats |
| `343998` | Five guarded A800 MMA mainloop calibration processes |
| `344000` | Official grouped-mm TorchCompile OLMoE B1 pilot |
| `344007` | SM80 OLMoE B1 prompt prefill and full correctness pilot |
| `344029` | Qwen3-0.6B B1 separate prefill and native decode correctness |
| `344037`, `344039`, `344041` | vLLM, SGLang and TensorRT-LLM B1 first round on five additional dense models |
| `344049`, `344051`, `344053`, `344055` | Initial B1 rounds for remaining Qwen3-30B-A3B and OLMoE baselines |

After successful pipeline selection, the login-node watcher verifies the selected artifact and submits five fresh Llama B1 correctness processes followed by three independent formal timing processes. Once B1 and the linked MMA calibration pass, it collects the MMA profile and uses it for B16 selection and validation. It records unsuccessful stages and stops the dependent chain. At the last status check the ordinary queue reported `Priority` for `343965` on the main calibration node.

## Flexible A800 retries

**Verified.** Llama B1 selected-artifact job `344085` passed five fresh correctness processes and three formal timing processes. Median TPOT was 2.603 ms. OLMoE B1 job `344007` passed 1,024 Hugging Face positions and step-versus-loop checks. The live result CSVs are refreshed under `runs/a800/collected/`.

**Verified.** In each of two original native rejections the reported PID had been observed as an owned worker in 19 preceding snapshots. The transition is consistent with a stale NVML sample after process exit. The A800 guard now requeries a previously verified owner whose process entry has disappeared. Unknown or reused PIDs, persistent observations and throttle still reject the attempt. Twenty-five Linux guard and scheduler checks passed, including the retired-owner cases.

**Verified.** Slurm now chooses the node and physical card for each one-GPU job. Job `344694` ran on a different A800 with the same measured hardware resources, linked and replayed both Llama B16 candidates, and selected rank 1 in 6 minutes 6 seconds. The selected arm's tuning TPOT was 3.013 ms. Independent correctness and final timing are queued in `344725`. Both the source calibration UUID and allocated UUID are recorded. Baseline repetitions retain their own UUID and only share a result table within the same hardware class.

**Verified.** Qwen3-0.6B correctness retry `344696` passed step-versus-loop and all 1,024 teacher-forced positions, with maximum logit gap 0. The guard observed a previously verified owner during its exit, repeated the NVML query and accepted the clean result. The retry was on the new A800.

**Verified.** The TileMega environment accepted the pinned `accelerate` 1.15.0 and `psutil` 7.2.2 wheels after checksum and dependency checks. TensorRT-LLM failures on Llama occurred in its global C++ attention cache destructor after successful measurement and explicit engine shutdown. TP1 now exits through the completed child process without executing that destructor. Fresh rounds replace the rejected attempts. The installed TensorRT-LLM 0.21.0 PyTorch backend rejected the OLMoE architecture before timing. Its five batch rows carry the observed N/A evidence.

**Verified.** TensorRT-LLM retry `344698` passed the three previously rejected Llama rounds on the new A800. The three-process TPOT medians are 2.503 ms at B1 and 3.208 ms at B16. Each cell retains both physical UUIDs in its summary. The complete Llama B1 baseline comparison gives a fastest-baseline to TileMega ratio of 0.962. TensorRT-LLM dense follow-ups are `344743` and `344745`.

**Verified.** TorchCompile OLMoE retry `344700` passed on the new A800. Its one process recorded 4.999 ms TPOT, two compiled graphs, `grouped_mm` for prefill and the official `batched_mm` switch for decode. The remaining independent rounds are queued in `344765`. Qwen3-30B-A3B TorchCompile is running under `344702`. Its separate native prefill has a bounded CPU rebuild, and the login-node watcher verifies the binary before enqueuing GPU correctness.

## Parallel single-GPU campaign

The ordinary Slurm partition places each independent job on an available A800.
The strict guard records the physical GPU and device resources for every
process. Compatible A800-SXM4-80GB devices can contribute to the same cell.
Each queue process retains the frozen BF16 TP1 checkpoint, prompt IDs and
64-input/1,024-output request protocol. Six-entry light arrays run up to three
tasks, and two-entry heavy arrays run one task. The login-side feeders record
each submitted range and resume at B2, B4, B8 and B16 once the preceding batch
has valid evidence for the model and framework.

Qwen3-0.6B B1 ran four baseline retries concurrently on four nodes
(`344948`, `344950`, `344952`, `344954`). The longer parallel cohort names
exposed the Unix socket path limit in vLLM 0.10.2. The launcher now supplies
`VLLM_RPC_BASE_PATH=/tmp` and keeps compiler caches scoped to their cohorts.
The Qwen3-1.7B B1 retry `344999` completed after the fix. Failed attempts and
the exact-repeat retries remain in the corresponding cohort directories.

The native SM80 decode path uses the weighted CTG, pipeline shortlist and
replay, tmexec, SM80 CUDA generation, linked occupancy correction and measured
request-wall selection. The MMA mainloop profile is in
`runs/a800/calibration/343998/sm_80_a800_mma.json`. It combines the five guarded
MMA calibration processes with the expanded component calibration. Its SHA-256
is `03a33c6a8d297abd433fbbd985e1505982fa35d1697c36d04fca33a36a26a5e8`.
CPU-only decode emission succeeded for all 40 model and batch cells.

| Model and batch | TileMega TPOT | Fastest baseline TPOT | Baseline / TileMega |
|---|---:|---:|---:|
| Llama-3.2-1B B1 | 2.603 ms | TensorRT-LLM, 2.503 ms | 0.962× |
| Llama-3.2-1B B16 | 3.025 ms | SGLang, 2.978 ms | 0.985× |
| Qwen3-0.6B B1 | 1.658 ms | TensorRT-LLM, 2.066 ms | 1.246× |

Each TileMega row passed five fresh correctness processes and three separate
formal timing processes. B16 finished in `344725`. The Llama B16
`weight_stream` trial `344994` changed that one search decision in a copy of the
calibrated target. Its winning tuning arm measured 3.021 ms versus 3.013 ms for
the existing search winner, on separate allocations. The formal B16 artifact
and results still use the original target.

Qwen3-0.6B B1 selected a winner in `344987` after the guard rejected
`344956` for an unresolved GPU process. OLMoE B1 selected a winner in
`344958`. Their search TPOTs were 1.656 and 2.538 ms respectively.
Correctness arrays `345022` and `345005` both passed in five fresh
processes with matching token receipts and a 1,024-position teacher-forced
check. Qwen3-0.6B formal timing array `345057` completed. OLMoE formal
timing array `345073` is in progress. Llama B4 search
`344997` advanced to correctness array `345051`. Independent prefill
libraries for Llama B2 and B8 also linked. Their initial searches `345040`
and `345041` rejected compiled candidates because compiler inputs changed
during the build. Retries `345080` and `345081` use a compiler verified against
the frozen source fingerprint.

Qwen3-30B-A3B uses its separate 64-token prefill library. The uninstrumented
`344800` loaded the shared checkpoint and initialized the engine before the
first L1 prefill failed at CUDA synchronization. Peak observed GPU usage was
63,767 MiB. The Compute Sanitizer attempt `344973` stopped in expert packing
at 80,447 MiB before launching the prefill kernel. The uninstrumented L2
diagnostic `345008` exposed a separate runner defect when a one-token request
configured a zero-length decode ring. The runner now skips that unused ring.
A CPU mock verified both one-token initialization and initialization with a
decode step. L2 retry `345058` uses the same weights and artifact pair.
These attempts provide no formal timing.

The SM80 prefill seed now accepts explicit nonpaged MMA16 geometry against a
bound SM80 target while preserving shared weight packing and checking final
shared-memory and grouped expert row constraints. Llama B4's
`M32N64K64S3` prefill linked as SM80, while the oversized geometry probe was
rejected before code generation. The guarded first-token probe for the new
prefill is `345083`. Prefill GPU validation and end-to-end selection across
the model matrix are still in progress.

The committed snapshot has 25 baseline cells with three accepted replicates,
5 TensorRT-LLM OLMoE N/A cells and 130 remaining baseline cells. Three native
cells have final timing. OLMoE B1 passed the exact-artifact correctness gate
and is collecting independent timings. The live tables are under
`runs/a800/collected/`.
