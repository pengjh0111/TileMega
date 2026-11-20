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

## Initial queue pilot

The initial `gpu_a800` pilot used a serial Slurm dependency chain. Each allocation and child process had a finite deadline. The initial tail and stage IDs are in the workspace `logs/`.

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
| Llama-3.2-1B B16 | 2.864 ms | SGLang, 2.978 ms | 1.040× |
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

## Follow-up tuning and validation

**Verified.** OLMoE B1 and Llama-3.2-1B B4/B8 completed all independent
formal timing rounds. Llama B4 and B8 recorded 2.246 ms and 2.345 ms TPOT,
respectively, with fastest-baseline ratios of 1.175× and 1.195×. OLMoE B1
recorded 2.523 ms over five fresh processes. The comparison column requires
all available baseline families to finish their independent repetitions.

**Verified.** Slurm job `345176` paired the original and
`M32N64K64S3` Llama B4 prefill libraries on one A800 with the same
solver-selected decode. The order was control–trial, trial–control,
control–trial. Both arms used BF16 TP1, the frozen input, one warmup and
three N=1 and N=1024 timed requests per process. All six generated-token
tables and checkpoint hashes matched. Median request-wall first-token
latency decreased from 21.591 to 19.502 ms. TPOT moved from 2.228 to
2.226 ms and N=1024 wall time from 2.301 to 2.297 s. The three paired
N=1024 ratios were 1.00157×, 1.00175× and 1.00156×. The new prefill
passed all five fresh validation processes in `345213` on four A800s.
Three independent candidate timings in `345290` confirmed 19.522 ms first
token latency versus 21.618 ms for the original pair. TPOT was 2.253 ms
versus the original 2.246 ms. The formal table retains the original
prefill because it has the lower independent TPOT. Raw paired evidence is in
`runs/a800/optimization/prefill-pairs/345176/`.

**Verified.** Llama B16 `layer_loop=on` completed measured selection in
`345185` with 3.017 ms tuning TPOT. The reference tuning TPOT was 3.013 ms
on a separate allocation. The reference configuration remains selected.

**Verified.** Llama B2 search `345080` was rejected for a GPU process
without verified ownership after both cubins had linked. The strict
occupancy helper now remains visible to the guard before initializing CUDA
and briefly after context teardown. Search retry `345167` passed the full
guard, linked-occupancy correction and measured selection at 2.423 ms
tuning TPOT. Five independent correctness processes passed in `345205`.
Three formal processes in `345240` gave 2.431 ms TPOT against
TensorRT-LLM at 2.509 ms (1.032×). The earlier rejection remains
recorded with unresolved process ownership.

**Verified.** The full Qwen3-30B-A3B L2 retry `345058` reached the first
prefill launch and returned the same CUDA launch failure class as L1.
The reduced diagnostic keeps the original hidden width, expert count,
routing top-k and frozen prompt. It reduces the layer count to two to
fit Compute Sanitizer and the native executor's layer grouping. The
reduced diagnostic is outside the formal benchmark matrix.


**Verified.** Qwen3-1.7B B1 and Llama-3.2-3B B1 completed measured
selection in `345191` and `345204`, respectively. OLMoE B16 completed
selection in `345194`. Their independent validation jobs are `345248`,
`345257` and `345300`. Llama B1 recalibration in `345193` screened at
2.603 ms TPOT, matching the original selection at the reported precision.

**Verified.** The guarded B16 decode trace `345279` used the same selected
geometry, matched reference tokens, and retained 16 decode steps near
the midpoint. FFN gate/up and down accounted for 1,030 µs and 783 µs per
step. Down had the largest stage-level gap to the 655 µs prediction.
The follow-up compared two versus three `cp.async` slots in the down
projection on the same GPU. The raw attribution and trace are in
`runs/a800/optimization/decode-traces/345279/`.

**Verified.** The two-layer Qwen3-30B diagnostic `345265` reproduced CUDA
error 719 at its first L1 prefill without exhausting device memory.
Compute Sanitizer reported a hardware exception that it could not
classify. Its subsequent CUDA free errors followed the first launch
failure. A CUDA-GDB probe of the same pair is `345324`.


**Verified.** The same-card Llama B16 paired trial `345345` changed only the
FFN down projection from two to three `cp.async` slots. All six runs used
the same packed weights, prefill library and prompt, and emitted the same
output tokens. Median TPOT decreased from 3.011 to 2.939 ms (2.40%) and
1,024-token request-wall latency from 3.150 to 3.076 s. All three pairs
favored the three-slot configuration. The raw request and guard evidence
is in `runs/a800/optimization/decode-pairs/345345/`. Job `345393` pins this
measured GEMM geometry in pipeline search, including shortlist replay and
linked occupancy checks. The pinned search selected rank 0 at 2.943 ms tuning TPOT. The fresh
correctness array `345417` passed all five processes on three A800s. Its
teacher-forced check covered 16,384 positions at a maximum logit gap of
0.03125. Independent timing allocation `345450` uses the same artifact
pair and separate candidate receipts.

**Verified.** The SM80 pipeline derivation previously reset the requested
ring depth to the first feasible depth for a tile. The regression in
`e61cbac` exercises a legal three-slot coordinate and fails with the old
implementation, which returned two slots. The corrected implementation
passes. The isolated worktree `TileMega-a800-ring-depth` retains the tested
compiler and source. Unconstrained search `345343` increased the number
of scored B16 placements from 101 to 1,204 and selected multiple eight-slot
rings. Its measured winner at 3.284 ms lost to the original 3.013 ms
selection. The experiment provides evidence for processing measured depths
as search coordinates while retaining empirical selection.

**Verified.** Qwen3-30B's reduced prefill failed at a deliberate device trap.
CUDA-GDB job `345346` identified the `BPT.TRAP` following a non-null deferred
normalization statistics output in the 16-column attention output GEMM.
The packed prefill epilogue requires 32 columns to write one statistics
block. The reduced repair widens attention output packing to 32 columns,
keeps the router at 16 columns and regenerates the decode and prefill ABI
from the same packed recipe. Full-shape results require the original Qwen
configuration and shared checkpoint.

**Verified.** OLMoE B16 passed all five new correctness processes in
`345300` and all five formal timing processes in `345363`. Its median TPOT
was 9.213 ms and N=1,024 request time was 9.968 s. The speedup is pending
completion of the relevant baselines.


**Verified.** Qwen3-0.6B B2 and B8 selected 1.620 and 1.880 ms tuning
TPOT in `345395` and `345394`. The independent correctness arrays
`345424` and `345423` each passed five processes. Mistral-7B B1 exceeded
the initial stage deadline during selection. Bounded retry `345388`
completed all four tuning trials and selected 11.731 ms. These tuning
medians remain separate from the independent timings.


**Verified.** The three independent Llama B16 timings in `345450` passed on
two A800s. The down-only artifact has 2.929 ms median TPOT, 69.799 ms T1 and
3.0661 s T1024, with median request throughput of 5,343.6 tokens/s.
The earlier independent cohort recorded 3.025 ms TPOT and 3.1642 s T1024.
The same-card experiment measured a 2.4% TPOT reduction. The independent
cohort measured a 3.2% reduction and a 1.017× ratio against SGLang at
2.978 ms. The collector accepts an explicit artifact identity in
`configs/bench/a800_native_selection.json`. The earlier formal process
receipts remain available. This selection keeps five fresh correctness
receipts and three complete independent request timings for the exact
prefill, decode and target hashes.

**Verified.** The corrected two-layer Qwen3-30B-A3B first-token probe
`345437` completed under the strict GPU guard in 69 seconds of allocation
time. This exercises the original expert geometry and the repaired
32-column prefill output packing. The launcher started the original
full-shape CPU build under `runs/a800/build/prefill/qwen30b-o32-b1/`.
The full-shape artifact linked, and `345624` passed the first-token
request using the original config, the exact checkpoint shared by the
baselines and a strict guard. Peak observed use was 63,753 MiB. This
first-token diagnostic returned 283.4 ms T1. Pipeline search `345665`
used the original shape and corrected shared weight layout. It completed
four full tuning processes in 31 minutes 1 second, and chose 9.577 ms
TPOT. The five independent correctness processes are queued in `345794`.
The separate Qwen decode trace starts after that array passes. CPU
regression for `a8621ee` emitted the old 16-column configuration with the
original compiler, rejected it with the new check, and emitted the corrected
32-column configuration.

**Verified.** Llama-3.1-8B B1 bounded retry `345389` completed in 16 minutes
26 seconds after the original selection timeout. Its chosen tuning TPOT
was 12.189 ms. The extended Mistral and Llama searches both used one A800
and retained the original protocol. The two model lanes resume larger
batches after B1 independent timing completes.

**Verified.** Qwen3-0.6B B2 and B8 each passed three fresh formal timing
processes. Median TPOT was 1.621 ms at B2 and 1.883 ms at B8. B2 measured
1.699× against the fastest baseline. The B8 comparison awaits the remaining
baselines. B4 and B16 have independent prefill artifacts and selection jobs.

**Verified.** The B16 follow-up `345466` compared two and three `cp.async`
slots in FFN gate/up with FFN down held at three slots. The three paired
requests used the same A800, checkpoint, prefill and generated tokens.
Median TPOT decreased from 2.942 to 2.861 ms. Bounded pipeline search
`345503` selected this gate/up geometry at 2.868 ms tuning TPOT. Array
`345543` passed five correctness processes on four A800s, including the
16,384-position teacher-forced check at 0.03125 maximum logit gap.
The three independent timings in `345601` passed at 2.864 ms median TPOT,
69.922 ms T1 and 2.9991 s T1024 with 5,463.0 output tokens/s. The final
TPOT is 5.3% lower than the initial 3.025 ms and 1.040× against SGLang
at 2.978 ms. `configs/bench/a800_native_selection.json` pins this exact
artifact pair. The earlier timing and tuning cohorts remain reproducible.

**Verified.** OLMoE B2 search `345449` reached code generation and rejected
both candidates for inconsistent expert-stage packing. The selected gate/up
tile uses a per-task 128-by-64 slice. Its down tile is also a per-task slice
with 64-by-64 dimensions, which the original code mistook for Qwen3's shared
fixed-page recipe. CPU code generation reproduced the rejection. A
three-slot 64-by-128 down slice fits the fused arena and emitted SM80 CUDA.
Job `345505` resumed B2 with this legal geometry and selected 3.370 ms
in four tuning processes. All five independent correctness rounds passed in `345615`. Formal timing is `345668`. The layout fix in
`b79f9a8` built in `TileMega-a800-moe-pack-build` and passed the compiler
fingerprint check. The exact OLMoE geometry that was previously rejected
emits a per-task slice, and Qwen3 still emits shared 64-by-64 pages. OLMoE
B4 completed measured selection in `345692` with this compiler.

**Verified.** Llama-3.2-1B B1 trace `345633` completed with equivalent
reference tokens. The FFN down stage accounted for 849.9 µs per step
against a 391.0 µs prediction. The current tile uses N8 K1024 and has up to
three residual tasks per CTA. The follow-up tests N16 K512 and three slots
with the original checkpoint and prefill in a single-card paired run.

**Verified.** Mistral-7B B1 and Llama-3.1-8B B1 finished five
fresh correctness processes and five independent timing processes each.
Mistral measured 11.764 ms TPOT in `345568`. Llama
measured 12.233 ms TPOT in `345608`. Their B2 lane builds
separate packed prefill libraries from the same export and checkpoint.

**Verified.** Llama B1 paired job `345681` completed all six requests on
one A800 with a common checkpoint. The candidate and reference diverged
at output token 88, with 23 different positions over 1,024 tokens. The
paired protocol rejected the comparison. Separate correctness array
`345753` passed five processes for the candidate. Its own step and loop
outputs agreed and the teacher-forced maximum logit gap was 0.03125
across 1,024 positions. Pipeline selection and fresh timings remain
pending. A second B1 trial varies only the depth of the original down
tile's `cp.async` ring.

**Verified.** The search planner records its deadline and heartbeat
limit. The strict GPU guard bounds silent searches to at most nine
minutes within the Slurm time limit. The ten CPU guard tests passed.

## Result snapshot

At 2026-11-20T15:43:14+08:00, 23 of 40 TileMega cells and 66 of 160 baseline cells had complete independent timings. TensorRT-LLM rejected the OLMoE architecture for all five batch sizes. The timestamped table includes partial process counts when available. The server publishes subsequent accepted results in `runs/a800/collected/`. [Request times](A800_execution_times.md)

For Llama-3.2-1B B1 the selected solver artifact took 2.467 seconds for 1024 tokens and 2.404 ms per output interval. All five independent correctness checks passed, including the full HF check. The earlier paired trial had 23 token differences across implementations. Its failure stays in the experimental record. [SM80 decode analysis](A800_SM80_analysis.md)
