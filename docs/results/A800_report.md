# TileMega single-GPU A800 experiment

## Revisions and environment

**Verified.** The paper audit base is `7425a5242580af16a7ea43be742e313a6eedb351`. The inherited experimental parent is `cc62bb703de5e8f4f0b03b21c4e58f04ef4ce0b3`. The A800 branch is independent. CUTLASS in the inherited worktree matches the recorded gitlink `dc45f979ae336a235da1676b311f35efeb30149a`.

**Verified.** The Paracloud `n46h` deployment uses Slurm 24.11.6 and Apptainer 1.4.5. Login-node checks confirmed CUDA compiler 12.8.93, Codex CLI 0.170.0, the HTTPS proxy, and an empty CUDA mask outside Slurm. The workspace launcher injects the driver through Apptainer only in a GPU allocation. Slurm assigns memory from its per-GPU policy.

## A800 evidence

**Verified.** Device probe `343820` completed with strict host process ownership, 31 samples at one-second intervals, and CUDA/NVML identity agreement. The allocated A800-SXM4-80GB has compute capability 8.0, 108 SMs, 40 MiB L2, 65,536 registers per SM, 166,912 bytes of opt-in shared memory per CTA, and MIG disabled. The NVIDIA driver was 580.105.08. The raw receipt and uncalibrated target are in `runs/a800/probes/343820/`.

**Verified.** A later Slurm allocation on the same node exposed a different UUID. The pinned calibration gate rejected it in one second. New diagnostic cohorts record their device within the same allocation. The result collector requires the exact cohort UUID.

**Verified.** Bounded job `343865` ran the first SM80 native transport calibration process and completed in 76 seconds, including both guarded admission windows. Diagnostic medians include 1,658.28 GB/s for the tested HBM transfer, 2,218.77 GB/s for the tested L2-resident transfer, and 12.8805 GB/s for the tested per-CTA swizzled activation copy. These are single-process component measurements. The command, binary hash, and receipts are in `runs/a800/calibration/343865/`.

**Verified.** The first full calibration attempt `343873` reached the MoE epilogue and caught an illegal memory access. The older calibration fixture omitted the current shared routing state. The fixture now supplies the routing layout consumed by the production epilogue and MoE body.

## Measurement protocol

The baseline matrix contains 160 cells, each requiring three independently accepted processes. Each process uses one of the frozen prompt batches, TP1, BF16 weights and KV, and a shared seed-zero checkpoint. N=1 and N=1024 each use one warmup and three formal request-wall repetitions. Process TPOT is `(median(T1024) - median(T1)) / 1023`. The final cell uses the median of process TPOTs.

The native pilot retains separate Llama-3.2-1B prefill and decode artifacts at B1 and B16. Both pairs passed A800 login-node artifact identity, architecture and batch checks. Their full 1,024-token teacher-forced and step-versus-loop execution checks require GPU runs. Formal timing requires a measured solver selection and five fresh correctness receipts for the exact artifact pair. The final native matrix has 40 cells with three or five independent timing processes per cell.

Pipeline selection uses the guarded linked-cubin occupancy probe and mirrored end-to-end shortlist measurements. Formal timing uses fresh processes. The A800 target requires its own measured constants, and CPU build evidence from A100 is retained under its original run directory.

## Status

Framework migration and the first baseline pilot are pending. The raw per-attempt output, Slurm receipts and thermal telemetry are under `runs/a800/` and the workspace `logs/`. The two result CSVs are generated from accepted attempts. At this point speedups have no end-to-end evidence.
