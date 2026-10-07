# Forward phase core

verified: commit `1de42e239` adds phase 2 without changing ABI version 1.
Forward plans have zero KV capacity/past, accept only step 0, and retain
L1 and L2 execution. The default phase inference remains unchanged.

verified: forward batch and token axes are distinct. Dense GEMM may use
batch × rows_per_batch; independent MoE regions bind batch=1 and seq=T.
Forward plans reject decoder attention, RoPE, KV append, token embedding,
argmax and buffers that depend on past/total. Encoder attention has its own
appended task kind. Unused decoder attention is not instantiated for phase 2.

verified: `results/CI3_forward_core.json` joins four generated libraries to
complete source/compiler/CUDA/binary/macro/resource identities. Numerical
checks use independent PyTorch FP32 linear references from BF16 inputs, the
unchanged 1.6e-2 absolute/relative tolerance, and same-binary L1/L2 bitwise
comparison: B=1/8 and token-axis T=16/128/4096 pass on sm_89. These are dense
GEMM probes, not upstream DNN or MoE model gates. All four libraries report
zero ptxas spill stores/loads.

verified: three CPU ABI tests pass. The targeted frontend/semantic tests pass
in `runs/dm1-ci3-forward-final/events/`. The architecture/fresh-process job
was deliberately terminated when CI-4 edited transitive host headers during
compilation; generated libraries had already completed. Its incomplete
results are not a passing gate. The repeat checker now snapshots transitive
NVCC dependencies and compiles host support objects once.

verified: the sealed repeat compiles seq=1/128/4096 for all five targets
sm_80/89/90/100/120. On sm_89, each seq passes 50/50 fresh processes (150/150
total), including B=1/8 and L1/L2 bitwise equality. All fifteen builds have
zero spill stores/loads. Complete transitive input hashes remain unchanged
(`results/CI3_forward_architecture.json`). This is a synthetic forward/ABI
check, not a DNN/MoE model or §8.A synchronization gate.

Pending: forward CLI model import; DNN and MoE states/weights; page execution
for forward/prefill under CI-5; complete G-REG at the Phase 1 checkpoint.
