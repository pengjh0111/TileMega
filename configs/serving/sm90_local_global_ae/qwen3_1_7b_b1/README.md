# Operator-local × planner experiment

Status: implementation in progress. No A–E TPOT has been measured on this branch.
Base: `sm90-explicit-stage-ae-v1`, `a63def29570f1b28ab0a3b8a29e269ce78562cd5`.
Historical results remain unchanged and are not new-arm results.

## Local boundaries fixed before measurement

The export describes RMS normalization, grouped Q/K/V projections, Q/K
normalization and rotary position embedding, attention, output projection and
residual, normalized gate/up projections, SwiGLU, down projection and residual,
final normalization, vocabulary projection/reduction and token embedding.
The following standalone units define this experiment's local structure.

| Unit | Input-ready/output-complete contract | Selection freedom |
|---|---|---|
| RMS statistics, before QKV/GateUp/head | FP32 inverse RMS; gamma remains folded in packed weights; original GEMM output scaling before BF16 rounding | Common implementation; WK-compatible summation |
| QKV grouped linear | Raw residual + inverse RMS → packed BF16 Q/K/V | TN/TK/WN/WK/slots/register stages |
| Q/K preparation | Q/K normalization, RoPE, one cache append → prepared Q/K/V | Common original body preparation |
| Attention | Prepared Q/K/V and complete cache → complete BF16 context, including all partial/LSE reduction | chunks/block_keys/slots; direct, LA or separate internal merge |
| O linear | Complete context → BF16 projected output | TN/TK/WN/WK/slots/register stages |
| O residual | Projected output + residual → BF16 residual stream | Common implementation |
| Gate/Up grouped linear | Raw residual + inverse RMS → packed BF16 gate and up | TN/TK/WN/WK/slots/register stages |
| SwiGLU | Packed gate/up → complete BF16 activation | Common implementation, packing U=8 |
| Down linear | Complete activation → BF16 output | TN/TK/WN/WK/slots/register stages |
| Down residual | Projected output + residual → BF16 residual stream | Common implementation |
| Head with bounded argmax output | Residual + inverse RMS → final token; tile partial and final reduction included | TN/TK/WN/WK/slots/register stages |
| Embedding | Final token → next residual input | Common implementation; separate from head |

Shared frontend linear grouping and gamma folding are explicit compatibility
choices. Norm statistics remain separate; inverse scaling stays at the original
linear output rounding point. Head reduction is an internal completion of the
vocabulary projection: no complete vocabulary tensor is materialized. Neither
residual nor SwiGLU nor embedding is fused into local GEMM candidates.
Q/K preparation is a separate, untuned unit. Attention timing starts with its
prepared inputs, includes its entire reduction, and does not include preparation.

## Candidate and timing protocol (predeclared)

`local_domain.json` records every candidate and preselection, before GPU timing.
The finite GEMM domain is TN={16,32,48,64,128,256}, TK={64,256,512,1024},
WN={1,2,4}, WK=4/WN, slots=2..8, register stages={1,2,4}. Divisibility,
body constraints and the full 232448-byte per-CTA shared-memory limit apply.
No old half-arena or task-count-window restriction is imposed. The same domain
is applied to each shape; D gets no special insertion or tie preference.

Attention uses chunks={1,2,4,8,16,32,64}, BK={64,128}, slots={2,3,4} and
both LA and separate merge for split cases. S1 has no redundant merge.
TP1/cluster1 excludes DSMEM/multicast implementations. All candidates use
132 tracks and normal MMA/TMA producer-consumer pipelines.

Two compatible implementations per (family,TN,TK) or (chunks,BK,merge) survive
local calibrated-model preselection, ties broken by complete candidate identity
hash. The screening model uses existing calibrated bandwidth/latency and
epilogue terms; the old fused epilogue is only a proxy for the new standalone
linear output. Missing epilogue TN and S64 merge constants are explicitly
extrapolated for screening. No decomposition is removed due to tied predictions.
Final selection must use measured complete-operator medians, never these scores.

The lightweight harness instantiates native task bodies, not full megakernels.
Weights and inputs use deterministic finite BF16 values, prepared before timing;
the inverse-RMS input is 1. Weights are already in candidate-native packed order.
Every sample first evicts L2 with 256 MiB of cache-global stores. Eviction,
initialization, weight packing and graph capture are outside the event interval.
Each complete-operator CUDA Graph gets 3 warmups and 15 timed samples. Attention
uses five predeclared equal decode intervals, weighted by step counts, covering
past=64..1086. The selected structure will be frozen before whole-network timing.

Evidence: boundaries and protocol are stated experimental choices; candidate
enumeration and later compilation/measurement receipts distinguish verified
results. Component measurements alone are not A–E acceptance.
