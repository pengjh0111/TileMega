# Production Fusion and Two-Edge Calibration

## Scope and Evidence

✅ `selected_runtime_second/` contains 400/400 fresh BF16 processes: gqa2 and
mha4, seq 4/128, past 3, separate/DP-selected RoPE-KV states, 50 each. All
L0.5/L1/L2 comparisons pass; state and level output hashes are identical.
The 200 fused processes independently match symbolic task/wait counts
(400/400 fields). All have 212 L2 registers, zero spill, 24576 B TaskSmem,
2 CTA/SM and grid256. Sources, selected CG, ptxas, hashes and logs are retained.

✅ `chain_runtime_first/` contains another 400/400 fresh BF16 processes:
GEMM/add and GEMM/RMSNorm chains, seq4/128, separate/fused, 50 each. These are
genuine two-stage calibration subgraphs, not full decoder results. N=K=512
comes from the exported model projection; tile M=32 is an explicit experiment
choice. N tile is128 for add and the complete512 for RMSNorm. Inputs are
nonconstant fixed-seed random BF16 tensors; PyTorch supplies the reference.
Original tolerance is unchanged. Every state and execution level is bit equal.

Both runners use warmup5/repeat11 and rotate the complete state/case product.
GPU execution did not overlap: the chain runner was paused during compilation
until the full-model matrix completed. CPU compilation/tests ran concurrently.
Primary analysis uses the first25 paired rounds, 20000 bootstrap draws and
tie-corrected normal-approximation Wilcoxon. JSON includes50-round sensitivity.
`summarize_runtime.py` checks original logs against receipts before statistics.

## Full-Model Results

Times below are process medians in ms; ratios are medians of within-round
ratios, not quotients of the two marginal medians.

| Model / seq | L2 separate | L2 fused | Paired ratio | 95% paired CI | Wilcoxon p | Predicted event delta ns |
|---|---:|---:|---:|---|---:|---:|
| gqa2 /4 | .405664 | .400608 | .984460 | [.981651,.987356] | .000227 | -5747.6073 |
| gqa2 /128 | .567584 | .566336 | .996509 | [.992941,1.001789] | .203438 | -6915.5073 |
| mha4 /4 | .809984 | .800768 | .989873 | [.985955,.993568] | .000119 | -11570.5630 |
| mha4 /128 | 1.153024 | 1.145856 | .995548 | [.991205,.997162] | .028292 | -16242.1628 |

gqa2/128 has no significant benefit. mha4/128 loses Wilcoxon significance in
the50-round sensitivity (p=.1781), so the result is not robust across both
analyses. These are RoPE/KV results, not substitutes for the requested GEMM edges.

## GEMM Chain Results

| Edge / seq | L0.5 separate/fused | L1 separate/fused | L2 separate/fused | L2 paired ratio | 95% CI | p |
|---|---|---|---|---:|---|---:|
| add /4 | .028672/.020224 | .030720/.021504 | .028864/.020480 | .705882 | [.682234,.739030] | 1.30e-5 |
| add /128 | .035840/.023552 | .038240/.024576 | .035936/.023552 | .655387 | [.633690,.657143] | 1.26e-5 |
| RMS /4 | .030848/.033792 | .031776/.035744 | .032800/.035584 | 1.062500 | [1.049906,1.081325] | 7.18e-5 |
| RMS /128 | .033792/.047840 | .034496/.048160 | .034816/.048096 | 1.352113 | [1.323529,1.400182] | 1.29e-5 |

| State | L2 registers | Spill stores/loads | TaskSmem | CTA/SM | Grid |
|---|---:|---|---:|---:|---:|
| add separate |112|0/0|16384|4|512|
| add fused |148|0/0|23552|3|384|
| RMS separate |218|0/0|52224|1|128|
| RMS fused |232|0/0|84992|1|128|

All E2E_SCHEDULE fields are retained per process. Each fusion removes one
stage and its internal coupling. Add refs8→4 and32→16, waits0→0;
RMS refs5→4 and132→128, waits3→0 and127→0. `audit_runtime_prices.py` compares
task/wait/register/shared/residency against GPU:2000/2000 fields match.

## Local Stop: RMS Price Direction

✅ Add's prediction and measured direction agree. RMS prediction does not:
seq4 predicted total28330.53→24669.00ns, seq12828927.03→25245.35ns, whereas
both measurements worsen. B1.4's local stop is triggered; joint fusion search
cannot be called an accepted optimizer using this price. No coefficients were
retuned and no acceptance threshold was changed.

The four cost components are distinguishable:

- Tile legality: full N=512 is enforced by C and lowering, not partial-row work.
- Residency: compiled232/218 registers,84992/52224B, both1CTA/SM match exactly.
- Recompute cardinality: seq4 has1 producer→4 executions (3 extra); seq128 has
  4→128 (124 extra). Global traffic grows544768→2121728 and2752512→71565312B.
- Wave pricing: `PriceFusionTasks` computes each active wave using max task
  latency and `max(1,active/num_sms)`. Both producer and consumer counts fit
  one128-worker wave. Thus the124 extra GEMMs barely change predicted fused
  task time (21797.29→21854.75ns). `recompute_ns=2605669.26` is aggregate work,
  not additive wall time, and cannot simply be added to force a losing result.

⚠️ The evidence locates the model gap in recomputation combined with wave and
memory-service pricing, not missing tasks or a residency miscount. Whether
the runtime bottleneck is shared epilogue service, global traffic contention,
or another collective execution effect requires profiling; no specific GPU
mechanism is claimed proven. A prior commentary attributing it to queue
imbalance was too strong and is corrected here.

## Code and Remaining Scope

`lib/Codegen/Codegen.cpp:727` (`LowerFusedRuntime`),
`lib/Solver/FusedRuntimeProjection.cpp:14` (`ProjectWrittenFusionQueues`), and
`lib/Codegen/ExactRuntimeTaskGraph.cpp:10` (`MaterializeExactRuntimeTaskGraph`) connect
verified CG to physical queues. `ModelHarness.cuh` consumes exact fine-event
predecessors. `FusedGemmTaskBody.h` and `FusedRoPEKVTaskBody.h` execute shared
intermediates. `lib/Solver/FusionIntervalDP.cpp:17` chooses intervals inside pinned residency;
full-pattern ptxas evidence is optional but mandatory in this chain audit.

Joint tile/split/chunk/fusion search is still unfinished and now depends on
the failed RMS pricing gate. The production residual epilogue remains the
default. Explicit add stages exist behind an import option; multi-incoming
fusion is still rejected. These chains do not claim a new residual fusion
speedup in the existing decoder epilogue.

sm120 scripts remain unrun. Portable CTest44/44 and policy pass; the final
shared-floor change passes targeted fusion/access, chain-DP and five-target
audit tests3/3. Historical failed
compilation and pre-allocation resource estimates remain in the record.
