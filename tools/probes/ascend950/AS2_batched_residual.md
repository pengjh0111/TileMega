# Batched residual epilogue

`verified-code`: `ResidualTile.h` borrows the existing worker UB arrays. Two
strided BF16 loads, FP32 Add, BF16 round-to-nearest output and a strided store
replace up to16 row-wise calls. MTE3 completes before returning. GM output,
ready/done/ack and the external owner prefix are unchanged. It retains raw
GEMM output and does not fuse a partial-row Norm or introduce a local channel.

The component accepts1–16 rows,1–128 columns and separate16-element-aligned
strides up to16384. Tail padding is local and never exported. Scratch uses
three BF16 and three FP32 arrays of2048 elements,36864B borrowed from the
existing106496B reservation; no extra TPipe or flag allocator is constructed.

The model candidate is deliberately narrower: Qwen3-0.6B/B16 decode/G8,
56 unsplit O/down projections, M16/N1024/K2048 or3072, eight16x128 tasks/stage.
GEMM variant12, capability `TILEMEGA_LLM_BATCHED_RESIDUAL`, explicit checked-in
binding and independent implementation ID preserve the replay precision
contract. Old images, missing operands, split reductions and unregistered
shapes reject before launch. Prices must come from this new implementation;
row-wise prices are not interchangeable. Default remains off.

Component preparation (CPU only, no ACL initialization):

```bash
source scripts/as2_cpu_env.sh
python tools/probes/ascend950/as2_residual_inputs.py --out runs/as2/probes/NEW_INPUTS
```

Build target `as2_batched_residual` in the existing Ascend probe project,
nice10/parallel1. Run only through `tilemega.bench.as2_run`, one healthy
physical device visible as logical0, a new output and finite120s timeout:

```text
as2_batched_residual INPUTS NEW_OUTPUT
```

Four cases include real M1/M16, 97/17-column tails, distinct left/right/output
strides, cancellation, rounding and signed zero. Four AIV workers have distinct
inputs. A single MIX launch/case compares row-wise and batched recipes over
four epochs, alternating order, then compares all BF16 bits and untouched
guard holes with an independent CPU reference. Five fresh processes are
required for device admission. Component cycle samples are diagnostic only.

Current device execution/model numerics/C3/pricing/TPOT: **not run**. On
2026-11-04 both950PR devices report Alarm; device jobs remain blocked-health.
The native CTG bootstrap and CANN compilation do not establish numerical or
performance admission. Preserve frozen tolerances, teacher512, C2/C3 and
complete64/1024 request checks before enabling this candidate.
