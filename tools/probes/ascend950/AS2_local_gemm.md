# Single-AIV local GEMM transport probe

Status: **compiled, not device-verified**. No model implementation/default is
enabled. `LocalGemmTile.h` composes the pinned CATLASS Block/Tile copy with
NO_SPLIT output explicitly selecting AIV0. The caller supplies the output
slot and solved tile coordinate; no whole-operator scheduler is imported.

The initial contract is BF16 NK weights/activations, FP32 accumulation and
BF16 Fixpipe output. Logical tails remain unchanged; physical output columns
are padded to32 bytes inside a16×128 slot. The probe compares this result
against the ordinary GM tile bitwise, including large K and N129. It uses
exactly one group/one AIV participant, not a two-slot pipeline or dual-AIV
epilogue. The fixed slot is131584B, with512B guards before/after its4096B
payload, separate from the model vector body's106496B scratch.

Control deliberately retains conservative GM ready/consumed frontiers:
AIV poisons and pre-reads the old slot → unique AIV publisher advertises it
free → AIC computes, completes FIX_S and publishes ready → AIV copies the
valid output and completes MTE3_S → AIV acknowledges reuse. Hardware
CrossCore flags are not used. This local payload visibility hypothesis
requires its own real litmus; GM-control correctness alone does not prove it.
Every wait checks abort and a system-counter deadline. The10e9 tick bound
uses the retained1GHz AS-2 counter evidence (`runs/as2/jobs/timer_differential`);
the wrapper must also impose a finite Host timeout and track only this job.

`verified-run` compile only: CANN9.1.0/BiSheng15.0.5 builds
`tilemega-as2/build/taskbody_local_gemm_20261102_v1/as2_local_gemm`.
Image SHA256 `b335afb52db77bbd4a66449428b2b5209557279268f0a8b60a6457eae2194e97`.
The first compile fails on a BF16 Scalar equality instruction; raw uint16
guard comparison fixes it. The original source/header/command/error are
protected in `runs/as2/failures/local_gemm_bf16_scalar_compile_20261102_v1/`.

CLI: `as2_local_gemm M N K inputs new_output`. Inputs `a.bin` contain four
independent BF16 epochs [4,M,K]; `b.bin` is BF16 [N,K]. All source/output
paths must be inside `runs/as2`. The Host checks a single visible950PR,
preserves output/controls before numerical failure, and logs init/launch/
launch-return/stream-completion. It performs no device reset.

Prepared cases/reference/tolerance identities:
`runs/as2/probes/local_gemm_20261102_v1/cases.json`. Device tests are pending
the unrelated G8 timing-timeout investigation. Required admission is five
fresh processes covering M1/M16, regular/tail, large K and buffer reuse;
then real model binding/correctness/prices/full-request comparison. No local
route is silently substituted into the current generated runtime.
