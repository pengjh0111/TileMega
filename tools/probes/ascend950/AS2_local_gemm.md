# Single-AIV local GEMM transport probe

Status: **component verified at G2/G8, 5/5 fresh processes each**. No model implementation/default is
enabled. `LocalGemmTile.h` composes the pinned CATLASS Block/Tile copy with
NO_SPLIT output explicitly selecting AIV0. The caller supplies the output
slot and solved tile coordinate; no whole-operator scheduler is imported.

The initial contract is BF16 NK weights/activations, FP32 accumulation and
BF16 Fixpipe output. Logical tails remain unchanged; physical output columns
are padded to32 bytes inside a16×128 slot. The probe compares this result
against the ordinary GM tile bitwise, including large K and N129. It uses
G independent groups/one AIV payload participant per group, not a two-slot
pipeline or dual-AIV epilogue. AIV1 independently poisons and exports its own
private slot, proving that AIV0 output does not overwrite it. The fixed slot
is131584B, with512B guards before/after its4096B
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

`verified-run`: physical1/55:00.0, CANN9.1.0/BiSheng15.0.5, pinned CATLASS
2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52. The v5 selected component image is
`tilemega-as2/build/taskbody_local_gemm_witness_20261102_v5/as2_local_gemm`.
SHA256 `8d1df886506910d4aca10b8cc9bb1e78c6f626069aec39aac3c40b5c3ab6bd71`.
The first compile fails on a BF16 Scalar equality instruction; raw uint16
guard comparison fixes it. The original source/header/command/error are
protected in `runs/as2/failures/local_gemm_bf16_scalar_compile_20261102_v1/`.

CLI: `as2_local_gemm M N K inputs new_output [G]`, G in1/2/8. Inputs `a.bin`
contain four independent BF16 epochs [G,4,M,K]; `b.bin` is BF16 [N,K]. All source/output
paths must be inside `runs/as2`. The Host checks a single visible950PR,
preserves output/controls before numerical failure, and logs init/launch/
launch-return/stream-completion. It performs no device reset.

Each G2/G8 cohort covers M1/M16, N128/N129, K8192/K14336 and four reuse
epochs, with independently initialized groups and a repeated case. GM/local
outputs are bitwise equal, independent FP32 reference checks pass the frozen
BF16 policy, guards/padding remain intact, and AIV1 witness bytes are unchanged.
The N129 GM reference uses a legal physical stride144; padded weights/columns
are excluded from the logical numerical comparison. Host and per-lane clocks
are diagnostic component measurements, not model latency or TPOT evidence.
Evidence: `runs/as2/probes/local_gemm_witness_g2_20261102_v5/` and
`runs/as2/probes/local_gemm_witness_g8_20261102_v6/`, including frozen sources,
images, independent references, five process receipts and summaries.

The first v1 G1 cohort stops after an M16 Host timeout60s, with launch-return
but no original stream-completion evidence; it is not a verified kernel
deadlock. Failure source/image/logs remain under
`runs/as2/failures/local_gemm_timeout_20261102_p1_v1/`. Later marked G1 M16
recovery passes (Host launch-to-sync0.782273ms), followed by the separate v5
G2/G8 cohorts above. These do not relabel the original timeout or enable a
model default. No hardware CrossCore flag protocol is admitted by this probe.
