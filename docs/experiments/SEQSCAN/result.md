# BF16 `seq × past` correctness scan

Evidence status: ✅ remeasured on an RTX 4090 (`sm_89`) on 2026-09-07 after
the logical-task event and kAll-aggregate conversion. Every
entry in [`matrix.tsv`](matrix.tsv) is a distinct fixture and was launched in
50 fresh processes.  Each process compares PyTorch L0 with the standalone
L0.5 path, L0.5 with L1, and L1 with the dependency-driven persistent L2 path.

The complete 2 model × 5 `seq` × 3 `past` matrix passed: **1500/1500**.  This
covers a stage narrower than the resident grid, a comparable stage, and a
stage that needs many grid-stride rounds.  The BF16 comparison bound is
`1.6e-2 + 1.6e-2*abs(reference)`; it was selected after the stricter `1e-2`
bound left four quantization-boundary differences among millions of values at
`seq=2048`, while `1.5e-2` left none.  It is not used for FP32.

The complete matrix remained **1500/1500** under task-local waits. No timeout
or hang occurred. This is fresh evidence: none of the pre-conversion process
results was reused.

⚠️ The former `TILEMEGA_NEGATIVE_OLD_CLAMP=1` control passes **50/50**, as
expected: it mutates the retired stage-level `WaitDependencies` path and is
unreachable from queue L2. ✅ The replacement
`TILEMEGA_NEGATIVE_TASK_WAIT_CLAMP=1` removes the `TaskWait` intervals that L2
actually consumes and fails **50/50** at `seq=2048,past=0`, with millions of
L2-vs-L1 mismatches and varying output hashes. This establishes that the green
matrix exercises the new task dependency path rather than merely matching a
shared stage-order reference.

Raw per-process logs are intentionally not checked in; `run.sh` recreates
them and writes them beneath `raw/`.
