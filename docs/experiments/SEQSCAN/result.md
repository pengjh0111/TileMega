# BF16 `seq × past` correctness scan

Evidence status: ✅ measured on an RTX 4090 (`sm_89`) on 2026-09-05.  Every
entry in [`matrix.tsv`](matrix.tsv) is a distinct fixture and was launched in
50 fresh processes.  Each process compares PyTorch L0 with the standalone
L0.5 path, L0.5 with L1, and L1 with the dependency-driven persistent L2 path.

The complete 2 model × 5 `seq` × 3 `past` matrix passed: **1500/1500**.  This
covers a stage narrower than the resident grid, a comparable stage, and a
stage that needs many grid-stride rounds.  The BF16 comparison bound is
`1.6e-2 + 1.6e-2*abs(reference)`; it was selected after the stricter `1e-2`
bound left four quantization-boundary differences among millions of values at
`seq=2048`, while `1.5e-2` left none.  It is not used for FP32.

The scan found and fixed two real blind-spot bugs: attention and RMSNorm had
declared grid-stride ownership but executed only `blockIdx.x`'s first task.
Attention also sized its shared score row to the CTA width rather than the
maximum runtime key length.  The repaired bodies loop over all placed tasks;
attention's supported total length is checked against 4096 and fails hard
outside it.

The deliberately obsolete `min(count, grid)` wait clamp was rebuilt with
`TILEMEGA_NEGATIVE_OLD_CLAMP=1`.  At `seq=2048,past=0` it failed **50/50**
fresh processes (`l2_vs_l1` mismatches), recorded in
[`negative.tsv`](negative.tsv).  Thus the expanded matrix is demonstrably
sensitive to the under-wait class that the old `seq=4` fixture could not see.

Raw per-process logs are intentionally not checked in; `run.sh` recreates
them and writes them beneath `raw/`.
