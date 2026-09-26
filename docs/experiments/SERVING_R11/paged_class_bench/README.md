# Isolated paged GEMM class throughput

The benchmark runs one full grid of the selected Llama B1 decode class with the exact page count, offsets, shared-memory size, tile shape and original collective stages from the generated plan. Both arms use the same 128 SM grid and input dimensions; the standard arm calls the R10 CUTLASS collective with 128 compute threads, and the page arm calls the R11 160-thread loader/compute page ring. A 3×L2 buffer is touched before each timed run, six timings per arm are retained, and the median is reported. Inputs are zero BF16 values; this is a transport/loop throughput probe, not a correctness test. The table reports effective weight bytes divided by wall time, which includes compute and epilogue. Raw build/run logs and exact commands are in `raw.tar.xz` (SHA256 `287fa0941ef4e0d1f3c7478574f50665fb3756696b97bf3a0e797e5380daefcd`).

| Class | Standard ms / GB/s | Paged ms / GB/s | Paged / standard |
|---|---:|---:|---:|
| qkv | 0.024 / 534 | 0.030 / 416 | 1.29× |
| gate_up | 0.103 / 654 | 0.154 / 436 | 1.50× |
| down | 0.055 / 614 | 0.109 / 309 | 1.99× |
| lm_head | 0.620 / 847 | 1.956 / 269 | 3.16× |

The isolated gate/up and down stage gaps, repeated 16 times, plus the single lm_head gap add to about 3.02 ms per decode step. Including QKV adds about 0.11 ms. The corresponding fixed-geometry full-request PG-off to PG-1 difference is about 2.93 ms per generated token. This close magnitude localizes the regression to GEMM page transport, though isolated-stage sums do not prove the exact per-step causal allocation because full execution overlaps tasks and shares memory bandwidth.
