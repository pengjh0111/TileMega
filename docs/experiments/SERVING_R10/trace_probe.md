# First serving trace diagnostic

The selected Llama decode B=1 plan was rebuilt with
`-DTILEMEGA_TRACE_V2=1`; the ordinary measured `.so` was not changed. At
past=575, 32 L2 diagnostic launches on one plan instance produced 8,472
task-slot rows and 3,384 event rows. The mean step time of this instrumented
build was 4.316448 ms; its recorded kernel span was 3.978240 ms. The existing
`TRACE_V2/analyze.py --window 1` completed, with no negative hops and a
7.84% reconstructed-path error versus the instrumented step timer. This is
diagnostic evidence, not an EV-1 timing or a claim that the trace overhead is
zero. The selected uninstrumented winner uses L1.

The rate below divides CG-derived no-producer bytes by the measured span of
each task space, then aggregates within each operator class. It is an
effective service rate, not a hardware DRAM-counter reading.

| class | effective GB/s | measured critical-chain share |
| --- | ---: | ---: |
| QKV GEMM | 254.0 | 20.1% |
| output GEMM | 675.6 | 5.0% |
| gate/up GEMM | 864.4 | 30.3% |
| down GEMM | 830.9 | 16.9% |
| lm_head GEMM | 929.4 | 14.1% |
| fused attention | 86.1 | 5.9% |

The exact compiler invocation is in
`/root/r10_work/serving_trace/llama_decode_B1/build_command.json`;
raw `slots.tsv`, `waits.tsv`, `events.tsv`, `meta.tsv`, analyzer output,
GPU guard, and FP64 audit (zero instructions) are in the same directory.
The source path is `python/tilemega/serving/trace.py`; the four-cell runner
is `docs/experiments/SERVING_R10/trace_serving.py`.
