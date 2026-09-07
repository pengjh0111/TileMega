# Direct cross-operator overlap

`TILEMEGA_TASK_TRACE=1` assigns each task start and end a value from one global
atomic sequence.  Unlike clocks sampled on different SMs, these values have a
single device-wide order.  A task counts as early when it starts before every
task in an earlier scheduled stage has ended.

✅ Fifty fresh-process traces per model all retained their accepted hashes.
On gqa2, a mean 68.72/200 task starts (34.36%, range 68–70) were early, with
132.72 prior-stage/task overlap pairs. On mha4, the mean was 205.70/512
(40.18%, range 197–217), with 397.70 overlap pairs. The instrument reports a
stable 184/496 queue stage transitions; the reduction is in
[`trace.tsv`](trace.tsv).

⚠️ The retired executor was not rebuilt with this new trace ABI. Its outer
loop cannot select a later-stage slot before finishing the current stage, so
the corresponding scheduling transition count is zero by control-flow
inspection; that is an inferred control, not a new hardware measurement.

⚠️ Trace atomics perturb timing, so these runs are evidence of overlap only; their
latency is not used by the L2/L1 performance comparison.
