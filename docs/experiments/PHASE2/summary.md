# R7 B0 — the whole-pipeline exposed wait denominator

Written for: the TileMega maintainers reviewing R7 §5.1 and deciding whether to
build B1.

```
FORK7 rule=1 whole_pipeline_exposed_wait_share=0.151 gemm_share=0.149 simt_share=0.002 cells=4
```

✅ Verified on RTX 4090 / sm_89. Four cells x nine fresh processes per cell for
the phase dumps, plus 50 fresh processes per cell for correctness (200 total,
200/200 `RESULT status=PASS`). Reproduce with `run.py build|correctness|dump`
then `analyze.py`; raw dumps under `raw/<cell>/phase/selected_r<i>/`. With the
probe compiled out (the default) the tree still builds and `ctest` passes 50/50.

## 1. Why the denominator was redone

FORK6 reported `cp_kloop_wait_share=0.278` against a threshold of 0.25. That
denominator was **critical-path GEMM mainloop cycles only**
(`COSTMODEL/analyze_kloop.py`). It excluded every SIMT body, and within a GEMM
it excluded setup, the first-operand wait and the epilogue. A cross-task
pipelining change is sized against the whole pipeline, so quoting 27.8% as the
headroom for B1 would have sized the investment against the wrong base — the
same error as the earlier rounds' "measure placement on a mismatched
configuration".

B0 keeps FORK6's four cells and their frozen `selected` configurations
verbatim (read back from `COSTMODEL/raw_kloop/<cell>/specs.json`) and widens
only the denominator to the whole critical path: every task's `run` interval,
all kinds. FORK6's 0.278 and FORK7's 0.151 are therefore the same measurement
under two scopes, not two disagreeing numbers.

## 2. What is measured

Per phase slot the `run` interval is partitioned into four segments that close
exactly against `run_cycles` (asserted per slot in `analyze.py::partition`):

| segment | definition |
|---|---|
| `setup` | `run_begin -> setup_end` |
| `wait` | `load_wait` (`setup_end -> first_operand_ready`) + the inner wait below |
| `compute` | `mainloop_cycles` minus the inner wait |
| `epilogue` | `mainloop_end -> run_end` |

The inner wait is `operand_wait_cycles` for an instrumented GEMM K-loop
(`TILEMEGA_TRACE_KLOOP`, R6) and `simt_wait_cycles` for a SIMT body
(`TILEMEGA_TRACE_SIMT`, new here). The SIMT probe brackets barriers the body
**already executed** — `RMSNormTaskBody::RunRow`'s reduction barriers and
`AttentionChunkTaskBody::RunTask`'s two — with `clock64` reads on thread 0 only.
No atomic, no new barrier, and no store inside a polling loop is added; the
instrumented body issues exactly the barriers it issued before.
`TILEMEGA_TRACE_SIMT` defaults to 0 and `#error`s without `TILEMEGA_TRACE_PHASE`,
so the default build is untouched (SASS stamp: `E2E_REAL/sass_identity/`).

`QKNormTaskBody` delegates to `RMSNormTaskBody::RunRow` and inherits the probe.

## 3. Per-cell result

Median over each cell's nine fresh processes, then the median over cells —
FORK6's aggregation.

| cell | configuration | cp_wait | gemm | simt | simt_body | simt_head |
|---|---|---:|---:|---:|---:|---:|
| gqa2 s4 | `32x16x64s2k1_k1_r1` | 0.1214 | 0.1191 | 0.0023 | 0.3482 | 0.0815 |
| gqa2 s128 | `32x16x32s2k1_k1_r5` | 0.1736 | 0.1725 | 0.0010 | 0.4516 | 0.0346 |
| mha4 s4 | `32x16x64s2k1_k1_r1` | 0.1275 | 0.1252 | 0.0023 | 0.3465 | 0.0793 |
| mha4 s128 | `32x16x16s2k1_k2_r5` | 0.2417 | 0.2406 | 0.0010 | 0.4414 | 0.0361 |

Median `simt_head_share=0.058`, `gemm_head_share=0.211`.

## 4. ⚠️ The margin is 0.0005 and the verdict is not robust

The pre-registered statistic clears the pre-registered threshold, so **rule=1**.
The unrounded value is **0.15055**, so it clears 0.15 by **0.0005** — the emitted
line reports three decimals, and reading "0.151 against 0.150" as a 0.001 margin
overstates it twofold. Per H6 and CLAUDE.md the threshold is not moved and the
margin is reported rather than smoothed over.

With four cells the median is the mean of the two middle cells, mha4 s4
(0.1275) and gqa2 s128 (0.1736); the two outer cells do not enter it. Those two
cells' own round-to-round ranges are 0.1240–0.1322 and 0.1695–0.1767, each some
fifteen times the margin. Taking one round at a time instead of the per-cell median:

| statistic | value | rule |
|---|---:|---|
| per-cell median of 9, then median of 4 (pre-registered) | 0.15055 | 1 |
| round-wise median of the four cells, min over rounds | 0.1485 | 2 |
| round-wise median of the four cells, max over rounds | 0.1521 | 1 |
| rounds whose four-cell median clears 0.15 | 6 of 9 | — |

Envelope from each cell's round extremes: [0.1468, 0.1544]. So rule=1 is the
honest reading of the statistic that was fixed in advance, but the measurement
does not separate this pipeline from the threshold. **B1 should not be justified
by this line alone**; §5 is the sizing argument that does not depend on the
margin.

Within-cell spreads over nine rounds: gqa2 s4 0.0115, gqa2 s128 0.0071,
mha4 s4 0.0082, mha4 s128 0.0250. The across-cell spread (0.1214 to 0.2417) is
structural, not noise: s128 exposes more GEMM wait than s4 in both models.

**One thing the margin does not depend on: which denominator.** "Whole pipeline"
was fixed in advance as the **critical path** — every task's `run` interval along
`cp_corrected_path` — because that is FORK6's denominator and keeping it is what
makes FORK7 comparable. Computing the same numerator over **all** tasks instead
of only the critical path gives 0.176 (per cell 0.1302, 0.1715, 0.1796, 0.1842),
which clears 0.15 more comfortably. The pre-registered statistic is the
critical-path one and it is what the line reports; the alternative is disclosed
here rather than substituted, because picking the denominator after seeing both
is exactly the error H6 forbids. That the weaker of the two still clears is the
reason rule=1 survives at all, and it is a better argument than the 0.0005.

## 5. What the line does not say, and the sizing that replaces it

**Most of the 0.151 is intra-task, but not all of it.** Splitting the wait by
where it occurs (median over each cell's nine rounds, then over cells):

| component | share of path | reachable by cross-task overlap |
|---|---:|---|
| GEMM wait inside the instrumented K-loop | 0.116 | no — own `cp.async`, own rendezvous |
| GEMM first-operand wait (`setup_end -> first_operand_ready`) | 0.033 | yes — this is the head |
| SIMT barrier wait | 0.002 | partly |

The three are independent medians, so exact additivity is not guaranteed; they
sum to 0.1508, which is the reported 0.151. The first-operand component is
strongly sequence dependent: 0.041/0.043 at s4 against 0.011/0.025 at s128, so
the short cells put about a third of their GEMM wait in the part B1 can reach
while the long cells put nearly all of it in the K-loop.

So rule=1 does **not** say B1 recovers 15%. It says the pipeline has 15%
measured idle time, of which roughly 0.035 is in a place cross-task overlap
reaches directly and roughly 0.116 is a CTA waiting on itself inside one task's
mainloop.

`simt_share=0.002` must not be read as "the SIMT bodies are busy". Two
measurement limits make it a lower bound:

- Thread 0's barrier time is a lower bound on CTA idle time. A thread that
  arrives last waits for nobody, so a barrier whose cost is real but whose
  thread 0 is the straggler reports ~0.
- A body with **no barrier at all** reports structurally zero exposed wait even
  though its loads stall. From `raw/segments.tsv`, `barriers=0` for rope,
  kvappend and elementwise; their `wait_share` is exactly 0.0000 by
  construction, not by measurement. A load-to-use bracketing was designed and
  rejected: it needs `memory`-clobbered stamps that serialize the load against
  its use and inflate the quantity being measured.

So B1 is sized against the **head share** — `setup + wait`, the part of a body
that a predecessor's epilogue could overlap. This is not idle time; it is work
that runs too late, which is a different claim and a different fix. From
`raw/segments.tsv` (critical path, round 0):

| kind | body share of path (s4 / s128) | setup share of body | compute share |
|---|---|---:|---:|
| gemm | 0.654 / 0.549–0.560 | 0.068–0.131 | 0.390–0.577 |
| attention | 0.100 / 0.333–0.343 | 0.023–0.164 | 0.809–0.973 |
| rmsnorm | 0.118 / 0.051–0.054 | 0.178–0.215 | 0.456–0.506 |
| elementwise | 0.077 / 0.030 | 0.154–0.170 | 0.818–0.824 |
| rope | 0.029 / 0.014 | 0.380–0.427 | 0.496–0.523 |
| kvappend | 0.021 / 0.013 | 0.465–0.545 | 0.384–0.428 |

SIMT bodies occupy 35–45% of the critical path (`simt_body` above), and their
heads are 3.5–8% of it (`simt_head`). GEMM heads are 21%. rope and kvappend
spend 38–55% of their body in `setup` alone — operand resolution ahead of any
load. That is the overlappable region, and it is large even where measured idle
time is near zero. rmsnorm's `epilogue_share` of 0.274–0.315 is the other
candidate: it is the tail B1's paging would overlap with a successor's head.

## 6. Portability

FORK7 is architecture specific. The sm_89 line does not carry over to sm_120 —
`cp.async` latency, barrier cost and occupancy all differ. `run_sm120.sh` is
write-only per H7 (this machine is a 4090); it was self-checked here with
`SELF_CHECK=1` (PASS) and carries a header saying it was not run on sm_120.
Re-derive the line on the target; do not quote the sm_89 one for sm_120.
