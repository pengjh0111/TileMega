# Single-page loader diagnosis

The paged GEMM loader computed `byte / PageBytes` and tested it for **every**
16-byte copy, even when `kGroupPages == 1`. `PageLayout::StageFits` and the
swizzled layout's exact `cosize` prove that every vector of each packed stage
lies in page zero in that case. The source now compiles out this test for
single-page groups; multi-page routing is unchanged.

The matched, isolated down-stage benchmark uses the archived generated plan
geometry and the same model-specific N and K in both binaries. For each model,
the unchanged and modified binaries were run alternately four times. Every
run measured six timed samples after a warmup and reported the median of the
middle two. The table in `microbench.tsv` lists these per-run medians. It is
an isolated transport diagnostic, not an end-to-end result or a claim that
PG-1 now beats the standard collective.

| Cell | Previous paged median | Single-page median | Change | Standard collective |
|---|---:|---:|---:|---:|
| Llama B1, down | 0.104544 ms | 0.075776 ms | −27.52% | 0.055296 ms |
| Qwen3 B16, down | 0.079872 ms | 0.058056 ms | −27.31% | 0.043008 ms |

One-launch Nsight Compute samples for Llama down (`profiler.tsv`) found
11,322,728 → 9,599,756 executed warp instructions and 1,928,336 →
1,275,192 branch instructions, while hardware DRAM read bytes stayed at
33.61 MB. These counters locate the improvement in instruction work, not
reduced DRAM traffic. The profiler changes timing, so its elapsed time is not
used above.

Two other isolated prototypes were rejected and never entered production:
pre-zeroing invalid M rows once per task changed Llama/Qwen3 paged down time
from about 0.1045/0.0797 to 0.1065/0.0809 ms; having one thread poll the
page barrier and then synchronize the group changed it to 0.1096/0.0829 ms.
The remaining paged/standard down-stage ratio is still about 1.37×/1.35×.

The previous binaries are from `c1f922acaf3054a525f3237c8ac33f3733f7071c`;
the changed binaries were built from that source plus the single conditional
in `PagedGemmTaskBody.h`. Exact binary hashes and runs are in `microbench.tsv`.
The generated serving plans are under `/root/r11_work/protocol/page_v2/`, and
the microbenchmark source is `docs/experiments/SERVING_R11/paged_class_bench.cu`.
`paged_gemm` and `page_ring` ctests passed after the production edit. Fresh
process checks and matched end-to-end controls for the newly compiled serving
binaries are separate evidence and are required before attributing a request
speedup to this change.

The four fixed-geometry serving libraries rebuilt from the changed header
have zero FP64 instructions in both serving kernels, as recorded in
`sass_audit.json`. This diagnostic audit does not replace the final EV-2
selected-plan SASS gate.

The changed header also compiles from the same generated decode plan for
sm_80, sm_89, sm_90, sm_100 and sm_120 (`arch_compile.json`). The sm_89
binary uses the SM80-class asynchronous-copy path; the newer-target SASS
includes TMA and barrier instructions. This is compilation evidence only for
architectures other than the native sm_89 GPU.

The new source passed **200/200 distinct-process checks**: Llama and Qwen3
at B=1/16, 50 fresh processes per cell, with reference L1 and candidate
L1/L2 token hashes equal in every process. The per-process records, binary
hashes and guard logs are retained in `diagnostics/protocol_raw.tar.xz`.

The matched full-request control is in `e2e/summary.tsv`. The PG-off/PG-1
E2E pairs are 3.346/4.211 s (Llama B1), 3.678/4.511 s (Llama B16),
5.006/7.231 s (Qwen3 B1), and 6.106/8.246 s (Qwen3 B16). Thus PG-1 remains
slower by 22.7–44.5% on the fixed geometry. The source edit improved the
isolated down stage, but did not make paged execution competitive. These
controls are not solver-selected EV-2 plans.

The requested trace results are in `diagnostics/summary.tsv`; every cell
also has 1023 per-step records and exact CG DRAM floors. At past 575 the
realized chains contain 135/119/232/204 links (Llama B1/B16, Qwen3 B1/B16),
and their measured spans exceed the DRAM floors by 2.128/2.076/3.770/3.748
ms, or 15.8/17.4/16.3/18.4 µs per link. The page-full plus dependency-wait
intersection averages 3.13/3.31/6.37/6.46 ms per CTA per step. Median
adjacent-launch gaps are 4.096/3.072/4.096/4.096 µs, totaling only about
3.65–3.78 ms per 1023-step decode request. Trace instrumentation perturbs
timing; compare performance using `e2e/`, not trace wall times. CTA-local
wait intervals cannot be summed to infer request wall time.
