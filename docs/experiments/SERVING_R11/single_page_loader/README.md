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

The new full-request and trace evidence is queued behind 200 fresh-process
token checks. `run_single_page_followup.py` checks that gate before timing
the four endpoint cells, then collects the requested chain, page-stall and
launch-gap diagnostics from trace builds of the same generated plans.
