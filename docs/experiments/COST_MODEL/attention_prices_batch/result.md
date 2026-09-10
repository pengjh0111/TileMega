# Attention plans: work, resources, prices and candidate selection

Verified on source `4c98c25a`. This is CPU analysis of the frozen
`../attention_models/` 800-process BF16 experiment, not another GPU run.
`manifest.json` identifies the price executable and GPU receipt table.
Reproduce work with `verify_attention_work.py`, prices with
`verify_attention_prices.py --out NEW_DIRECTORY`, plan selection with
`verify_attention_dp.py`, and paired statistics with `summarize_attention.py`.

## Implementation and checks

- `lib/Solver/AttentionWork.cpp:54` derives four phase task spaces and exact
  physical R/W sets from CG sequence/cache extents and the chosen chunk
  partition. Scores and partials are FP32; original Q/K/V/output use model
  dtype. Masked scores still have writes, but no QK arithmetic. Empty chunks
  still write their zero partials. Task coordinates remain symbolic.
- `lib/Analysis/OpArithmetic.cpp` holds the four phase signatures alongside
  the original declarations. Scores: `2*width+1` per valid key. Normalize:
  subtraction, zero-seeded sum and division plus exp. Partial PV: two FLOPs
  per `(key, component)`. Combine: `chunks` zero-seeded additions per output.
- `lib/Solver/AttentionWork.cpp:10` carries per-stage choices, unique FP32
  workspace bytes and the shared union into ModelDescription. Changing a
  plan invalidates its old runtime-event metrics. No stale event counts are
  reused. `FromGeneratedCuda` remains unchanged.
- `lib/Solver/CostModel.cpp:467` prices each phase through the shared scalar
  resource evaluator and its actual tail-wave occupancy. Four runtime
  stages also mean four L1 barriers. The Stream-K fitted combine intercept
  is not reused: this combine is a different, zero-seeded per-head kernel;
  its exact FP32 reads, additions and BF16 writes use the common lanes.
- `lib/Solver/ChainDP.cpp:28` selects among supplied complete attention
  plans, each with its compiled register requirement, while its inner GEMM
  DP still pins residency outside the chain. Both ordinary and interface
  DPs include attention shared bytes in the whole-kernel resource bound.

Verified: 2720 independent phase task-work checks, 14 rejection branches
with zero ISL reference delta, 14 frozen scalar-price tables byte-identical,
and both historical out-of-domain rejections unchanged. Full CTest: 37/37.
Policy check passed. All 3200 comparisons of task_refs, waits, shared bytes
and residency against 800 GPU receipts match exactly. Resources remain
128 threads, 212 registers, 24576 dynamic shared bytes and 2 CTA/SM for all
four compiled plans. Additional workspace grows with chunk; it is included
in the cache footprint, not hidden in the shared union.

## Ranking

Ascending chunk order; GPU columns here are 50-process descriptive medians.
The primary 25-round paired confidence intervals and Wilcoxon tests are in
`../attention_models/paired_stats.json`; the 50-round sensitivity is separate.

| Model/seq | Predicted L1 | Measured L1 | Predicted L2 | Measured L2 |
|---|---|---|---|---|
| gqa2/4 | 1,4,2,8 | 1,4,8,2 | 1,2,4,8 | 1,4,2,8 |
| gqa2/128 | 1,2,4,8 | 1,2,4,8 | 1,2,4,8 | 1,2,4,8 |
| mha4/4 | 1,4,2,8 | 1,4,8,2 | 1,2,4,8 | 1,4,2,8 |
| mha4/128 | 1,2,4,8 | 1,2,8,4 | 1,2,4,8 | 1,2,4,8 |

Verified: the candidate solver selects chunk 1 in all four cells, matching
the measured winner. All 16 alternative direct-price bit patterns equal
their returned CostBreakdown; the exhaustive minimum matches each choice.
Detailed tables are in `../attention_dp/`. Internal rankings are not all
correct, and absolute prices still substantially underpredict latency.
For example gqa2/128 chunk8 predicts L2 0.545651 ms vs 0.938992 ms measured.
No coefficient was changed to fit these observations.

## Limitations and retained failures

This is an exact selection over the supplied four compiled full-kernel
plans, not the Cartesian product of every per-layer chunk. The interface
can carry nonuniform per-stage plans, but that full candidate domain has
not been tested. Joint L2 candidate transitions still explicitly reject;
these tests do not close that implementation debt or symbolic (a).

R/W counts are unique physical elements, not issued-load instruction counts.
Normalize currently executes serially in tid0; the aggregate CUDA/SFU lane
does not yet represent its serial instruction throughput. Exact work and
event counts do not establish accurate latency. This remains a possible
cause of price residuals, not a measured attribution.

The old slow run is preserved at `../attention_prices/`: repeated QP parses
inside per-task evaluation stalled CPU analysis, not a GPU kernel. Batch
`EvalPoints` retains exact ISL evaluation while parsing each work QP once;
the completed old chunk2 TSV and new chunk2 TSV are byte-identical.
The first regression runner incorrectly treated an already-archived domain
rejection as a new failure; it now verifies its exit code and exact output.
The DP decomposition differs from whole-model accumulation by at most
1.7462298274040222e-10 ns due to FP64 addition order; returned CostBreakdown
uses the actual evaluator and is bit-identical. No tolerance was introduced.

The requested large-context old/new attention fraction was already run at
seq512/past512 and remains in `../scalar_work/result.md`; these chunk-plan
tests do not overwrite or extrapolate it.
