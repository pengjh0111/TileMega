# R10 implementation completion audit

This corrects the earlier statement that only experiments remained. The missing
merge vector access was an implementation omission, not a dependency or a prompt
restriction. A source audit found additional omissions listed below. Numerical
unit-test success alone had not established the required implementation structure.

The original experiment worktree remains at `b22f9c3427cc5233f6c27139d26650896cf03285`.
Its search processes and EV-1 queue were not stopped, restarted or redirected.
Corrections live in `/root/r10_work/implementation_completion`, with an independent
build. Results produced by the original queue describe the original sources only.

R10 baseline: `f6c270ce09988f600a667dc2fc63f3527909c759`.
External prompt: `/root/Prompt/TileMega_R10_prompt.md`.
SHA256: `655fe479331d1cf771de6ecbe6b1bfdc9957847544b7b21752692d60ec3a0367`.

## Implementation corrections

| Requirement | Missing or incomplete behavior | Correction |
| --- | --- | --- |
| §4.3(c) merge | Scalar FP32 partial loads | Two float4 loads per eight BF16 output values; explicit 16-byte global output stores. SASS contains LDG.E.128 and STG.E.128. |
| §4.2(b)(c), §4.4 | Unswizzled epilogue spill, scalar partial/residual accesses and argmax inputs | Swizzled shared epilogue; paired accumulator spill; float4 split stores; 16-byte residual/BF16 accesses; aligned float4/int4 argmax reads with masked tails. Combine uses the same epilogue. |
| §4.2(a) | Shared-storage assertion covered only mainloop | Explicit mainloop/epilogue union and exact sizeof assertion; explicit Sm90/Sm100 extension specializations inherit the SM80 implementation. |
| §4.3(a) | QK scores/P round-tripped through shared memory; cached KV lacked overlapped double buffering | Warp-local online softmax and register P→PV mapping; two alternating KV buffers; issue next KV load before current QK/PV. Shared memory retains only cross-warp statistics and final output reduction. |
| §4.3(a)(b) | KV tile hardcoded to 64; prefill loop traversed beyond the last query owned by qb | Shared 64/32 selector used by codegen, pruning, occupancy estimate and pricing; prefill causal loop limit follows qb. |
| §4.8(a) | MMA work omitted padded rows; selected KV width not reflected in in-flight bytes | Charge padded rows/iterations, selected KV tile, and per-qb prefill work; include KV selection in price reuse key. Fused attention marked in critical-path dumps. |
| §4.8(b)(d) | In-flight calibration tail could consume before all outstanding groups completed; attention fit features described a full-S query although the benchmark executed one qb | Drain tail groups; attention calibration uses valid past=0 prefill and actual qb features consistent with PriceParts. New calibration measurements remain required. |
| §4.9(c) | Attention/argmax structural transitions discarded preparation reuse | Save/restore structure and flow/price/release state by (Ec,Rq,argmax tile-N), gated by incremental_prepare; full-control path rebuilds. No budget improvement is claimed before measurement. |
| Evidence integrity | Existing plan binaries could be silently reused after source changes | Hash actual compiler/backend/runtime/Python sources and target/hop calibration inputs; refuse unstamped/mismatched plan reuse. Current-source SASS gate excludes historical audits. |

These are source changes, not test-only work. Legacy event primitives, MPK-style
multi-step device scheduling, PG, TF, RMSNorm-to-GEMM fusion, TMA/WGMMA/tcgen05,
and CUTLASS submodule changes remain outside this patch.

## Remaining R10 sections checked

| Section | Existing implementation inspected | Remaining work |
| --- | --- | --- |
| SB-4 | vllm_baseline.py, hf_check.py, common prompt IDs and checkpoint metadata | Same-session final matrix/HF statistics; no new baseline implementation introduced here. |
| SB-1 | serving export, ModelPlan packing recipes, semantic lifting, batch roles, DramFloor | Recheck floor and dumps with final plans. |
| SB-2 | ServingAbi/ServingRuntime; external buffers, Params ring, past bounds | Final integrated C-1/C-2 and residency checks after rebuild. |
| SB-3 | plan.py, weights.py, state.py, engine.py, measurement guard | Re-run M1/M2 after attention changes, then final EV-1. |
| SV-16 | shared runtime release helper, weighted DRAM model, new task pricing | Recalibrate changed bodies and in-flight tail; do not treat the old fitted coefficients as measured for this implementation. |
| SV-17 | named pruning, class reuse, Simpson objective, warm starts, 0.98 A/B gate, top-3 mode selection | Final incremental check, pruning equivalence, 20 plan rebuild/solve and budget measurement. Existing G-7 failures remain failures. |
| EV-1 / reports | existing matrix, HF/token comparison, SASS audit, report collectors | Current-source matrix and measured decomposition are not complete. Old queue cannot fill this requirement for changed kernels. |

The errata remain applied: independent iteration counters per (instance, mode)
in ServingRuntime and plan.py; int32 token export; predicated 16-byte cp.async;
attention shared-memory R-1 check; thread count 128 excluded from the forbidden
SM-count literals; no new synchronization/race claim and no new 50-process gate.

## Validation scope and limits

`final_ctest.log` records nine affected CTests passing. `merge_vector_sass.json`
records real disassembly instruction counts. `prefill{32,64}_final/` contains
current-source dumps and PyTorch comparisons. `gemm_final/` and
`incremental_final/` are the final-version checks; their raw records determine
pass/fail, not this prose. Earlier unsuffixed development runs are retained but
are not substituted for final runs (the first incremental run overlapped compiler
rebuilds). No throughput improvement is inferred from numerical checks.

`verify.txt` is the complete structural verifier output. Current-source plan
SASS audits are absent, so K-12 and G-1 remain FAIL/pending evidence even when
the other 15 source checks pass. G-3/G-4/G-9/G-10/G-11 cannot be closed by these
focused tests. Rebuild and refit before accepting new performance results.

## Declared implementation choices

- Attention partitions a 64-position KV block into four 16-key warp slices (two
  active slices for KV=32); their softmax statistics are combined across warps.
  P remains in registers. This is an implementation detail of the specified
  tensor-core/online-softmax path, not a replacement scalar implementation.
- Explicit vector PTX is used for selected global loads/stores because nvcc
  scalarized C++ uint4 stores in the initial merge version. This does not alter
  the mathematical operation or introduce architecture-specific branches.
- Fixed-size aligned groups use vector accesses; predicates handle ragged tails.
  A serial reduction across live split-KV blocks remains, as required by LSE
  merge; it is not a scalar QK/PV matrix multiplication.
- The original working tree deliberately remains pinned until its running
  queue finishes. Pushing this branch does not change those running processes.

Final compiler incremental/full equivalence: Llama 20/20 and Qwen3 20/20,
maximum relative error exactly 0 (`incremental_final/report.json`).
Both KV widths pass 8/8 prefill PyTorch cases each with zero violations.
The final serving seed compiles for sm_80, sm_90 and sm_120; this is compilation
coverage only, not target-device execution (`arch_compile.json`).

A supplementary CPU check initially failed because the isolated worktree did
not contain ignored SEQSCAN bridge fixtures. Read-only symlinks to the original
fixtures were installed in the isolated tree; `cpu_regression_final.log` records
the retry. No original fixture or experiment process was modified.

The three compiled seed objects each have zero FP64 instructions in their L1/L2
kernels (`seed_sm{80,90,120}_sass.json`). These objects are not the final 20
selected serving libraries, and therefore do not close K-12/G-5. Runtime-release
and pruning CPU regression passes 2/2 after making ignored fixtures available
(`cpu_regression_final.log`). All 222 default build steps complete successfully
(`all_build.log`).

## Final focused results for this patch

- Serving CTest: **15/15 PASS** (`all_serving_ctest.log`).
- GEMM PyTorch full matrix: **1,344/1,344 PASS** (`gemm_final/summary.json`, raw `cases.jsonl`).
- Prefill PyTorch: **8/8 PASS for KV=64 and 8/8 PASS for KV=32** (zero violations).
- Incremental/full preparation: **40/40 PASS**, maximum relative error 0.
- Runtime-window/pruning CPU regression: **2/2 PASS**.
- sm_80/sm_90/sm_120 seed compilation: PASS; seed L1/L2 FP64 count 0 for each.
- Structural verifier: **15/16 PASS; K-12 FAIL** because current-source final
  20-plan SASS evidence has not been generated. Consequently G-1 is not passed.

No benchmark throughput, final C-1/C-2, G-7 budget or final EV-1 claim is made.
