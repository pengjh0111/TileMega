# DNN1 H800 external baselines

Verified checkpoint: TVM **25/25 BF16 cells accepted**; AITemplate **GPU collection pending**. TileMega native optimization remains paused. Existing TRT/torch.compile primary125 rows are unchanged.

Current data: [canonical150 rows](DNN1_baselines.csv), [TVM diagnostics](DNN1_tvm_diag.csv), [per-task allocation](DNN1_tvm_task_coverage.csv). Superseded importer failures are retained in [history](DNN1_tvm_history.csv) and [diagnostic history](DNN1_tvm_diag_history.csv); the five historical FP16 failures are not a new fallback cohort.

## Protocol and latency

Each accepted cell has five fresh processes passing the original FP32 numerical oracle, including three independent timings:50 warmups,200 CUDA-event samples per process. Cell median/p90 are medians of the three process statistics; all process ranges, clocks, artifacts and source receipts remain in CSV. No partial or rejected cohort is combined with an accepted cell. New observations use process/extra-memory checks only, physical flock, NUMA affinity and explicit device selection; GPU6 is excluded. Container PID ownership remains provisional, not strict host ownership proof.

TVM medians in milliseconds:

| model | B1 | B2 | B4 | B8 | B16 |
|---|---:|---:|---:|---:|---:|
| resnet18 | 1.094640 | 1.034608 | 1.657744 | 2.530784 | 3.780800 |
| mbv1 | 0.514240 | 0.657696 | 0.955200 | 3.308768 | 4.870688 |
| mbv2 | 1.485520 | 2.854848 | 3.399696 | 12.201648 | 14.903312 |
| bert | 1.381456 | 1.559728 | 1.857264 | 2.616784 | 3.914048 |
| nafnet | 29.082048 | 71.762402 | 69.100273 | 250.149696 | 715.048248 |

## Fixed recipe and source segments

TVM source7b5acb32bd271342a24130f4ef439ff6c2e52c6c, FFI daf594da8e6950fdc02ae35fcf4e0ab2e59a4979. All25 use BF16 operands/weights and the original FP32 accumulation conversions, unchanged export/checkpoint/input assets, keep_params_as_input=True, detached resident parameters, cuBLAS/cuDNN partition attempts, RunCodegen, zero, default evolutionary/XGB MetaSchedule and prescribed DLight ordering. Requested budget2000 global trials/1800s is not a successful-trial guarantee.

| source diff SHA256 | cells | repair scope |
|---|---|---|
| 053686eaff6a873f88ea9ec30f05816234a6d5ac0c89ecfd5ed9e2a895c439c9 | bert_b1, bert_b2, bert_b4, bert_b8, mbv1_b1, mbv1_b16, mbv1_b2, mbv1_b4, mbv1_b8, mbv2_b1, mbv2_b16, mbv2_b2, mbv2_b4, mbv2_b8, nafnet_b1, resnet18_b1, resnet18_b16, resnet18_b2, resnet18_b4, resnet18_b8 | frozen recipe6 |
| 2259304e32cb4c98e519b52d2ab7b4afa0a9c43bdb901ec446c952f721dd181f | nafnet_b2, nafnet_b4 | XGB/schedule Python repair; unchanged C++ compiler |
| c2333a7e6a8770d70de1c451b3f179a648d5c1a38300be0bec0d3b41d11eb994 | nafnet_b8, nafnet_b16, bert_b16 | typed runner plus XGB/schedule Python repair; unchanged C++ compiler |

The C++ compiler library SHA is b2796b9acf8568e54345f28aba7ea59ef207e88bd69093c3a0fb68cab7916f9e. NAF2/4 and the final BERT16 measurement reuse completed databases with zero new tuning. NAF8 includes197 prior records plus1280 fresh; BERT16 includes164 prior plus1418 fresh. Prior task IDs/journals remain separate. Cancelled/dirty attempts and the original20 accepted rows are retained. The finalizer verified official device-annotation fields by stripping only those fields and reproducing the original exact receipt byte SHA; it did not rewrite retained measurements.

## Effective search budget and task coverage

Selected databases total37,285 committed records: 32,386 successful, 4,899 failed. This sums the25 selected databases, including retained prior records; it is not all historical compute or convergence evidence. Source fresh epochs hit the1800s limit; measurement reuse time and source epoch time are separate diagnostic columns.

| cell | committed / success / fail | prior + fresh | fresh tasks successful / initialized | unallocated / all-failed |
|---|---:|---:|---:|---:|
| bert B1 | 1407 / 1384 / 23 | 0 + 1407 | 19 / 23 | 0 / 4 |
| bert B2 | 1403 / 1379 / 24 | 0 + 1403 | 19 / 23 | 0 / 4 |
| bert B4 | 1531 / 1517 / 14 | 0 + 1531 | 19 / 23 | 0 / 4 |
| bert B8 | 1593 / 1585 / 8 | 0 + 1593 | 20 / 23 | 0 / 3 |
| mbv1 B1 | 1823 / 1622 / 201 | 0 + 1823 | 33 / 33 | 0 / 0 |
| mbv1 B16 | 1184 / 919 / 265 | 0 + 1184 | 29 / 33 | 4 / 0 |
| mbv1 B2 | 1636 / 1386 / 250 | 0 + 1636 | 33 / 33 | 0 / 0 |
| mbv1 B4 | 1426 / 1182 / 244 | 0 + 1426 | 31 / 33 | 2 / 0 |
| mbv1 B8 | 1222 / 1063 / 159 | 0 + 1222 | 29 / 33 | 4 / 0 |
| mbv2 B1 | 1969 / 1579 / 390 | 0 + 1969 | 37 / 58 | 21 / 0 |
| mbv2 B16 | 1466 / 1116 / 350 | 0 + 1466 | 36 / 58 | 22 / 0 |
| mbv2 B2 | 1852 / 1412 / 440 | 0 + 1852 | 37 / 58 | 21 / 0 |
| mbv2 B4 | 1560 / 1251 / 309 | 0 + 1560 | 34 / 58 | 24 / 0 |
| mbv2 B8 | 1530 / 1231 / 299 | 0 + 1530 | 36 / 58 | 22 / 0 |
| nafnet B1 | 1379 / 1271 / 108 | 0 + 1379 | 37 / 72 | 35 / 0 |
| resnet18 B1 | 1707 / 1431 / 276 | 0 + 1707 | 28 / 28 | 0 / 0 |
| resnet18 B16 | 1335 / 1084 / 251 | 0 + 1335 | 28 / 28 | 0 / 0 |
| resnet18 B2 | 1663 / 1488 / 175 | 0 + 1663 | 28 / 28 | 0 / 0 |
| resnet18 B4 | 1507 / 1273 / 234 | 0 + 1507 | 28 / 28 | 0 / 0 |
| resnet18 B8 | 1476 / 1244 / 232 | 0 + 1476 | 28 / 28 | 0 / 0 |
| nafnet B2 | 1425 / 1224 / 201 | 0 + 1425 | 38 / 72 | 34 / 0 |
| nafnet B4 | 1042 / 904 / 138 | 0 + 1042 | 30 / 72 | 42 / 0 |
| nafnet B8 | 1477 / 1275 / 202 | 197 + 1280 | 29 / 72 | 43 / 0 |
| nafnet B16 | 1090 / 992 / 98 | 0 + 1090 | 26 / 72 | 46 / 0 |
| bert B16 | 1582 / 1574 / 8 | 164 + 1418 | 24 / 24 | 0 / 0 |

## Fairness limits

- TVM times direct uncaptured Relax VM calls. TRT uses explicit CUDA Graph replay; torch.compile modes enable backend-controlled graph optimization, but retained receipts alone do not prove actual capture per mode. Shared50/200 events do not make launch optimization equivalent.
- TVM weights are GPU-resident and cached once with VM.save_function; no per-forward H2D or664-weight Python marshalling claim is warranted. TC/freezing can prepack constants: the accepted NAF wrapper removes an intro-weight layout conversion. Same-export CPU BindParams reduces NAF call_tir921→555 and PrimFuncs72→60, with unchanged32.7382G weighted scalar TIR FLOP estimate; no GPU speedup or whole-branch DCE was measured. Actual checkpoint72 beta/gamma tensors (19,712 elements) are zero, while both inspected TC sources retain226 convolution callsites.
- The pinned default rules admit FP16/int8 tensorization in the inspected branch; the128³ BF16 CPU design-space probe produced zero tensor-core sketches versus two FP16 sketches. cuDNN BYOC excludes BF16 for this graph; enabling cuDNN build/linking does not establish BF16 offload. cuBLAS covers100% of BERT counted conv/matmul MACs, so this TIR rule gap cannot alone explain BERT. Generated TIR markers do not inspect opaque library instructions.
- The BYOC fraction counts only static conv/matmul2×MAC, not the full graph. Missing task coverage, compiler failures and continued isolated-task improvements are separate from launch/constant visibility. Convergence curves are task-latency proxies, not E2E speedup. NAF16 has97 builder failures and one invalid-value candidate launch; its selected full model still passes5/5.
- Inactive codegen, BindParams, CUDA Graph and extended-budget candidates are not used in these25 results. No fair-gap attribution, graph speedup, extra-budget speedup or complete optimization equivalence is claimed.

## AITemplate status

Pinned AIT23a2e348857591ac41f4946f0177c53d833a0104; uniform static ATen/NHWC BF16, FP32 accumulation, SM90a, graph_mode=True,50/200 and original oracle. Recipe3 adds only explicit -lcuda for SM90 TMA profiler executables; CPU25 import readiness was refreshed after the source change. The original CPU profiler link error is retained. AIT GPU engines, profiler selections, correctness, capture traces and timings remain pending; CPU build success is not GPU support/performance evidence. AIT starts only after all25 validated TVM outcomes and drainage. No FP16 substitution or silent no-graph fallback.

## Evidence

Raw publication/hash/PID proofs: runs/dnn1/resume_tvm/completed25_publication_v1/. Exact source/task/receipt audits: runs/dnn1/baseline_completion_audit/tvm_cpu_audit*. Fairness probes: runs/dnn1/resume_tvm/fairness_v1/{convergence.json,design_space_receipt.json,static_weights_v1/summary.json}. Final BERT16: baseline_only_resume_20261105_v2_integrity. AIT source/CPU25: runs/dnn1/resume/aitemplate/linkfix_graph_v3/. All raw data, errors, traces, binaries and generated sources remain ignored under runs/.
