# R10 serving backend coverage

The generated seed plans contain the following TaskKinds. Split-K adds
`kGemmCombine` at runtime when the chosen geometry requests more than one
chunk. Counts are from the generated `kStages` arrays, before split expansion.

| Model and phase | GEMM | RMSNorm | fused attention | merge | embedding | argmax reduce |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama prefill | 65 | 33 | 16 | 0 | 1 | 1 |
| Llama decode | 65 | 33 | 16 | 16 | 1 | 1 |
| Qwen3 prefill | 113 | 57 | 28 | 0 | 1 | 1 |
| Qwen3 decode | 113 | 57 | 28 | 28 | 1 | 1 |

| Serving operation | TaskBody / implementation | Architecture path | Focused test |
| --- | --- | --- | --- |
| QKV, output, gate/up, down and lm_head GEMM | `GemmStageTaskBody` → `ServingGemmTaskBody`; 16-byte cp.async, swizzled shared memory, ldmatrix and BF16 mma.sync; shared epilogue handles store, residual, SwiGLU and argmax partial. TileM=16, TileN=32 uses ldmatrix.x2 for B because each N warp holds only two 32-bit registers; all other listed shapes use x4 | SM80-class default selected by `Caps<Arch>`; SM90/SM100 specializations remain extension points | `serving_gemm_config`, `serving_gemm_task_body`, `serving_epilogue` |
| Split-K combine | `GemmCombineTaskBody` → `ServingGemmCombineTaskBody`; ordered FP32 reduction and the same serving epilogue | 128-thread SIMT reduction and vector epilogue on all supported architectures | `serving_combine`; B=16 end-to-end regression in `solver_interval/b16_correctness` |
| Fused attention | `FusedAttentionTaskBody`; QK and PV use BF16 mma.sync, both ldmatrix directions, 16-byte cp.async for cached KV and FP32 online softmax; Q/K normalization and RoPE are in the task | SM80-class default for all supported architectures | `serving_attention_mma`, `serving_fused_attention`, `serving_attention_cases`, `serving_task_index` |
| Split-KV LSE merge | `AttentionMergeTaskBody`; thread-parallel output dimensions, FP32 `exp2f` reweighting across blocks | 128-thread CUDA-core path | `serving_fused_attention`, `serving_attention_cases` |
| Row RMSNorm and final selected-row norm | `ServingRMSNormTaskBody`; vector load and warp/shared reduction | 128-thread CUDA-core path | `serving_rmsnorm` |
| Token embedding | `ServingEmbeddingTaskBody`; 16-byte row copy from tied embedding | 128-thread CUDA-core path | `serving_state_tasks` |
| Argmax reduce and token write | `ServingArgmaxReduceTaskBody`; warp/shared reduction with lowest-index tie break | 128-thread CUDA-core path | `serving_state_tasks` |

No serving GEMM or attention matrix product uses the legacy scalar QK/PV
loop. The LSE merge retains a serial loop over live KV blocks per output
thread; it is a reduction over split blocks, not a scalar matrix product.
The b22 checkpoint used scalar FP32 partial loads; that implementation gap is
now corrected: two float4 partial loads and one explicit 16-byte BF16 store per
eight outputs. Real SASS counts are in `implementation_completion/merge_vector_sass.json`.
Attention now keeps P in registers, double-buffers cached KV, and selects KV=64
or 32 according to the selected GEMM shared-storage budget. The GEMM epilogue
uses swizzled shared spill and vector residual/output access; argmax uses aligned
float4/int4 input groups. See `implementation_completion/audit.md` for exact
changes, current focused evidence and the required final-source reruns.

The following numerical results describe the earlier checkpoint:
The serving GEMM matrix passes 1,344/1,344 PyTorch comparisons over the
specified M, N/K, tile, split-K and epilogue cases
(`task_body_tests/gemm_matrix/summary.json`). The attention tests pass 60/60
decode boundary cases (`task_body_tests/attention_decode_matrix.log`) and
8/8 prefill cases (`task_body_tests/prefill_attention/torch_comparison.json`).
The scalar serving bodies and combine are covered by the named CTest cases
above; all 80 registered tests passed in the disjoint groups recorded in
`ctest_partial_reason.md`.
