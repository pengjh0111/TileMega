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
The current focused tests cover representative shapes; the full PyTorch
cross-product in R10 §4.2(d), §4.3(e) and §4.4 remains a separate G-2 gate.
