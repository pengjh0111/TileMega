# Reproduction commands

Run from the completion worktree. CUDA target sm_89 in these local test commands
comes from the installed RTX 4090; production code derives architecture/resources
from TargetSpec. No performance claim is made from these unit checks.

```sh
cmake --build /root/r10_work/completion_build -j4
flock /root/r10_work/serving_gpu.lock ctest --test-dir /root/r10_work/completion_build --output-on-failure -R 'serving_(gemm_config|pruning|epilogue|state_tasks|combine|attention_merge|attention_cases|rmsnorm|prefill_attention)$'
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_89 --expt-relaxed-constexpr -DTILEMEGA_MIDPOINT_REFINE=0 -Iinclude -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include test/unit/serving_gemm_matrix_test.cu -o /root/r10_work/completion_gemm_final
/root/venvs/tilemega-torch213-cu126/bin/python docs/experiments/SERVING_R10/test_gemm_torch_matrix.py --binary /root/r10_work/completion_gemm_final --out docs/experiments/SERVING_R10/implementation_completion/gemm_final --work /root/r10_work/completion_gemm_final_data
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_89 --expt-relaxed-constexpr -DTILEMEGA_TEST_KV_TILE=32 -DTILEMEGA_MIDPOINT_REFINE=0 -Iinclude -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include test/unit/serving_prefill_attention_test.cu -o /root/r10_work/completion_prefill32_final
flock /root/r10_work/serving_gpu.lock python3 docs/experiments/SERVING_R10/test_prefill_torch.py --binary /root/r10_work/completion_prefill32_final --out docs/experiments/SERVING_R10/implementation_completion/prefill32_final
flock /root/r10_work/serving_gpu.lock python3 docs/experiments/SERVING_R10/test_prefill_torch.py --binary /root/r10_work/completion_build/serving_prefill_attention_test --out docs/experiments/SERVING_R10/implementation_completion/prefill64_final
python3 docs/experiments/SERVING_R10/run_incremental_serving.py --compiler /root/r10_work/completion_build/tools/tilemega-compile --out docs/experiments/SERVING_R10/implementation_completion/incremental_final --work /root/r10_work/completion_incremental_final
python3 docs/experiments/SERVING_R10/verify.py
```

The incremental directory records all actual compiler arguments per invocation.
The cross-architecture check compiles `/root/r10_work/completion_arch/seed.cu`
with nvcc `-std=c++17 -O3 -DTILEMEGA_MIDPOINT_REFINE=0
--expt-relaxed-constexpr -Xptxas=-v -c -x cu -Iinclude
-Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include
-Ithird_party/cutlass/test -arch=sm_<80|90|120>`; per-target ptxas logs are retained.
GPU tests use the same GPU file lock as the old measurement queue.
