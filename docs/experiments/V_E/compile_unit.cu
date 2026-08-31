#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <unit/gemm/device/default_gemm_configuration.hpp>

using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float>;
using Collective = Config::CollectiveMainloop;

extern "C" __global__ __launch_bounds__(128) void compile_unit(int* output) {
  extern __shared__ unsigned char storage[];
  int i = (threadIdx.x * 257 + blockIdx.x) % sizeof(Collective::SharedStorage);
  storage[i] = threadIdx.x;
  if (threadIdx.x == 0) output[blockIdx.x] = storage[i];
}
