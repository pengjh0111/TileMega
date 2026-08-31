// Cross-compile probe for the TMA/warp-specialized collective selected by
// CollectiveBuilder on datacenter Hopper and Blackwell GeForce targets.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/float8.h>
#include <tilemega/Target/ArchDispatch.h>

using Arch = tilemega::arch::CurrentArch;
static_assert(tilemega::arch::Caps<Arch>::kTma);
static_assert(tilemega::arch::Caps<Arch>::kWarpSpecialized);
static_assert(!tilemega::arch::Caps<Arch>::kTcgen05);

constexpr bool kSm120 = cute::is_same_v<Arch, tilemega::arch::Sm120>;
using Element = cute::conditional_t<kSm120, cutlass::float_e4m3_t,
                                    cutlass::half_t>;
constexpr int kAlignment = kSm120 ? 16 : 8;

using Collective = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp,
    Element, cutlass::layout::RowMajor, kAlignment,
    Element, cutlass::layout::ColumnMajor, kAlignment,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

extern "C" __global__ void tilemega_builder_probe(unsigned long long* output) {
  if (threadIdx.x == 0) {
    output[0] = sizeof(typename Collective::SharedStorage);
    output[1] = sizeof(typename Collective::Params);
  }
}
