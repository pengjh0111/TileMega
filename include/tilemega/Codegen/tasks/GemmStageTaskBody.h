// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 CUTLASS/CuTe GEMM TaskBody.  Handwritten body; the
// problem shape arrives as a generated GemmDesc, never as a constant here.
#pragma once

#include <tilemega/Codegen/tasks/ModelRuntime.h>

#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <unit/gemm/device/default_gemm_configuration.hpp>

#include <type_traits>

namespace tilemega::codegen {

// The direct-collective adapter family validated by V-B.  FP32 SIMT keeps the
// numerical oracle the FP32 ExportedProgram; CUTLASS still selects
// MainloopSm80CpAsync<3> for this configuration.
using GemmConfig = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    float, cutlass::layout::RowMajor,
    // CUTLASS's logical B tensor is (N,K); ColumnMajor gives (K,1) logical
    // strides and so consumes PyTorch's contiguous [N,K] weight directly.
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float>;
using GemmMainloop = GemmConfig::CollectiveMainloop;
using GemmEpilogue = GemmConfig::CollectiveEpilogue;
using GemmProblem = cute::Shape<int, int, int, int>;
using ExpectedContiguousWeightStride = cute::tuple<int64_t, cute::C<1>, int64_t>;
static_assert(std::is_same_v<typename GemmMainloop::StrideB,
                             ExpectedContiguousWeightStride>,
              "logical CUTLASS B(N,K) must expose contiguous [N,K] as (K,1)");
inline constexpr int kGemmThreads = GemmConfig::ThreadCount;
inline constexpr int kGemmTileN = cute::size<1>(typename GemmMainloop::TileShape{});

/// Host-built launch arguments for one generated GemmDesc.
struct GemmInvocation {
  GemmProblem problem;
  GemmMainloop::Params mainloop;
  GemmEpilogue::Params epilogue;
  int tiles_n;
};

template <class Arch, class SmemUnion, int Threads>
struct GemmStageTaskBody {
  using SharedStorage = GemmMainloop::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = Threads == kGemmThreads;
  static constexpr char const* kLogicalA = "(M,K), strides (K,1)";
  static constexpr char const* kLogicalB = "(N,K), strides (K,1)";

  /// One N-tile per CTA: `blockIdx.x` is the task index.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    return {TaskOwnershipKind::kTilePerBlock,
            static_cast<GemmInvocation const*>(p.gemms)[stage.gemm].tiles_n};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    using namespace cute;
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    int tile_n = static_cast<int>(blockIdx.x);
    if (tile_n >= invocation.tiles_n) return;
    constexpr auto tile_shape = typename GemmMainloop::TileShape{};
    auto [M, N, K, L] = invocation.problem;
    Tensor matrix_a = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_A),
                                  make_shape(M, K, L), invocation.mainloop.dA);
    Tensor matrix_b = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_B),
                                  make_shape(N, K, L), invocation.mainloop.dB);
    auto block_coord = make_coord(0, tile_n, _, 0);
    Tensor gA = local_tile(matrix_a(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<_1, X, _1>{});
    Tensor gB = local_tile(matrix_b(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<X, _1, _1>{});
    auto residue = make_tuple(M, N - size<0>(gB) * tile_n,
                              K - size<1>(gA) * size<2>(gA));
    typename GemmMainloop::TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    char* shared = reinterpret_cast<char*>(&smem.gemm);
    GemmMainloop mainloop;
    mainloop(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
             static_cast<int>(threadIdx.x), shared);
    GemmEpilogue epilogue(invocation.epilogue);
    epilogue(invocation.problem, tile_shape, make_coord(0, tile_n, 0, 0),
             accum, tiled_mma, residue, static_cast<int>(threadIdx.x), shared);
    __syncthreads();
  }
};

}  // namespace tilemega::codegen
