// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 CUTLASS/CuTe GEMM TaskBody.  Handwritten body; the
// problem shape arrives as a generated GemmDesc, never as a constant here.
#pragma once

#include <tilemega/Backend/CutlassGemmCandidate.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>

#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <type_traits>

namespace tilemega::codegen {

// The granularity `g` of every GEMM task space.  It is a compile-time knob so
// one generated model can be rebuilt at another point of the implementation
// search space without touching the generator (§2.3 Reparam); the defaults
// reproduce CUTLASS's own SIMT f32 TN configuration byte for byte.
#ifndef TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_TILE_M 128
#endif
#ifndef TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_TILE_N 128
#endif
#ifndef TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_TILE_K 16
#endif
#ifndef TILEMEGA_GEMM_STAGES
#define TILEMEGA_GEMM_STAGES 3
#endif
// The reduction chunk count of §2.4's Split, applied to every GEMM's `k`.
// 1 leaves the task space unsplit.
#ifndef TILEMEGA_GEMM_SPLIT_K
#define TILEMEGA_GEMM_SPLIT_K 1
#endif

using GemmImpl =
    backend::GemmCandidate<TILEMEGA_GEMM_TILE_M, TILEMEGA_GEMM_TILE_N,
                           TILEMEGA_GEMM_TILE_K, TILEMEGA_GEMM_STAGES>;
static_assert(GemmImpl::kShapeLegal,
              "the selected GEMM tile shape is not a legal candidate; query "
              "backend::GemmCandidate::kShapeLegal before compiling");
using GemmMainloop = GemmImpl::Mainloop;
using GemmEpilogue = GemmImpl::Epilogue;
using GemmProblem = cute::Shape<int, int, int, int>;
// CUTLASS's logical B tensor is (N,K); ColumnMajor gives (K,1) logical strides
// and so consumes PyTorch's contiguous [N,K] weight directly.
using ExpectedContiguousWeightStride = cute::tuple<int64_t, cute::C<1>, int64_t>;
static_assert(std::is_same_v<typename GemmMainloop::StrideB,
                             ExpectedContiguousWeightStride>,
              "logical CUTLASS B(N,K) must expose contiguous [N,K] as (K,1)");
inline constexpr int kGemmThreads = GemmImpl::kThreads;
inline constexpr int kGemmTileM = cute::size<0>(typename GemmMainloop::TileShape{});
inline constexpr int kGemmTileN = cute::size<1>(typename GemmMainloop::TileShape{});

/// Host-built launch arguments for one generated GemmDesc.
struct GemmInvocation {
  GemmProblem problem;
  GemmMainloop::Params mainloop;
  GemmEpilogue::Params epilogue;
  int tiles_m;
  int tiles_n;
  /// How many chunks §2.4's Split cut this GEMM's `k` into. The chunks are
  /// consecutive invocations built on the host, each a K-offset view of A/B
  /// writing its own partial, so the body needs no CUTLASS parameter surgery.
  int chunks = 1;
};

template <class Arch, class SmemUnion, int Threads>
struct GemmStageTaskBody {
  using SharedStorage = GemmMainloop::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = Threads == kGemmThreads;
  static constexpr char const* kLogicalA = "(M,K), strides (K,1)";
  static constexpr char const* kLogicalB = "(N,K), strides (K,1)";

  /// One output tile per CTA: `blockIdx.x` is the task index, decomposed
  /// N-major so a one-M-tile problem keeps today's `blockIdx.x == tile_n`.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    return {TaskOwnershipKind::kTilePerBlock,
            invocation.tiles_m * invocation.tiles_n * invocation.chunks};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    using namespace cute;
    auto const* table = static_cast<GemmInvocation const*>(p.gemms);
    int task = static_cast<int>(blockIdx.x);
    int tiles = table[stage.gemm].tiles_m * table[stage.gemm].tiles_n;
    if (task >= tiles * table[stage.gemm].chunks) return;
    int chunk = task / tiles;
    auto const& invocation = table[stage.gemm + chunk];
    task -= chunk * tiles;
    int tile_n = task % invocation.tiles_n;
    int tile_m = task / invocation.tiles_n;
    constexpr auto tile_shape = typename GemmMainloop::TileShape{};
    auto [M, N, K, L] = invocation.problem;
    Tensor matrix_a = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_A),
                                  make_shape(M, K, L), invocation.mainloop.dA);
    Tensor matrix_b = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_B),
                                  make_shape(N, K, L), invocation.mainloop.dB);
    auto block_coord = make_coord(tile_m, tile_n, _, 0);
    Tensor gA = local_tile(matrix_a(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<_1, X, _1>{});
    Tensor gB = local_tile(matrix_b(_, _, 0), tile_shape,
                           take<0, 3>(block_coord), Step<X, _1, _1>{});
    auto residue = make_tuple(M - size<0>(gA) * tile_m,
                              N - size<0>(gB) * tile_n,
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
    epilogue(invocation.problem, tile_shape, make_coord(tile_m, tile_n, 0, 0),
             accum, tiled_mma, residue, static_cast<int>(threadIdx.x), shared);
    __syncthreads();
  }
};

}  // namespace tilemega::codegen
