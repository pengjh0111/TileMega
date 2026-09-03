// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 CUTLASS/CuTe GEMM TaskBody.  Handwritten body; the
// problem shape arrives as a generated GemmDesc, never as a constant here.
#pragma once

#include <tilemega/Backend/CutlassGemmCandidate.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

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

// P4.5 acceptance (b) asks what a *per-operator* granularity is worth, which
// needs more than one compiled tile shape in one megakernel.  The variant
// count defaults to 1 and every per-variant knob defaults to the global one
// above, so a default build is byte for byte the single-variant build it was.
#ifndef TILEMEGA_GEMM_VARIANT_COUNT
#define TILEMEGA_GEMM_VARIANT_COUNT 1
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT < 1 || TILEMEGA_GEMM_VARIANT_COUNT > 4
#error "TILEMEGA_GEMM_VARIANT_COUNT must be between 1 and 4"
#endif
#ifndef TILEMEGA_GEMM_V1_TILE_M
#define TILEMEGA_GEMM_V1_TILE_M TILEMEGA_GEMM_TILE_M
#endif
#ifndef TILEMEGA_GEMM_V1_TILE_N
#define TILEMEGA_GEMM_V1_TILE_N TILEMEGA_GEMM_TILE_N
#endif
#ifndef TILEMEGA_GEMM_V1_TILE_K
#define TILEMEGA_GEMM_V1_TILE_K TILEMEGA_GEMM_TILE_K
#endif
#ifndef TILEMEGA_GEMM_V1_STAGES
#define TILEMEGA_GEMM_V1_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V2_TILE_M
#define TILEMEGA_GEMM_V2_TILE_M TILEMEGA_GEMM_TILE_M
#endif
#ifndef TILEMEGA_GEMM_V2_TILE_N
#define TILEMEGA_GEMM_V2_TILE_N TILEMEGA_GEMM_TILE_N
#endif
#ifndef TILEMEGA_GEMM_V2_TILE_K
#define TILEMEGA_GEMM_V2_TILE_K TILEMEGA_GEMM_TILE_K
#endif
#ifndef TILEMEGA_GEMM_V2_STAGES
#define TILEMEGA_GEMM_V2_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V3_TILE_M
#define TILEMEGA_GEMM_V3_TILE_M TILEMEGA_GEMM_TILE_M
#endif
#ifndef TILEMEGA_GEMM_V3_TILE_N
#define TILEMEGA_GEMM_V3_TILE_N TILEMEGA_GEMM_TILE_N
#endif
#ifndef TILEMEGA_GEMM_V3_TILE_K
#define TILEMEGA_GEMM_V3_TILE_K TILEMEGA_GEMM_TILE_K
#endif
#ifndef TILEMEGA_GEMM_V3_STAGES
#define TILEMEGA_GEMM_V3_STAGES TILEMEGA_GEMM_STAGES
#endif
// The per-operator plan itself: which variant runs GEMM `i`, and how many
// chunks its `k` is cut into.  Both default to the model-wide knob, so an
// unplanned build is the uniform build.
#ifndef TILEMEGA_GEMM_VARIANT_OF
#define TILEMEGA_GEMM_VARIANT_OF(i) 0
#endif
#ifndef TILEMEGA_GEMM_SPLIT_OF
#define TILEMEGA_GEMM_SPLIT_OF(i) TILEMEGA_GEMM_SPLIT_K
#endif

template <int Variant>
struct GemmVariant;

#define TILEMEGA_DEFINE_GEMM_VARIANT(index, M, N, K, S)                     \
  template <>                                                               \
  struct GemmVariant<index> {                                               \
    using Impl = backend::GemmCandidate<M, N, K, S>;                        \
    static_assert(Impl::kShapeLegal,                                        \
                  "the selected GEMM tile shape is not a legal candidate; " \
                  "query backend::GemmCandidate::kShapeLegal before "       \
                  "compiling");                                             \
    using Mainloop = typename Impl::Mainloop;                               \
    using Epilogue = typename Impl::Epilogue;                               \
    static constexpr int kTileM = M;                                        \
    static constexpr int kTileN = N;                                        \
    static constexpr int kTileK = K;                                        \
    static constexpr int kStages = S;                                       \
  }

TILEMEGA_DEFINE_GEMM_VARIANT(0, TILEMEGA_GEMM_TILE_M, TILEMEGA_GEMM_TILE_N,
                             TILEMEGA_GEMM_TILE_K, TILEMEGA_GEMM_STAGES);
#if TILEMEGA_GEMM_VARIANT_COUNT > 1
TILEMEGA_DEFINE_GEMM_VARIANT(1, TILEMEGA_GEMM_V1_TILE_M, TILEMEGA_GEMM_V1_TILE_N,
                             TILEMEGA_GEMM_V1_TILE_K, TILEMEGA_GEMM_V1_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
TILEMEGA_DEFINE_GEMM_VARIANT(2, TILEMEGA_GEMM_V2_TILE_M, TILEMEGA_GEMM_V2_TILE_N,
                             TILEMEGA_GEMM_V2_TILE_K, TILEMEGA_GEMM_V2_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
TILEMEGA_DEFINE_GEMM_VARIANT(3, TILEMEGA_GEMM_V3_TILE_M, TILEMEGA_GEMM_V3_TILE_N,
                             TILEMEGA_GEMM_V3_TILE_K, TILEMEGA_GEMM_V3_STAGES);
#endif
#undef TILEMEGA_DEFINE_GEMM_VARIANT

using GemmImpl = GemmVariant<0>::Impl;
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
inline constexpr int kGemmVariantCount = TILEMEGA_GEMM_VARIANT_COUNT;

/// The host needs each variant's tiling to lay out its invocations, and the
/// megakernel needs the union of their shared storage; both are folded here so
/// no caller re-derives them from the macros.
struct GemmVariantInfo {
  int tile_m, tile_n, tile_k, stages;
  std::size_t smem_bytes;
};

template <int Variant>
inline constexpr GemmVariantInfo MakeGemmVariantInfo() {
  using V = GemmVariant<Variant>;
  static_assert(V::Impl::kThreads == kGemmThreads,
                "every GEMM variant must run at the harness thread count");
  // Every variant reads the same operands: the epilogue collective does not
  // depend on the tile at all, and the mainloop's element and stride types are
  // its template arguments, not its tiling.  Only `Mainloop::Params` itself is
  // a distinct nominal type per variant, which is why GemmInvocation carries
  // the operands rather than a lowered Params.
  static_assert(std::is_same_v<typename V::Epilogue, GemmEpilogue> &&
                    std::is_same_v<typename V::Mainloop::ElementA,
                                   typename GemmMainloop::ElementA> &&
                    std::is_same_v<typename V::Mainloop::ElementB,
                                   typename GemmMainloop::ElementB> &&
                    std::is_same_v<typename V::Mainloop::StrideA,
                                   typename GemmMainloop::StrideA> &&
                    std::is_same_v<typename V::Mainloop::StrideB,
                                   typename GemmMainloop::StrideB>,
                "GemmInvocation holds one operand set for every variant");
  return {V::kTileM, V::kTileN, V::kTileK, V::kStages,
          sizeof(typename V::Mainloop::SharedStorage)};
}

inline constexpr GemmVariantInfo kGemmVariantInfo[] = {
    MakeGemmVariantInfo<0>(),
#if TILEMEGA_GEMM_VARIANT_COUNT > 1
    MakeGemmVariantInfo<1>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
    MakeGemmVariantInfo<2>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
    MakeGemmVariantInfo<3>(),
#endif
};

/// §4.3: shared memory is a property of the whole kernel, so one variant's
/// tile shape costs every other operator the same SM slot.  Spelling that as a
/// union rather than a byte count keeps the alignment of the widest variant.
union GemmVariantSmem {
  typename GemmVariant<0>::Mainloop::SharedStorage v0;
#if TILEMEGA_GEMM_VARIANT_COUNT > 1
  typename GemmVariant<1>::Mainloop::SharedStorage v1;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
  typename GemmVariant<2>::Mainloop::SharedStorage v2;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
  typename GemmVariant<3>::Mainloop::SharedStorage v3;
#endif
};

/// The mainloop operands, spelled without the collective that consumes them:
/// `Mainloop::Params` is a member type of a tile-parameterised class, so it is
/// a different type per variant even though its fields are not.
struct GemmMainloopOperands {
  typename GemmMainloop::ElementA const* ptr_A;
  typename GemmMainloop::StrideA dA;
  typename GemmMainloop::ElementB const* ptr_B;
  typename GemmMainloop::StrideB dB;
};

/// Host-built launch arguments for one generated GemmDesc.
struct GemmInvocation {
  GemmProblem problem;
  GemmMainloopOperands mainloop;
  GemmEpilogue::Params epilogue;
  int tiles_m;
  int tiles_n;
  /// How many chunks §2.4's Split cut this GEMM's `k` into. The chunks are
  /// consecutive invocations built on the host, each a K-offset view of A/B
  /// writing its own partial, so the body needs no CUTLASS parameter surgery.
  int chunks = 1;
  /// Which compiled tile shape runs this GEMM.  Host-assigned, so the DP's
  /// per-operator plan reaches the device without a second task kind.
  int variant = 0;
};

template <class Arch, class SmemUnion, int Threads>
struct GemmStageTaskBody {
  using SharedStorage = GemmVariantSmem;
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

  template <int Variant>
  __device__ static void RunTask(GemmInvocation const& invocation, int local,
                                 char* shared) {
    using namespace cute;
    using Mainloop = typename GemmVariant<Variant>::Mainloop;
    using Epilogue = typename GemmVariant<Variant>::Epilogue;
    int tile_n = local % invocation.tiles_n;
    int tile_m = local / invocation.tiles_n;
    constexpr auto tile_shape = typename Mainloop::TileShape{};
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
    typename Mainloop::TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape));
    clear(accum);
    auto k_iter = make_coord_iterator(shape<2>(gA));
    Mainloop mainloop;
    mainloop(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
             static_cast<int>(threadIdx.x), shared);
    Epilogue epilogue(invocation.epilogue);
    epilogue(invocation.problem, tile_shape, make_coord(tile_m, tile_n, 0, 0),
             accum, tiled_mma, residue, static_cast<int>(threadIdx.x), shared);
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    auto const* table = static_cast<GemmInvocation const*>(p.gemms);
    int const tiles = table[stage.gemm].tiles_m * table[stage.gemm].tiles_n;
    int const count = tiles * table[stage.gemm].chunks;
    char* shared = reinterpret_cast<char*>(&smem.gemm);
    // Grid-stride over the whole task space. `Ownership` exceeds the resident
    // grid whenever a narrow N tile meets a large split-K factor, and without
    // the stride those tasks are simply never run -- a silently wrong result,
    // not a launch error.
    for (int task = PlacedBlock(); task < count; task += gridDim.x) {
      int chunk = task / tiles;
      auto const& invocation = table[stage.gemm + chunk];
      int local = task - chunk * tiles;
#if TILEMEGA_GEMM_VARIANT_COUNT == 1
      // A single-variant build must emit exactly the code it emitted before
      // variants existed, or every per-operator delta is measured against a
      // baseline this feature made slower.
      RunTask<0>(invocation, local, shared);
#else
      // Uniform across the stage: the variant is a property of the GEMM, not
      // of the tile, so the branch never diverges inside a CTA.
      switch (invocation.variant) {
        case 0: RunTask<0>(invocation, local, shared); break;
        case 1: RunTask<1>(invocation, local, shared); break;
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
        case 2: RunTask<2>(invocation, local, shared); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
        case 3: RunTask<3>(invocation, local, shared); break;
#endif
        default: break;
      }
#endif
      // The next iteration reuses `shared`, so the barrier is the loop's, not
      // the body's.
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
