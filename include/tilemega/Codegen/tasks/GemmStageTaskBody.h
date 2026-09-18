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
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#if TILEMEGA_TRACE_PHASE
#include <tilemega/Codegen/tasks/PhaseGemmMainloop.cuh>
#endif

#ifndef TILEMEGA_FUSION_SHARED_EPILOGUE
#define TILEMEGA_FUSION_SHARED_EPILOGUE 1
#endif

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
#if TILEMEGA_GEMM_VARIANT_COUNT < 1 || TILEMEGA_GEMM_VARIANT_COUNT > 16
#error "TILEMEGA_GEMM_VARIANT_COUNT must be between 1 and 16"
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
// Fields 4..15 are provided by generated source; the defaults keep standalone
// TaskBody contract tests usable.
#ifndef TILEMEGA_GEMM_V4_TILE_M
#define TILEMEGA_GEMM_V4_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V4_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V4_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V4_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V5_TILE_M
#define TILEMEGA_GEMM_V5_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V5_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V5_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V5_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V6_TILE_M
#define TILEMEGA_GEMM_V6_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V6_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V6_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V6_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V7_TILE_M
#define TILEMEGA_GEMM_V7_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V7_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V7_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V7_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V8_TILE_M
#define TILEMEGA_GEMM_V8_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V8_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V8_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V8_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V9_TILE_M
#define TILEMEGA_GEMM_V9_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V9_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V9_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V9_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V10_TILE_M
#define TILEMEGA_GEMM_V10_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V10_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V10_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V10_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V11_TILE_M
#define TILEMEGA_GEMM_V11_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V11_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V11_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V11_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V12_TILE_M
#define TILEMEGA_GEMM_V12_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V12_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V12_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V12_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V13_TILE_M
#define TILEMEGA_GEMM_V13_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V13_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V13_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V13_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V14_TILE_M
#define TILEMEGA_GEMM_V14_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V14_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V14_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V14_STAGES TILEMEGA_GEMM_STAGES
#endif
#ifndef TILEMEGA_GEMM_V15_TILE_M
#define TILEMEGA_GEMM_V15_TILE_M TILEMEGA_GEMM_TILE_M
#define TILEMEGA_GEMM_V15_TILE_N TILEMEGA_GEMM_TILE_N
#define TILEMEGA_GEMM_V15_TILE_K TILEMEGA_GEMM_TILE_K
#define TILEMEGA_GEMM_V15_STAGES TILEMEGA_GEMM_STAGES
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
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
TILEMEGA_DEFINE_GEMM_VARIANT(4, TILEMEGA_GEMM_V4_TILE_M, TILEMEGA_GEMM_V4_TILE_N, TILEMEGA_GEMM_V4_TILE_K, TILEMEGA_GEMM_V4_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
TILEMEGA_DEFINE_GEMM_VARIANT(5, TILEMEGA_GEMM_V5_TILE_M, TILEMEGA_GEMM_V5_TILE_N, TILEMEGA_GEMM_V5_TILE_K, TILEMEGA_GEMM_V5_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
TILEMEGA_DEFINE_GEMM_VARIANT(6, TILEMEGA_GEMM_V6_TILE_M, TILEMEGA_GEMM_V6_TILE_N, TILEMEGA_GEMM_V6_TILE_K, TILEMEGA_GEMM_V6_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
TILEMEGA_DEFINE_GEMM_VARIANT(7, TILEMEGA_GEMM_V7_TILE_M, TILEMEGA_GEMM_V7_TILE_N, TILEMEGA_GEMM_V7_TILE_K, TILEMEGA_GEMM_V7_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
TILEMEGA_DEFINE_GEMM_VARIANT(8, TILEMEGA_GEMM_V8_TILE_M, TILEMEGA_GEMM_V8_TILE_N, TILEMEGA_GEMM_V8_TILE_K, TILEMEGA_GEMM_V8_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
TILEMEGA_DEFINE_GEMM_VARIANT(9, TILEMEGA_GEMM_V9_TILE_M, TILEMEGA_GEMM_V9_TILE_N, TILEMEGA_GEMM_V9_TILE_K, TILEMEGA_GEMM_V9_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
TILEMEGA_DEFINE_GEMM_VARIANT(10, TILEMEGA_GEMM_V10_TILE_M, TILEMEGA_GEMM_V10_TILE_N, TILEMEGA_GEMM_V10_TILE_K, TILEMEGA_GEMM_V10_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
TILEMEGA_DEFINE_GEMM_VARIANT(11, TILEMEGA_GEMM_V11_TILE_M, TILEMEGA_GEMM_V11_TILE_N, TILEMEGA_GEMM_V11_TILE_K, TILEMEGA_GEMM_V11_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
TILEMEGA_DEFINE_GEMM_VARIANT(12, TILEMEGA_GEMM_V12_TILE_M, TILEMEGA_GEMM_V12_TILE_N, TILEMEGA_GEMM_V12_TILE_K, TILEMEGA_GEMM_V12_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
TILEMEGA_DEFINE_GEMM_VARIANT(13, TILEMEGA_GEMM_V13_TILE_M, TILEMEGA_GEMM_V13_TILE_N, TILEMEGA_GEMM_V13_TILE_K, TILEMEGA_GEMM_V13_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
TILEMEGA_DEFINE_GEMM_VARIANT(14, TILEMEGA_GEMM_V14_TILE_M, TILEMEGA_GEMM_V14_TILE_N, TILEMEGA_GEMM_V14_TILE_K, TILEMEGA_GEMM_V14_STAGES);
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
TILEMEGA_DEFINE_GEMM_VARIANT(15, TILEMEGA_GEMM_V15_TILE_M, TILEMEGA_GEMM_V15_TILE_N, TILEMEGA_GEMM_V15_TILE_K, TILEMEGA_GEMM_V15_STAGES);
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
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
    MakeGemmVariantInfo<4>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
    MakeGemmVariantInfo<5>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
    MakeGemmVariantInfo<6>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
    MakeGemmVariantInfo<7>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
    MakeGemmVariantInfo<8>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
    MakeGemmVariantInfo<9>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
    MakeGemmVariantInfo<10>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
    MakeGemmVariantInfo<11>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
    MakeGemmVariantInfo<12>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
    MakeGemmVariantInfo<13>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
    MakeGemmVariantInfo<14>(),
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
    MakeGemmVariantInfo<15>(),
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
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
  typename GemmVariant<4>::Mainloop::SharedStorage v4;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
  typename GemmVariant<5>::Mainloop::SharedStorage v5;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
  typename GemmVariant<6>::Mainloop::SharedStorage v6;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
  typename GemmVariant<7>::Mainloop::SharedStorage v7;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
  typename GemmVariant<8>::Mainloop::SharedStorage v8;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
  typename GemmVariant<9>::Mainloop::SharedStorage v9;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
  typename GemmVariant<10>::Mainloop::SharedStorage v10;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
  typename GemmVariant<11>::Mainloop::SharedStorage v11;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
  typename GemmVariant<12>::Mainloop::SharedStorage v12;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
  typename GemmVariant<13>::Mainloop::SharedStorage v13;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
  typename GemmVariant<14>::Mainloop::SharedStorage v14;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
  typename GemmVariant<15>::Mainloop::SharedStorage v15;
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
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
using PartialEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    float, typename GemmEpilogue::StrideC, typename GemmEpilogue::StrideD,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    typename GemmEpilogue::EpilogueSchedule>;
#endif
struct GemmInvocation {
  GemmProblem problem;
  GemmMainloopOperands mainloop;
  GemmEpilogue::Params epilogue;
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
  PartialEpilogue::Params partial_epilogue;
  ModelElement const* residual = nullptr;
  float residual_beta = 0;
#endif
  int tiles_m;
  int tiles_n;
  int tile_m;
  int tile_n;
  /// How many chunks §2.4's Split cut this GEMM's `k` into. The chunks are
  /// consecutive invocations built on the host, each a K-offset view of A/B
  /// writing its own partial, so the body needs no CUTLASS parameter surgery.
  int chunks = 1;
  /// Which compiled tile shape runs this GEMM.  Host-assigned, so the DP's
  /// per-operator plan reaches the device without a second task kind.
  int variant = 0;
  /// The undivided K. `problem.k` is this chunk's slice of it, so a consumer
  /// that has to reproduce the whole dot product -- the combiner refining an
  /// element -- cannot recover it from the chunk alone.
  int k_total = 0;
};

/// The exact dot product behind one output element. A BF16 product is exact in
/// FP64, so the only error left is the FP64 summation, some 2^-40 of the FP32
/// one; it settles the BF16 rounding the FP32 accumulator could not.
__device__ inline float ExactGemmElement(GemmInvocation const& invocation,
                                         int m, int n) {
  using namespace cute;
  auto [M, N, K, L] = invocation.problem;
  int const k_total = invocation.k_total;
  Tensor matrix_a = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_A),
                                make_shape(M, k_total, L), invocation.mainloop.dA);
  Tensor matrix_b = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_B),
                                make_shape(N, k_total, L), invocation.mainloop.dB);
  double sum = 0.0;
  for (int k = 0; k < k_total; ++k)
    sum += static_cast<double>(static_cast<float>(matrix_a(m, k, 0))) *
           static_cast<double>(static_cast<float>(matrix_b(n, k, 0)));
  return static_cast<float>(sum);
}

/// Whether the FP32 accumulation error could have put `value` on the wrong
/// side of the BF16 rounding boundary. F-203: one element of the covered Llama
/// graph misses the boundary by 1.7 FP32 ulp, which no association of a 2048
/// term FP32 sum resolves.
__device__ inline bool NearRoundingBoundary(float value) {
  if (!isfinite(value)) return false;
  float const rounded = static_cast<float>(ModelElement(value));
  std::uint32_t const bits = __float_as_uint(rounded);
  // One step of the BF16 significand, away from zero: the pattern is the top
  // half of the FP32 one, so incrementing it moves by exactly one BF16 ulp.
  float const next = __uint_as_float(((bits >> 16) + 1u) << 16);
  float const ulp = fabsf(next - rounded);
  if (!isfinite(ulp) || ulp == 0.0f) return false;
  return fabsf(fabsf(value - rounded) - 0.5f * ulp) <
         TILEMEGA_MIDPOINT_GUARD * ulp;
}

/// The FP32 sum an element should be rounded from: the accumulated one, unless
/// it sits near a boundary the accumulation cannot be trusted to have chosen.
__device__ inline float RefinedGemmElement(GemmInvocation const& invocation,
                                           int m, int n, float sum,
                                           float bias = 0.0f) {
#if TILEMEGA_MIDPOINT_REFINE
  if (NearRoundingBoundary(sum + bias)) return ExactGemmElement(invocation, m, n);
#else
  (void)invocation; (void)m; (void)n; (void)bias;
#endif
  return sum;
}

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
    return {OwnershipOf(TaskKind::kGemm),
            invocation.tiles_m * invocation.tiles_n * invocation.chunks};
  }

  template <int Variant, bool SharedOutput = false>
  __device__ static void RunTask(GemmInvocation const& invocation, int local,
                                 char* shared, ModelElement* tile_output = nullptr TILEMEGA_PHASE_ARG) {
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
#if TILEMEGA_TRACE_PHASE
    RunPhaseMainloop<Mainloop>(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
                              static_cast<int>(threadIdx.x), shared, phase);
#else
    Mainloop mainloop;
    mainloop(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
             static_cast<int>(threadIdx.x), shared);
#endif
    TILEMEGA_PHASE_STAMP(3);
#if TILEMEGA_MIDPOINT_REFINE
    // An unsplit GEMM rounds here, so this is where an accumulator too close to
    // a BF16 boundary has to be settled; a split one rounds in the combiner.
    if (invocation.chunks == 1 && invocation.epilogue.thread.alpha == 1.0f) {
      auto owned = tiled_mma.get_thread_slice(int(threadIdx.x))
                       .partition_C(make_identity_tensor(take<0, 2>(tile_shape)));
      auto source = make_tensor(make_gmem_ptr(invocation.epilogue.ptr_C),
                                make_shape(M, N, L), invocation.epilogue.dC);
      float const beta = invocation.epilogue.thread.beta;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        int global_m = tile_m * size<0>(tile_shape) + get<0>(owned(i));
        int global_n = tile_n * size<1>(tile_shape) + get<1>(owned(i));
        if (global_m >= M || global_n >= N) continue;
        float bias = (beta != 0.0f && invocation.epilogue.ptr_C != nullptr)
                         ? beta * static_cast<float>(source(global_m, global_n, 0))
                         : 0.0f;
        accum(i) = RefinedGemmElement(invocation, global_m, global_n, accum(i), bias);
      }
    }
#endif
    if constexpr (SharedOutput) {
      static_assert(TILEMEGA_FUSION_SHARED_EPILOGUE || !SharedOutput,
                    "shared fusion epilogue is disabled");
      // A split partial cannot be rounded to ModelElement before combining.
      if (invocation.chunks != 1) { asm volatile("trap;"); return; }
      auto coordinates = make_identity_tensor(take<0, 2>(tile_shape));
      auto owned = tiled_mma.get_thread_slice(int(threadIdx.x)).partition_C(coordinates);
      auto source = make_tensor(make_gmem_ptr(invocation.epilogue.ptr_C),
                                make_shape(M, N, L), invocation.epilogue.dC);
      typename Epilogue::ThreadEpilogueOp op(invocation.epilogue.thread);
      CUTE_STATIC_ASSERT_V(size(owned) == size(accum));
      // The same thread operation preserves the graph's BF16 rounding boundary;
      // only the destination address space changes from CUTLASS's global store.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        int m = get<0>(owned(i)), n = get<1>(owned(i));
        int global_m = tile_m * size<0>(tile_shape) + m;
        int global_n = tile_n * size<1>(tile_shape) + n;
        if (global_m < M && global_n < N)
          tile_output[m * size<1>(tile_shape) + n] = op.is_source_needed()
              ? op(accum(i), source(global_m, global_n, 0)) : op(accum(i));
      }
      return;
    }
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
    if (invocation.chunks > 1) {
      PartialEpilogue epilogue(invocation.partial_epilogue);
      epilogue(invocation.problem, tile_shape, make_coord(tile_m, tile_n, 0, 0),
               accum, tiled_mma, residue, static_cast<int>(threadIdx.x), shared);
      return;
    }
#endif
    Epilogue epilogue(invocation.epilogue);
    epilogue(invocation.problem, tile_shape, make_coord(tile_m, tile_n, 0, 0),
             accum, tiled_mma, residue, static_cast<int>(threadIdx.x), shared);
  }

  __device__ static void RunLogicalTask(Params const& p,
                                        StageDesc const& stage,
                                        SmemUnion& smem, int task TILEMEGA_PHASE_ARG) {
    auto const* table = static_cast<GemmInvocation const*>(p.gemms);
    int const tiles = table[stage.gemm].tiles_m * table[stage.gemm].tiles_n;
    // CG appends the split axis: row-major task ids are (m,n,chunk).
    auto const coordinate = DecodeSplitTask(task,tiles,table[stage.gemm].chunks);
    int const local = coordinate.tile;
    auto const& invocation = table[stage.gemm + coordinate.chunk];
    char* shared = reinterpret_cast<char*>(&smem.gemm);
#if TILEMEGA_GEMM_VARIANT_COUNT == 1
    RunTask<0>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS);
#else
    switch (invocation.variant) {
      case 0: RunTask<0>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
      case 1: RunTask<1>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
      case 2: RunTask<2>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
      case 3: RunTask<3>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
      case 4: RunTask<4>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
      case 5: RunTask<5>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
      case 6: RunTask<6>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
      case 7: RunTask<7>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
      case 8: RunTask<8>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
      case 9: RunTask<9>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
      case 10: RunTask<10>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
      case 11: RunTask<11>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
      case 12: RunTask<12>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
      case 13: RunTask<13>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
      case 14: RunTask<14>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
      case 15: RunTask<15>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS); break;
#endif
      default: break;
    }
#endif
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    auto const* table = static_cast<GemmInvocation const*>(p.gemms);
    int const tiles = table[stage.gemm].tiles_m * table[stage.gemm].tiles_n;
    int const count = tiles * table[stage.gemm].chunks;
    // Grid-stride over the whole task space. `Ownership` exceeds the resident
    // grid whenever a narrow N tile meets a large split-K factor, and without
    // the stride those tasks are simply never run -- a silently wrong result,
    // not a launch error.
    for (int task = PlacedBlock(); task < count; task += gridDim.x) {
      RunLogicalTask(p, stage, smem, task);
      // The next iteration reuses `shared`, so the barrier is the loop's, not
      // the body's.
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
