// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 GEMM TaskBody.  Handwritten body; the problem shape
// arrives as a generated GemmDesc, never as a constant here. On xcore1000 the
// GEMM invocation ABI is TileMega-owned POD (Backend/Mx/ServingGemmConfig.h)
// and only the serving runtime exists.
#pragma once

#include <tilemega/Backend/Mx/ServingGemmConfig.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <type_traits>
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Codegen/tasks/DwPwFusedTaskBody.h>
#endif

#ifndef TILEMEGA_SERVING_RUNTIME
#define TILEMEGA_SERVING_RUNTIME 0
#endif
#if !TILEMEGA_SERVING_RUNTIME
#error "xcore1000 builds only the serving runtime; the legacy CUTLASS GEMM path is excluded"
#endif
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#if TILEMEGA_TRACE_PHASE
#include <tilemega/Codegen/tasks/PhaseGemmMainloop.cuh>
#endif

#ifndef TILEMEGA_FUSION_SHARED_EPILOGUE
#define TILEMEGA_FUSION_SHARED_EPILOGUE 1
#endif

#include <tilemega/Codegen/tasks/ServingGemvTaskBody.h>

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

// R8 BE-2: the variants are file scope, so they read the Plan's architecture
// from the macro codegen emits rather than from a body's template parameter.
#ifndef TILEMEGA_ARCH_ID
#define TILEMEGA_ARCH_ID 7000
#endif
using GemmVariantArch = typename arch::ArchFromId<TILEMEGA_ARCH_ID>::type;

template<class Arch,int M,int N,int K,int S>
struct ServingGemvConfig : backend::ServingGemmConfig<Arch,16,32,K,2> {
  static constexpr bool kShapeLegal=solver::ServingGemvShapeLegal(M,N,K,S);
  static constexpr int kSharedBytes=solver::ServingGemvSmemBytes(M,N);
};
template <int Variant>
struct GemmVariant;

template <bool Serving, class Mainloop, int M, int N, int K, int S, bool Gemv=false>
struct GemmVariantStorage {
  using type = typename Mainloop::SharedStorage;
};
template <class Mainloop, int M, int N, int K, int S, bool Gemv>
struct GemmVariantStorage<true, Mainloop, M, N, K, S, Gemv> {
  struct alignas(16) type {
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    static constexpr auto mainloop_bytes = Gemv ? solver::ServingGemvSmemBytes(M,N)
        : solver::DmServingBF16SmemBytes(M,N,K,S);
    static constexpr auto chain_bytes = (M*N+2*M)*sizeof(float);
    static constexpr auto shared_epilogue_bytes = backend::ServingEpilogueScratchBytes(M,N,TILEMEGA_EP_PARALLEL_ARGMAX);
    static constexpr auto epilogue_bytes = chain_bytes>shared_epilogue_bytes?chain_bytes:shared_epilogue_bytes;
    unsigned char bytes[mainloop_bytes>epilogue_bytes?mainloop_bytes:epilogue_bytes];
#else
    unsigned char bytes[Gemv ? solver::ServingGemvSmemBytes(M,N) :
        backend::ServingGemmConfig<GemmVariantArch,M,(N<32?32:N),K,S>::kSharedBytes];
#endif
  };
};

#ifndef TILEMEGA_GEMM_IMPL_0
#define TILEMEGA_GEMM_IMPL_0 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_1
#define TILEMEGA_GEMM_IMPL_1 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_2
#define TILEMEGA_GEMM_IMPL_2 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_3
#define TILEMEGA_GEMM_IMPL_3 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_4
#define TILEMEGA_GEMM_IMPL_4 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_5
#define TILEMEGA_GEMM_IMPL_5 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_6
#define TILEMEGA_GEMM_IMPL_6 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_7
#define TILEMEGA_GEMM_IMPL_7 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_8
#define TILEMEGA_GEMM_IMPL_8 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_9
#define TILEMEGA_GEMM_IMPL_9 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_10
#define TILEMEGA_GEMM_IMPL_10 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_11
#define TILEMEGA_GEMM_IMPL_11 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_12
#define TILEMEGA_GEMM_IMPL_12 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_13
#define TILEMEGA_GEMM_IMPL_13 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_14
#define TILEMEGA_GEMM_IMPL_14 0
#endif
#ifndef TILEMEGA_GEMM_IMPL_15
#define TILEMEGA_GEMM_IMPL_15 0
#endif
// Which rule a dense tile must satisfy depends on the body that will run it.
// A DM build executes dense GEMM through ServingDmGemm
// (ServingGemmTaskBody::Run), whose legal shapes are DmServingBF16ShapeLegal
// -- k in {32,64,128}. Asserting the non-DM rule, k in {64,128}, rejected
// shapes the MoE solver had legitimately selected: a Qwen3-MoE-24L decode
// probe died on static_assert for ServingGemmConfig<Xcore1000,16,64,32,2>.
// The predicate is asserted rather than the config swapped, because
// ServingDmGemmConfig lives in Backend/ServingGemm.h, which pulls CUTLASS
// headers that a DM plan's include set does not carry.
template <int M, int N, int K, int S>
constexpr bool kDenseGemmShapeLegal =
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    solver::DmServingBF16ShapeLegal(M, N, K, S);
#else
    solver::ServingBF16ShapeLegal(M, N, K, S);
#endif

#define TILEMEGA_DEFINE_GEMM_VARIANT(index, M, N, K, S)                     \
  template <>                                                               \
  struct GemmVariant<index> {                                               \
    using Impl = std::conditional_t<TILEMEGA_GEMM_IMPL_##index != 0,      \
          ServingGemvConfig<GemmVariantArch,M,N,K,S>,                       \
          backend::ServingGemmConfig<GemmVariantArch,M,N,K,S>>;            \
    static_assert(TILEMEGA_GEMM_IMPL_##index != 0 ? Impl::kShapeLegal       \
                      : kDenseGemmShapeLegal<M,N,K,S>,                      \
                  "the selected GEMM tile shape is not a legal candidate; " \
                  "query the serving closed forms before compiling");      \
    using Mainloop = typename Impl::Mainloop;                               \
    using Epilogue = typename Impl::Epilogue;                               \
    using SharedStorage = typename GemmVariantStorage<                     \
        TILEMEGA_SERVING_RUNTIME != 0, Mainloop, M, N, K, S, TILEMEGA_GEMM_IMPL_##index != 0>::type;         \
    static constexpr bool kGemv = TILEMEGA_GEMM_IMPL_##index != 0;       \
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
/// (m, n, k, batch): `auto [m, n, k, batch] = invocation.problem`.
struct GemmProblem {
  int m, n, k, l;
};
// B is PyTorch's contiguous [N,K] weight: K is the unit-stride dimension and
// only the row pitch is carried.
static_assert(std::is_same_v<typename GemmMainloop::StrideB, backend::RowStride> &&
                  std::is_same_v<typename GemmMainloop::StrideA, backend::RowStride>,
              "serving operands are row-major with contiguous K");
inline constexpr int kGemmThreads = GemmImpl::kThreads;
inline constexpr int kGemmTileM = GemmMainloop::TileShape::kM;
inline constexpr int kGemmTileN = GemmMainloop::TileShape::kN;
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
          sizeof(typename V::SharedStorage)};
}

// Each generated serving plan may instantiate a different set of GEMM tiles.
// Give the host table internal linkage: an externally linked inline variable
// becomes GNU-unique in a shared object and is then coalesced across the
// prefill and decode libraries even when both are opened with RTLD_LOCAL.
static inline constexpr GemmVariantInfo kGemmVariantInfo[] = {
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
  typename GemmVariant<0>::SharedStorage v0;
#if TILEMEGA_GEMM_VARIANT_COUNT > 1
  typename GemmVariant<1>::SharedStorage v1;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
  typename GemmVariant<2>::SharedStorage v2;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
  typename GemmVariant<3>::SharedStorage v3;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
  typename GemmVariant<4>::SharedStorage v4;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
  typename GemmVariant<5>::SharedStorage v5;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
  typename GemmVariant<6>::SharedStorage v6;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
  typename GemmVariant<7>::SharedStorage v7;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
  typename GemmVariant<8>::SharedStorage v8;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
  typename GemmVariant<9>::SharedStorage v9;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
  typename GemmVariant<10>::SharedStorage v10;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
  typename GemmVariant<11>::SharedStorage v11;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
  typename GemmVariant<12>::SharedStorage v12;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
  typename GemmVariant<13>::SharedStorage v13;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
  typename GemmVariant<14>::SharedStorage v14;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
  typename GemmVariant<15>::SharedStorage v15;
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

/// Host-built launch arguments for one generated GemmDesc. Split-K partials
/// travel through `serving_partial`; the serving body has no partial epilogue
/// collective.
struct GemmInvocation {
  GemmProblem problem;
  GemmMainloopOperands mainloop;
  GemmEpilogue::Params epilogue;
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
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
  backend::ServingEpilogueOp serving_op = backend::ServingEpilogueOp::kStore;
  int* serving_argmax_index = nullptr;
  int serving_output_stride = 0;
  int serving_partial_stride = 0;
  float* serving_partial = nullptr;
  std::uint32_t serving_weight_buffer = 0xffffffffu;
  int serving_k_begin = 0;
  ModelElement const* serving_weight_base = nullptr;
  int serving_k_total_full = 0;
  int serving_tile_k = 0;
  float const* serving_norm_ss = nullptr;
  float* serving_ss_out = nullptr;
  PhaseGateDesc serving_phase_gate{};
  std::uint8_t serving_phase_class = 0; // qkv, o, gate/up, down, lm_head
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  DepthwiseConvOperands fused_depthwise{};
  std::uint32_t fused_program = kDmNoIndex;
  bool dm_enabled = true;
  std::uint32_t dm_gemm = 0;
  DmGemmAccess access{};
  DmEpilogueChain chain{};
  ConvDesc const* convolutions = nullptr;
  backend::ConvIterationGeometry conv_iteration{};
  void const* binding = nullptr;
  void const* rows = nullptr;
  float const* a_scale = nullptr;
  DmBufferView dm_buffers{};
#endif
};

#ifndef TILEMEGA_STAGE_DESC_CACHE
#define TILEMEGA_STAGE_DESC_CACHE (TILEMEGA_SERVING_RUNTIME && !TILEMEGA_PAGED)
#endif
#if TILEMEGA_STAGE_DESC_CACHE
/// MX3-4: a CTA's copy of the next stage's descriptor and of the GEMM
/// invocation its first task of that stage reads. Measured on Qwen3-0.6B B1
/// (TILEMEGA_DIAG_* plans), a task spends ~2.7 us (GEMM) or ~1.8 us
/// (split-K combine) walking dependent descriptor loads before it touches
/// data. The descriptors are read-only during a launch, so the second wave
/// copies them while thread 0 waits at the grid barrier (PrefetchStageDesc);
/// each copy is keyed by the global entry it holds, and a lookup that does
/// not match reads the table as before.
// Raw storage (both descriptors have default member initializers). With the
// launch's Params block (288 bytes), read by every descriptor lookup of every
// task: the L1 kernels copy it once per step and run from the copy.
struct StageDescCache {
  alignas(16) unsigned char stage[sizeof(StageDesc)];
  alignas(16) unsigned char invocation[sizeof(GemmInvocation)];
  alignas(16) unsigned char params[sizeof(Params)];
  StageDesc const* stage_source;
  GemmInvocation const* invocation_source;
};
/// MX4 prompt §5.3: the cache is the first kStageCacheBytes of the kernel's
/// dynamic shared memory, ahead of the TaskSmem union (ModelHarness.cuh
/// KernelTaskSmem), and counted in its size: as static __shared__ (672 B)
/// it was invisible to the resource estimate.
inline constexpr std::size_t kStageCacheBytes = (sizeof(StageDescCache) + 15) / 16 * 16;
extern __shared__ __align__(16) unsigned char tm_serving_dynamic_smem[];
__device__ __forceinline__ StageDescCache& StageCache() {
  return *reinterpret_cast<StageDescCache*>(tm_serving_dynamic_smem);
}
__device__ __forceinline__ void StageDescCacheReset() {
  if (threadIdx.x == 0) {
    StageCache().stage_source = nullptr;
    StageCache().invocation_source = nullptr;
  }
}
/// All threads copy; the caller synchronizes before and after.
__device__ inline Params const* CacheParams(Params const* source) {
  static_assert(sizeof(Params) % 4 == 0);
  using Word = __attribute__((address_space(1))) unsigned const;
  for (int w = int(threadIdx.x); w < int(sizeof(Params) / 4); w += int(blockDim.x))
    reinterpret_cast<unsigned*>(StageCache().params)[w] = ((Word*)source)[w];
  return reinterpret_cast<Params const*>(StageCache().params);
}
#else
inline constexpr std::size_t kStageCacheBytes = 0;
#endif
__device__ __forceinline__ StageDesc const& ServingStage(Params const& p, unsigned index) {
#if TILEMEGA_STAGE_DESC_CACHE
  if (StageCache().stage_source == p.stages + index)
    return *reinterpret_cast<StageDesc const*>(StageCache().stage);
#endif
  return p.stages[index];
}
__device__ __forceinline__ GemmInvocation const& ServingInvocation(Params const& p, unsigned index) {
  auto const* entry = static_cast<GemmInvocation const*>(p.gemms) + index;
#if TILEMEGA_STAGE_DESC_CACHE
  if (StageCache().invocation_source == entry)
    return *reinterpret_cast<GemmInvocation const*>(StageCache().invocation);
#endif
  return *entry;
}
#if TILEMEGA_STAGE_DESC_CACHE
/// Run by every thread once its CTA has finished the current stage (inside
/// the grid barrier, between its two CTA barriers); only the second wave
/// works, so the first wave's thread 0 is free to arrive and wait.
__device__ inline void PrefetchStageDesc(Params const& p, unsigned next) {
  int const lane = int(threadIdx.x) - 64;
  if (lane < 0 || next >= p.stage_count) return;
  static_assert(sizeof(StageDesc) % 4 == 0 && sizeof(GemmInvocation) % 4 == 0);
  using Word = __attribute__((address_space(1))) unsigned const;
  auto const* stage = p.stages + next;
  auto copy = [&](void* to, void const* from, int words) {
    for (int w = lane; w < words; w += 64)
      static_cast<unsigned*>(to)[w] = ((Word*)from)[w];
  };
  copy(StageCache().stage, stage, int(sizeof(StageDesc) / 4));
  auto const* table = static_cast<GemmInvocation const*>(p.gemms);
  GemmInvocation const* entry = nullptr;
  if (stage->kind == TaskKind::kGemmCombine) {
    entry = table + stage->gemm;
  } else if (stage->kind == TaskKind::kGemm) {
    // The first task this CTA runs, as GemmStageTaskBody's stage loop and
    // RunLogicalTask map it.
    auto const& base = table[stage->gemm];
    int const tiles = base.tiles_m * base.tiles_n;
    int const task = PlacedBlock();
    if (task < tiles * base.chunks)
      entry = table + stage->gemm + DecodeSplitTask(task, tiles, base.chunks).chunk;
  }
  if (entry) copy(StageCache().invocation, entry, int(sizeof(GemmInvocation) / 4));
  if (lane == 0) {
    StageCache().stage_source = stage;
    StageCache().invocation_source = entry;
  }
}
#endif

/// The exact dot product behind one output element. A BF16 product is exact in
/// FP64, so the only error left is the FP64 summation, some 2^-40 of the FP32
/// one; it settles the BF16 rounding the FP32 accumulator could not.
__device__ inline float ExactGemmElement(GemmInvocation const& invocation,
                                         int m, int n) {
  int const k_total = invocation.k_total;
  auto const* a = invocation.mainloop.ptr_A + std::int64_t(m) * invocation.mainloop.dA.row;
  auto const* b = invocation.mainloop.ptr_B + std::int64_t(n) * invocation.mainloop.dB.row;
  double sum = 0.0;
  for (int k = 0; k < k_total; ++k)
    sum += static_cast<double>(static_cast<float>(a[k])) *
           static_cast<double>(static_cast<float>(b[k]));
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
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  template<class V>
  struct FusedRunner {
    GemmInvocation const& invocation;
    ServingGemmOperands operands;
    int tile_m,tile_n;
    char* shared;
    template<int Channels,class DwProgram,class Spec>
    __device__ void Run() const {
      auto p=DwPwFusedOperands{invocation.fused_depthwise,operands};
      using Partial=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kDense,1,DmRounding::kFP32>;
      if(invocation.chunks>1) {
        p.pointwise.output=reinterpret_cast<tilemega::Bf16*>(operands.partial);
        p.pointwise.output_stride=operands.partial_stride;p.pointwise.chain={};
        p.pointwise.chain.store_rounding=DmRounding::kFP32;p.pointwise.access.write={};
        p.pointwise.dm_partial_rows=true;
        using Body=DwPwFusedTaskBody<Arch,Channels,V::kTileM,V::kTileN,V::kTileK,V::kStages,DwProgram,Partial>;
        Body::template Run<TILEMEGA_NONPAGED_TILED!=0>(p,tile_m,tile_n,shared);
      }else {
        using Body=DwPwFusedTaskBody<Arch,Channels,V::kTileM,V::kTileN,V::kTileK,V::kStages,DwProgram,Spec>;
        Body::template Run<TILEMEGA_NONPAGED_TILED!=0>(p,tile_m,tile_n,shared);
      }
    }
  };
  template <class Body>
  struct DmRunner {
    ServingGemmOperands const& operands;
    int tile_m, tile_n;
    char* shared;
    template <class Spec>
    __device__ void Run() const {
      using Gate = backend::DmGateShape<typename Spec::Chain>;
      if constexpr (!Gate::template kFits<Body::kTileColumns>) {
        maca::Trap();
      } else Body::template RunDm<Spec>(operands, tile_m, tile_n, shared);
    }
  };
#endif
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
                                 char* shared, ModelElement* tile_output = nullptr TILEMEGA_PHASE_ARG
#if TILEMEGA_TRACE_TASK
                                 ,ServingTaskProfile* profile=nullptr
#endif
                                 ) {
#if TILEMEGA_SERVING_RUNTIME
    (void)tile_output;
    using V = GemmVariant<Variant>;
    using Body = std::conditional_t<V::kGemv,
        ServingGemvTaskBody<Arch,V::kTileN,V::kTileK,TILEMEGA_NONPAGED_TILED!=0>,
        ServingGemmTaskBody<Arch,V::kTileM,V::kTileN,V::kTileK,V::kStages>>;
    auto [m, n, k, batch] = invocation.problem;
    (void)batch;
    ServingGemmOperands operands;
#if TILEMEGA_TRACE_TASK
    operands.profile=profile;
    if(profile)profile->bytes=2ull*min(V::kTileN,n-(local%invocation.tiles_n)*V::kTileN)*k;
#endif
    operands.a = invocation.mainloop.ptr_A;
    operands.b = invocation.mainloop.ptr_B;
    operands.residual = invocation.epilogue.ptr_C;
    operands.output = invocation.epilogue.ptr_D;
    operands.partial = invocation.serving_partial;
    operands.argmax_value = reinterpret_cast<float*>(invocation.epilogue.ptr_D);
    operands.argmax_index = invocation.serving_argmax_index;
    operands.m = m;
    operands.n = n;
    operands.k_total = k;
    operands.k_count = k;
#if TILEMEGA_NONPAGED_TILED
    operands.a-=invocation.serving_k_begin;
    operands.weight_base=invocation.serving_weight_base;
    operands.k_total=invocation.k_total;
    operands.k_total_full=invocation.serving_k_total_full;
    operands.k_begin=invocation.serving_k_begin;
#endif
    operands.output_stride = invocation.serving_output_stride;
    operands.partial_stride = invocation.serving_partial_stride;
    operands.a_row_stride = invocation.k_total;
    operands.b_row_stride = invocation.k_total;
    operands.epilogue = invocation.chunks > 1
        ? backend::ServingEpilogueOp::kPartial : invocation.serving_op;
    operands.norm_ss=invocation.serving_norm_ss;
    operands.ss_out=invocation.serving_ss_out;
    operands.norm_k=invocation.k_total;
    operands.norm_eps=TILEMEGA_NORM_EPSILON;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    operands.access=invocation.access;
    operands.chain=invocation.chain;
    operands.convolutions=invocation.convolutions;
    operands.conv_iteration=invocation.conv_iteration;
    if(invocation.access.a==DmAAccess::kIm2Col) {
      operands.a=invocation.mainloop.ptr_A;operands.b=invocation.mainloop.ptr_B;
      operands.weight_base=invocation.serving_weight_base;
      operands.k_total=invocation.k_total;
      operands.k_total_full=invocation.serving_k_total_full;
      operands.k_begin=invocation.serving_k_begin;
    }
    if(invocation.access.a==DmAAccess::kRowGather ||
       invocation.access.b==DmBAccess::kExpertIndirect) {
      operands.a=invocation.mainloop.ptr_A-invocation.serving_k_begin;
      operands.b=invocation.mainloop.ptr_B-invocation.serving_k_begin;
      operands.weight_base=invocation.serving_weight_base;
      operands.k_total=invocation.k_total;
      operands.k_total_full=invocation.serving_k_total_full;
      operands.k_begin=invocation.serving_k_begin;
    }
    operands.binding=invocation.binding; operands.rows=invocation.rows;
    operands.a_scale=invocation.a_scale; operands.dm_buffers=invocation.dm_buffers;
    operands.a_row_stride=static_cast<int>(invocation.mainloop.dA.row);
    if(invocation.fused_program!=kDmNoIndex) {
      operands.k_total=invocation.k_total;operands.k_begin=invocation.serving_k_begin;
      operands.b=invocation.mainloop.ptr_B-invocation.serving_k_begin;
      operands.weight_base=invocation.serving_weight_base;
      if(!DispatchDmVariantFused<Variant>(invocation.fused_program,FusedRunner<V>{invocation,operands,
          local/invocation.tiles_n,local%invocation.tiles_n,shared}))maca::Trap();
      return;
    }
    if (invocation.chunks == 1 && invocation.dm_enabled) {
      DispatchDmVariantEpilogue<Variant>(invocation.dm_gemm, DmRunner<Body>{
          operands, local / invocation.tiles_n, local % invocation.tiles_n, shared});
      return;
    }
#endif
    Body::Run(operands, local / invocation.tiles_n,
              local % invocation.tiles_n, shared);
    return;
#endif
  }

  __device__ static void RunLogicalTask(Params const& p,
                                        StageDesc const& stage,
                                        SmemUnion& smem, int task TILEMEGA_PHASE_ARG
#if TILEMEGA_TRACE_TASK
                                        ,ServingTaskProfile* profile=nullptr
#endif
                                        ) {
    auto const& base = ServingInvocation(p, stage.gemm);
    int const tiles = base.tiles_m * base.tiles_n;
    // CG appends the split axis: row-major task ids are (m,n,chunk).
    auto const coordinate = DecodeSplitTask(task,tiles,base.chunks);
    int const local = coordinate.tile;
    auto const& invocation = ServingInvocation(p, stage.gemm + coordinate.chunk);
    char* shared = reinterpret_cast<char*>(&smem.gemm);
#if TILEMEGA_GEMM_VARIANT_COUNT == 1
    RunTask<0>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        );
#else
    switch (invocation.variant) {
      case 0: RunTask<0>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
      case 1: RunTask<1>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#if TILEMEGA_GEMM_VARIANT_COUNT > 2
      case 2: RunTask<2>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 3
      case 3: RunTask<3>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 4
      case 4: RunTask<4>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 5
      case 5: RunTask<5>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 6
      case 6: RunTask<6>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 7
      case 7: RunTask<7>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 8
      case 8: RunTask<8>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 9
      case 9: RunTask<9>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 10
      case 10: RunTask<10>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 11
      case 11: RunTask<11>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 12
      case 12: RunTask<12>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 13
      case 13: RunTask<13>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 14
      case 14: RunTask<14>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
#if TILEMEGA_GEMM_VARIANT_COUNT > 15
      case 15: RunTask<15>(invocation, local, shared, nullptr TILEMEGA_PHASE_PASS
#if TILEMEGA_TRACE_TASK
        ,profile
#endif
        ); break;
#endif
      default: break;
    }
#endif
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    auto const& base = ServingInvocation(p, stage.gemm);
    int const tiles = base.tiles_m * base.tiles_n;
    int const count = tiles * base.chunks;
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
