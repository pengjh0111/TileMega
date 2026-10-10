// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Backend/ServingEpilogueScratch.h>
#include <tilemega/Target/ArchDispatch.h>
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <tilemega/Backend/ServingDmGemm.h>
#endif

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/matrix.h>
#include <type_traits>

namespace tilemega::backend {

/// Shared SM80-class implementation for every target with cp.async and BF16
/// tensor cores.  SM90 TMA/WGMMA and SM100 tcgen05 are future specializations;
/// they intentionally inherit this implementation until the task ABI supports
/// warp-specialized collectives.
template <class Arch, int TileM, int TileN, int TileK, int Stages>
struct ServingGemmSm80 {
  static_assert(arch::Caps<Arch>::kCpAsync &&
                    arch::Caps<Arch>::kBf16TensorCore,
                "serving GEMM needs cp.async and BF16 tensor cores");
  static_assert(solver::ServingBF16ShapeLegal(TileM, TileN, TileK, Stages),
                "serving BF16 tile is outside the legal family");

  using Element = cutlass::bfloat16_t;
  static constexpr int kThreads = solver::kServingBF16Threads;
  static constexpr bool kShapeLegal =
      solver::ServingBF16ShapeLegal(TileM, TileN, TileK, Stages);
  using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>,
                                cute::Int<TileK>>;
  using SmemLayoutAtom = decltype(cute::composition(
      cute::Swizzle<3, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::_64>,
                   cute::Stride<cute::_64, cute::_1>>{}));
  using GmemCopyAtom =
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
                      Element>;
  using GmemTiledCopy = decltype(cute::make_tiled_copy(
      GmemCopyAtom{},
      cute::Layout<cute::Shape<cute::_16, cute::_8>,
                   cute::Stride<cute::_8, cute::_1>>{},
      cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
  using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  // A 32-column tile spread across four N warps gives each warp only two
  // 32-bit B registers. ldmatrix.x4 would request four and fail CuTe's
  // copy-layout divisibility check; x2 is the matching SM80 instruction.
  using SmemCopyAtomB = std::conditional_t<TileM == 16 && TileN == 32,
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, Element>, SmemCopyAtom>;
  using WarpLayout = std::conditional_t<
      TileM == 16,
      cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>>,
      cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM80_16x8x16_F32BF16BF16F32_TN{}, WarpLayout{},
      cute::Tile<cute::Int<(TileM == 16 ? 16 : 32)>,
                 cute::Int<(TileM == 16 ? TileN : 32)>, cute::_16>{}));
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm80CpAsync<Stages>, TileShape,
      Element, cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>,
      Element, cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>,
      TiledMma, GmemTiledCopy, SmemLayoutAtom, SmemCopyAtom,
      cute::identity, GmemTiledCopy, SmemLayoutAtom, SmemCopyAtomB,
      cute::identity>;
  using Epilogue = cutlass::epilogue::collective::DefaultEpilogue<
      Element, cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::EpilogueDefault>;

  // Mainloop and epilogue use the same allocation.  The swizzle's atom is
  // already 16-byte aligned; no additional padding is required for this family.
  static constexpr int kSharedBytes =
      std::max(solver::ServingBF16SmemBytes(TileM, TileN, TileK, Stages),
          ServingEpilogueScratchBytes(TileM,TileN,TILEMEGA_EP_PARALLEL_ARGMAX));
  static_assert(sizeof(typename Mainloop::SharedStorage) <= kSharedBytes,
                "serving shared-memory closed form underestimates CUTLASS");
  union alignas(16) SharedStorage {
    typename Mainloop::SharedStorage mainloop;
    float epilogue[ServingEpilogueScratchBytes(TileM,TileN,TILEMEGA_EP_PARALLEL_ARGMAX)/4];
  };
  static_assert(sizeof(SharedStorage) == kSharedBytes,
                "serving mainloop/epilogue union must match the closed form exactly");
};

#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
// Invocation fields retain the existing collective ABI; execution of the new
// shapes belongs to ServingDmGemm and its explicit four-warp K reduction.
template<class Arch,int M,int N,int K,int S>
struct ServingDmGemmConfig : ServingDmGemm<Arch,M,N,K,S> {
  using Body=ServingDmGemm<Arch,M,N,K,S>;
  using Abi=ServingGemmSm80<Arch,16,64,64,2>;
  static constexpr bool kShapeLegal=solver::DmServingBF16ShapeLegal(M,N,K,S);
  struct Mainloop {
    using TileShape=cute::Shape<cute::Int<M>,cute::Int<N>,cute::Int<K>>;
    using TiledMma=typename Body::TiledMma;
    using ElementA=typename Abi::Mainloop::ElementA;
    using ElementB=typename Abi::Mainloop::ElementB;
    using StrideA=typename Abi::Mainloop::StrideA;
    using StrideB=typename Abi::Mainloop::StrideB;
    using Params=typename Abi::Mainloop::Params;
    using Arguments=typename Abi::Mainloop::Arguments;
    struct alignas(16) SharedStorage {unsigned char bytes[Body::kMainloopBytes];};
    template<class Shape>
    static Params to_underlying_arguments(Shape const& problem,Arguments const& args,void* workspace) {
      return Abi::Mainloop::to_underlying_arguments(problem,args,workspace);
    }
  };
  using Epilogue=typename Abi::Epilogue;
};
template<class Arch,int M,int N,int K,int S>
struct ServingGemmConfig : std::conditional_t<N==16 || K<64,
    ServingDmGemmConfig<Arch,M,N,K,S>,ServingGemmSm80<Arch,M,N,K,S>> {};
#else
template<class Arch,int M,int N,int K,int S>
struct ServingGemmConfig : ServingGemmSm80<Arch,M,N,K,S> {};
// Extension points: future TMA + WGMMA (SM90) and tcgen05 (SM100) task ABIs.
// Until those exist both explicitly inherit the common SM80-class collective.
template<int M,int N,int K,int S>
struct ServingGemmConfig<arch::Sm90,M,N,K,S> : ServingGemmSm80<arch::Sm90,M,N,K,S> {};
template<int M,int N,int K,int S>
struct ServingGemmConfig<arch::Sm100,M,N,K,S> : ServingGemmSm80<arch::Sm100,M,N,K,S> {};

#endif

}  // namespace tilemega::backend
