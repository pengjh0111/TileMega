// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Target/ArchDispatch.h>

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
struct ServingGemmConfig {
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
      cute::identity, GmemTiledCopy, SmemLayoutAtom, SmemCopyAtom,
      cute::identity>;
  using Epilogue = cutlass::epilogue::collective::DefaultEpilogue<
      Element, cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::EpilogueDefault>;

  // Mainloop and epilogue use the same allocation.  The swizzle's atom is
  // already 16-byte aligned; no additional padding is required for this family.
  static constexpr int kSharedBytes =
      solver::ServingBF16SmemBytes(TileM, TileN, TileK, Stages);
  static_assert(sizeof(typename Mainloop::SharedStorage) <= kSharedBytes,
                "serving shared-memory closed form underestimates CUTLASS");
};

}  // namespace tilemega::backend
