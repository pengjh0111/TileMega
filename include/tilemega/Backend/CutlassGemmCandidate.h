// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.2 candidate legality pruning, §5.3 implementation contract.
//
// The compile-time half of the backend query.  One `GemmCandidate<...>` is one
// point of the implementation search space; everything the middle pruning tier
// is allowed to read is a `constexpr` member, so a whole candidate family can
// be evaluated in a translation unit that emits no kernel (V-D / F-7).
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/dispatch_policy.hpp>

namespace tilemega::backend {

/// The trait record is `solver::BackendTraits`: the query interface and the
/// enumerator must read the same fields, and `kEstimatedRegisters` is absent
/// from both because only ptxas knows it (§4.3).
using solver::BackendTraits;

/// The SIMT f32 TN family the megakernel actually dispatches to, reparameterized
/// over the tile shape and pipeline depth.  Mirrors CUTLASS's
/// `DefaultGemmConfigurationToCutlass3Types<OpClassSimt, Sm80, float,
/// RowMajor, float, ColumnMajor, float, RowMajor, float>` with the tile shape
/// lifted out of the specialization.
template <int TileM, int TileN, int TileK, int Stages>
struct SimtGemmF32TN {
  using Element = float;

  // 16x16 threads of UniversalFMA: the MMA covers 16 rows and 16 columns, so
  // both output tile extents must be multiples of 16.
  static constexpr int kThreadsM = 16;
  static constexpr int kThreadsN = 16;
  static constexpr int kThreads = kThreadsM * kThreadsN;

  // The cp.async thread layout is K-major: `kCopyK` threads walk the K axis
  // and the rest walk M (resp. N).  K narrower than 16 needs a shorter row.
  static constexpr int kCopyK = TileK < 16 ? TileK : 16;
  static constexpr int kCopyM = kCopyK > 0 ? kThreads / kCopyK : 0;

  static constexpr bool kShapeLegal =
      TileM > 0 && TileN > 0 && TileK > 0 && Stages > 1 &&
      TileM % kThreadsM == 0 && TileN % kThreadsN == 0 &&
      kCopyK > 0 && kThreads % kCopyK == 0 && TileK % kCopyK == 0 &&
      kCopyM > 0 && TileM % kCopyM == 0 && TileN % kCopyM == 0;

  // Smem is the padded (M+1, K) / (N+1, K) A and B tiles, `Stages` deep.  The
  // formula is only used for the guard below; the authoritative number is
  // `sizeof(Mainloop::SharedStorage)`, which V-D measured to match ptxas to
  // the byte.
  static constexpr int kEstimatedSmemBytes =
      static_cast<int>(sizeof(Element)) * Stages * TileK *
      ((TileM + 1) + (TileN + 1));
};

namespace detail {

using cute::Int;
using cute::Layout;
using cute::Shape;
using cute::Stride;
using cute::_1;

template <class Info, bool Legal = Info::kShapeLegal>
struct SimtCollective;

template <class Info>
struct SimtCollective<Info, false> {
  // An illegal shape must not instantiate a CUTLASS type: its static_asserts
  // are hard errors, not a queryable `false`.  The pruning tier reads
  // `kShapeLegal` and never touches these aliases.
  using Mainloop = void;
  using Epilogue = void;
  static constexpr int kSmemBytes = 0;
};

template <class Info>
struct SimtCollective<Info, true> {
  using Element = typename Info::Element;
  using TileShape =
      Shape<Int<Info::kTileM>, Int<Info::kTileN>, Int<Info::kTileK>>;
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::UniversalFMA<Element, Element, Element, Element>>,
      Layout<Shape<Int<Info::kThreadsM>, Int<Info::kThreadsN>, _1>>>;
  using SmemLayoutAtomA =
      Layout<Shape<Int<Info::kTileM>, Int<Info::kTileK>>,
             Stride<_1, Int<Info::kTileM + 1>>>;
  using SmemLayoutAtomB =
      Layout<Shape<Int<Info::kTileN>, Int<Info::kTileK>>,
             Stride<_1, Int<Info::kTileN + 1>>>;
  using CopyThreadLayout =
      Layout<Shape<Int<Info::kCopyM>, Int<Info::kCopyK>>,
             Stride<Int<Info::kCopyK>, _1>>;
  using GmemTiledCopy = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>{},
      CopyThreadLayout{}));

  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm80CpAsync<Info::kStages>, TileShape,
      Element, cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>,
      Element, cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>,
      TiledMma,
      GmemTiledCopy, SmemLayoutAtomA,
      cute::Copy_Atom<cute::DefaultCopy, Element>, cute::identity,
      GmemTiledCopy, SmemLayoutAtomB,
      cute::Copy_Atom<cute::DefaultCopy, Element>, cute::identity>;

  using Epilogue = cutlass::epilogue::collective::DefaultEpilogue<
      Element, cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
      cutlass::epilogue::thread::LinearCombination<Element, 1, Element, Element>,
      cutlass::gemm::EpilogueDefault>;

  static constexpr int kSmemBytes =
      static_cast<int>(sizeof(typename Mainloop::SharedStorage));
};

}  // namespace detail

/// One point of the implementation search space (§5.1).  `Legal()` is the
/// cheap tier-1 predicate; `Info()` is what the tier-2 ranker reads.
template <int TileM, int TileN, int TileK, int Stages>
struct GemmCandidate {
  struct Shape_ : SimtGemmF32TN<TileM, TileN, TileK, Stages> {
    static constexpr int kTileM = TileM;
    static constexpr int kTileN = TileN;
    static constexpr int kTileK = TileK;
    static constexpr int kStages = Stages;
  };
  using Collective = detail::SimtCollective<Shape_>;
  using Mainloop = typename Collective::Mainloop;
  using Epilogue = typename Collective::Epilogue;

  static constexpr bool kShapeLegal = Shape_::kShapeLegal;
  static constexpr int kThreads = Shape_::kThreads;
  static constexpr int kSmemBytes = Collective::kSmemBytes;

  /// cp.async on SM80 moves one f32 at a time here, so both operands need
  /// only element alignment; the field exists because §4.2 makes it part of
  /// the query, not because this family constrains it.
  static constexpr int kAlignmentA = 1;
  static constexpr int kAlignmentB = 1;
  static constexpr int kArchSm = 80;
  /// This family is not cluster-launched; §4.5's cluster candidates come from
  /// the SM90 collective, not from here.
  static constexpr int kClusterM = 1, kClusterN = 1, kClusterK = 1;

  static constexpr bool Legal(int smem_budget_bytes) {
    return kShapeLegal && kSmemBytes > 0 && kSmemBytes <= smem_budget_bytes;
  }

  static constexpr BackendTraits Traits() {
    BackendTraits traits{};
    traits.tile_m = TileM;
    traits.tile_n = TileN;
    traits.tile_k = TileK;
    traits.stages = Stages;
    traits.threads = kThreads;
    traits.smem_bytes = kSmemBytes;
    traits.cluster = {kClusterM, kClusterN, kClusterK};
    traits.alignment = {kAlignmentA, kAlignmentB};
    traits.arch_sm = kArchSm;
    traits.shape_legal = kShapeLegal;
    return traits;
  }

  // The host-side enumerator prunes on `solver::SimtF32*` without compiling
  // anything; these are the assertions that make that legitimate.  A CUTLASS
  // upgrade that changes the collective's shared storage breaks the build here
  // rather than silently invalidating the candidate set.
  static_assert(kShapeLegal == solver::SimtF32ShapeLegal(TileM, TileN, TileK, Stages));
  static_assert(!kShapeLegal || kThreads == solver::kSimtF32Threads);
  static_assert(!kShapeLegal ||
                kSmemBytes == solver::SimtF32SmemBytes(TileM, TileN, TileK, Stages));
};

}  // namespace tilemega::backend
