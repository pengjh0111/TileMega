// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Target/ArchDispatch.h>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {

// A single swizzled BF16 MMA tile used by QK and PV. The caller owns the
// shared-memory lifetime and may refill the same tile between contractions.
template <class Arch, int M, int N, int K, bool TransposeB = false>
struct ServingAttentionMma {
  static_assert(arch::Caps<Arch>::kBf16TensorCore &&
                arch::Caps<Arch>::kCpAsync);
  static_assert(M == 16 && N % 32 == 0 && K % 16 == 0);
  using Element = cutlass::bfloat16_t;
  using AtomLayout = decltype(cute::composition(
      cute::Swizzle<3, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::_64>,
                   cute::Stride<cute::_64, cute::_1>>{}));
  using LayoutA = decltype(cute::tile_to_shape(
      AtomLayout{}, cute::Shape<cute::Int<M>, cute::Int<K>>{}));
  using TransposeAtomLayout = decltype(cute::composition(
      cute::Swizzle<3, 3, 3>{},
      cute::Layout<cute::Shape<cute::_64, cute::_8>,
                   cute::Stride<cute::_1, cute::_64>>{}));
  using LayoutB = decltype(cute::tile_to_shape(
      std::conditional_t<TransposeB, TransposeAtomLayout, AtomLayout>{},
      cute::Shape<cute::Int<N>, cute::Int<K>>{}));
  using Mma = decltype(cute::make_tiled_mma(
      cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>>{},
      cute::Tile<cute::_16, cute::Int<N>, cute::_16>{}));
  using LoadA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  using LoadB = std::conditional_t<TransposeB,
      cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>,
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>>;

  static constexpr int kThreads = 128;
  static constexpr int kAElements = cute::cosize(LayoutA{});
  static constexpr int kBElements = cute::cosize(LayoutB{});

  __device__ static void Run(Element* storage_a, Element* storage_b,
                             float* output, int output_pitch) {
    using namespace cute;
    auto sA = make_tensor(make_smem_ptr(storage_a), LayoutA{});
    auto sB = make_tensor(make_smem_ptr(storage_b), LayoutB{});
    Mma mma;
    auto thr = mma.get_slice(int(threadIdx.x));
    auto rA = thr.partition_fragment_A(sA);
    auto rB = thr.partition_fragment_B(sB);
    auto rC = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
    clear(rC);
    auto copy_a = make_tiled_copy_A(LoadA{}, mma);
    auto copy_b = make_tiled_copy_B(LoadB{}, mma);
    auto src_a = copy_a.get_slice(int(threadIdx.x)).partition_S(sA);
    auto src_b = copy_b.get_slice(int(threadIdx.x)).partition_S(sB);
    auto dst_a = copy_a.get_slice(int(threadIdx.x)).retile_D(rA);
    auto dst_b = copy_b.get_slice(int(threadIdx.x)).retile_D(rB);
    CUTE_STATIC_ASSERT_V(size<2>(rA) == size<2>(rB));
    for (int block = 0; block < size<2>(rA); ++block) {
      copy(LoadA{}, src_a(_, _, block), dst_a(_, _, block));
      copy(LoadB{}, src_b(_, _, block), dst_b(_, _, block));
      gemm(mma, rA(_, _, block), rB(_, _, block), rC);
    }
    auto coords = make_identity_tensor(Shape<Int<M>, Int<N>>{});
    auto owned = thr.partition_C(coords);
    for (int i = 0; i < size(rC); ++i) {
      auto m = get<0>(owned(i));
      auto n = get<1>(owned(i));
      output[int(m) * output_pitch + int(n)] = rC(i);
    }
  }
};

}  // namespace tilemega::backend
