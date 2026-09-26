// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/ArchDispatch.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {
// Each warp owns a disjoint 16-key slice. QK's accumulator coordinates equal
// PV's A coordinates within that warp, so probabilities never leave registers.
template<class Arch, int N, int K, bool TransposeB = false>
struct ServingAttentionWarp {
  static_assert(arch::Caps<Arch>::kCpAsync && arch::Caps<Arch>::kBf16TensorCore);
  using Element = cutlass::bfloat16_t;
  using Mma = decltype(cute::make_tiled_mma(
      cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>{},
      cute::Tile<cute::_16,cute::Int<N>,cute::_16>{}));
  using LayoutA = decltype(cute::composition(cute::Swizzle<3,3,3>{},
      cute::Layout<cute::Shape<cute::_16,cute::Int<K>>,
                   cute::Stride<cute::Int<K>,cute::_1>>{}));
  using LayoutB = std::conditional_t<TransposeB,
      decltype(cute::composition(cute::Swizzle<3,3,3>{},
          cute::Layout<cute::Shape<cute::Int<N>,cute::Int<K>>,
                       cute::Stride<cute::_1,cute::Int<N>>>{})),
      decltype(cute::composition(cute::Swizzle<3,3,3>{},
          cute::Layout<cute::Shape<cute::Int<N>,cute::Int<K>>,
                       cute::Stride<cute::Int<K>,cute::_1>>{}))>;
  using LoadA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using LoadB = std::conditional_t<TransposeB,
      cute::Copy_Atom<cute::SM75_U16x8_LDSM_T,Element>,
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>>;
  __device__ static auto Accumulator() {
    auto result=cute::partition_fragment_C(Mma{},cute::Shape<cute::_16,cute::Int<N>>{});
    cute::clear(result);return result;
  }
  template<class Acc>
  __device__ static void QK(Element* a,Element* b,Acc& out) {
    using namespace cute;
    Mma mma;auto thr=mma.get_slice(int(threadIdx.x)&31);
    auto sA=make_tensor(make_smem_ptr(a),LayoutA{});
    auto sB=make_tensor(make_smem_ptr(b),LayoutB{});
    auto rA=thr.partition_fragment_A(sA);auto rB=thr.partition_fragment_B(sB);
    auto ca=make_tiled_copy_A(LoadA{},mma);auto cb=make_tiled_copy_B(LoadB{},mma);
    auto as=ca.get_slice(int(threadIdx.x)&31).partition_S(sA);
    auto bs=cb.get_slice(int(threadIdx.x)&31).partition_S(sB);
    auto ad=ca.get_slice(int(threadIdx.x)&31).retile_D(rA);
    auto bd=cb.get_slice(int(threadIdx.x)&31).retile_D(rB);
    #pragma unroll
    for(int k=0;k<size<2>(rA);++k) {
      copy(LoadA{},as(_,_,k),ad(_,_,k));
      copy(LoadB{},bs(_,_,k),bd(_,_,k));
      gemm(mma,rA(_,_,k),rB(_,_,k),out);
    }
  }
  template<class Prob,class Coords,class Acc>
  __device__ static void PV(Prob const& probability,Coords const& source_coords,
                            Element* b,Acc& out) {
    using namespace cute;
    Mma mma;auto thr=mma.get_slice(int(threadIdx.x)&31);
    auto id=make_identity_tensor(Shape<_16,Int<K>>{});
    auto coords=thr.partition_A(id);
    auto rA=make_fragment_like<Element>(coords);
    #pragma unroll
    for(int i=0;i<size(rA);++i) {
      #pragma unroll
      for(int j=0;j<size(probability);++j)
        if(get<0>(coords(i))==get<0>(source_coords(j)) &&
           get<1>(coords(i))==get<1>(source_coords(j)))
          rA(i)=Element(probability(j));
    }
    auto sB=make_tensor(make_smem_ptr(b),LayoutB{});
    auto rB=thr.partition_fragment_B(sB);
    auto cb=make_tiled_copy_B(LoadB{},mma);
    auto bs=cb.get_slice(int(threadIdx.x)&31).partition_S(sB);
    auto bd=cb.get_slice(int(threadIdx.x)&31).retile_D(rB);
    copy(LoadB{},bs(_,_,0),bd(_,_,0));
    gemm(mma,rA(_,_,0),rB(_,_,0),out);
  }
};
}
