// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingAttentionWarp.h>

namespace tilemega::backend {
// O^T[D,8] = V^T[D,16] * P^T[16,8]. QK keeps its original fragment layout.
template<class Arch,int D,class ValueLayout>
struct ServingAttentionPVSwap {
  static_assert(arch::Caps<Arch>::kBf16TensorCore && arch::Caps<Arch>::kCpAsync);
  using Element=cutlass::bfloat16_t;
  using LayoutB=ValueLayout; // The transport still stores V in the same page image.
  using Mma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>{},
      cute::Tile<cute::Int<D>,cute::_8,cute::_16>{}));
  using LoadA=cute::Copy_Atom<cute::SM75_U16x8_LDSM_T,Element>;
  __device__ static auto Accumulator() {
    auto result=cute::partition_fragment_C(Mma{},cute::Shape<cute::Int<D>,cute::_8>{});
    cute::clear(result);return result;
  }
  template<class Prob,class Coords,class Acc>
  __device__ static void PV(Prob const& probability,Coords const& source_coords,
                            Element* values,Acc& out) {
    using namespace cute;
    Mma mma;auto thr=mma.get_slice(codegen::executor::ComputeThread()&31);
    auto coords=thr.partition_B(make_identity_tensor(Shape<_8,_16>{}));
    auto rB=make_fragment_like<Element>(coords);
    // The first eight QK rows map directly to the MMA B fragment. Both
    // loops unroll to register moves/conversions; no warp exchange is needed.
    #pragma unroll
    for(int i=0;i<size(rB);++i) {
      #pragma unroll
      for(int j=0;j<size(probability);++j)
        if(get<0>(coords(i))==get<0>(source_coords(j)) &&
           get<1>(coords(i))==get<1>(source_coords(j)))rB(i)=Element(probability(j));
    }
    auto sA=make_tensor(make_smem_ptr(values),ValueLayout{});
    auto rA=thr.partition_fragment_A(sA);
    auto copy_a=make_tiled_copy_A(LoadA{},mma);
    auto src=copy_a.get_slice(codegen::executor::ComputeThread()&31).partition_S(sA);
    auto dst=copy_a.get_slice(codegen::executor::ComputeThread()&31).retile_D(rA);
    copy(LoadA{},src(_,_,0),dst(_,_,0));
    gemm(mma,rA(_,_,0),rB(_,_,0),out);
  }
};
}
