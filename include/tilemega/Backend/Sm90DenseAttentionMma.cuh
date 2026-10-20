// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {

// Contiguous historical bulk copies feed these layouts without repacking.
// Each warp contracts a separate 16-key interval; inter-warp normalization is
// deferred until the end of the chunk rather than repeated for every block.
template<int N,int K,bool Transpose=false>
struct Sm90DenseAttentionMma {
  using Element=cutlass::bfloat16_t;
  using Mma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>{},
      cute::Tile<cute::_16,cute::Int<N>,cute::_16>{}));
  using LayoutA=cute::Layout<cute::Shape<cute::_16,cute::Int<K>>,
      cute::Stride<cute::Int<K>,cute::_1>>;
  using LayoutB=std::conditional_t<Transpose,
      cute::Layout<cute::Shape<cute::Int<N>,cute::Int<K>>,cute::Stride<cute::_1,cute::Int<N>>>,
      cute::Layout<cute::Shape<cute::Int<N>,cute::Int<K>>,cute::Stride<cute::Int<K>,cute::_1>>>;
  using LoadA=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using LoadB=std::conditional_t<Transpose,
      cute::Copy_Atom<cute::SM75_U16x8_LDSM_T,Element>,LoadA>;
  __device__ static auto Accumulator() {
    auto result=cute::partition_fragment_C(Mma{},cute::Shape<cute::_16,cute::Int<N>>{});
    cute::clear(result);return result;
  }
  __device__ static auto Coordinates() {
    return Mma{}.get_slice(threadIdx.x&31).partition_C(
        cute::make_identity_tensor(cute::Shape<cute::_16,cute::Int<N>>{}));
  }
  template<int I>
  __host__ __device__ static constexpr int QueryGroup() {
    auto coordinates=Mma{}.get_slice(cute::Int<0>{}).partition_C(
        cute::make_identity_tensor(cute::Shape<cute::_16,cute::Int<N>>{}));
    return int(cute::get<0>(coordinates(cute::Int<I>{})))/8;
  }
  template<class Acc>
  __device__ static void QK(Element* a,Element* b,Acc& result) {
    using namespace cute;
    Mma mma;auto thread=mma.get_slice(threadIdx.x&31);
    auto sa=make_tensor(make_smem_ptr(a),LayoutA{});
    auto sb=make_tensor(make_smem_ptr(b),LayoutB{});
    auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
    auto ca=make_tiled_copy_A(LoadA{},mma);auto cb=make_tiled_copy_B(LoadB{},mma);
    auto as=ca.get_slice(threadIdx.x&31).partition_S(sa);auto bs=cb.get_slice(threadIdx.x&31).partition_S(sb);
    auto ad=ca.get_slice(threadIdx.x&31).retile_D(ra);auto bd=cb.get_slice(threadIdx.x&31).retile_D(rb);
    for_each(make_int_sequence<size<2>(ra)>{},[&](auto k) {
      copy(LoadA{},as(_,_,k),ad(_,_,k));copy(LoadB{},bs(_,_,k),bd(_,_,k));
      gemm(mma,ra(_,_,k),rb(_,_,k),result);
    });
  }
  template<class Probability,class Coords,class Acc>
  __device__ static void PV(Probability const& probability,Coords const& coordinates,
                            Element* b,Acc& result) {
    using namespace cute;
    Mma mma;auto thread=mma.get_slice(threadIdx.x&31);
    auto owned=thread.partition_A(make_identity_tensor(Shape<_16,Int<K>>{}));
    auto ra=make_fragment_like<Element>(owned);
    for_each(make_int_sequence<size(ra)>{},[&](auto i) {
      for_each(make_int_sequence<size(probability)>{},[&](auto j) {
        if(get<0>(owned(i))==get<0>(coordinates(j)) && get<1>(owned(i))==get<1>(coordinates(j)))
          ra(i)=Element(probability(j));
      });
    });
    auto sb=make_tensor(make_smem_ptr(b),LayoutB{});
    auto rb=thread.partition_fragment_B(sb);auto cb=make_tiled_copy_B(LoadB{},mma);
    auto bs=cb.get_slice(threadIdx.x&31).partition_S(sb);auto bd=cb.get_slice(threadIdx.x&31).retile_D(rb);
    copy(LoadB{},bs(_,_,0),bd(_,_,0));gemm(mma,ra(_,_,0),rb(_,_,0),result);
  }
};

}  // namespace tilemega::backend
