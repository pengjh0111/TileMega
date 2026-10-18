// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/bfloat16.h>

// Mixed inlined MMA and outlined task calls trigger ptxas C7510. Keep the
// Hopper task dispatch in the same function as its async MMA region.
#if TILEMEGA_ARCH_ID==900
#define TILEMEGA_HOPPER_TASK_INLINE __attribute__((always_inline))
#else
#define TILEMEGA_HOPPER_TASK_INLINE
#endif

namespace tilemega::backend {

namespace gmma = cute::SM90::GMMA;

template<int Rows,int K>
using Sm90KLayout = decltype(cute::tile_to_shape(
    gmma::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
    cute::Shape<cute::Int<Rows>,cute::Int<K>>{}));

template<class Arch>
struct Sm90Gmma {
  using Availability = arch::RequireSm90All<Arch>;
  template<int N,gmma::Major A=gmma::Major::K,gmma::Major B=gmma::Major::K>
  using SS = decltype(cute::SM90::GMMA::ss_op_selector<cutlass::bfloat16_t,
      cutlass::bfloat16_t,float,cute::Shape<cute::_64,cute::Int<N>,cute::_16>,A,B>());
  template<int N,gmma::Major B=gmma::Major::K>
  using RS = decltype(cute::SM90::GMMA::rs_op_selector<cutlass::bfloat16_t,
      cutlass::bfloat16_t,float,cute::Shape<cute::_64,cute::Int<N>,cute::_16>,gmma::Major::K,B>());

  template<gmma::Major Major,class Tensor>
  __device__ static __forceinline__ auto Descriptor(Tensor const& tensor) {
    static_assert(sizeof(Availability)>0);
    return gmma::make_gmma_desc<Major>(tensor);
  }
  template<class Fragment>
  __device__ static __forceinline__ void Fence(Fragment& fragment) {
    static_assert(sizeof(Availability)>0);
    cute::warpgroup_fence_operand(fragment);
  }
  __device__ static __forceinline__ void Arrive() {
    static_assert(sizeof(Availability)>0);
    cute::warpgroup_arrive();
  }
  __device__ static __forceinline__ void Commit() {
    static_assert(sizeof(Availability)>0);
    cute::warpgroup_commit_batch();
  }
  template<int Pending>
  __device__ static __forceinline__ void Wait() {
    static_assert(sizeof(Availability)>0);
    cute::warpgroup_wait<Pending>();
  }
};

template<class Arch,int M,int N,int K>
struct Sm90DecodeGmma {
  static_assert((M==8 || M==16) && N>=64 && N%64==0 && (K==64 || K==128),
                "decode WGMMA requires M8/M16, N multiple of 64 and K64/128");
  using Gmma=Sm90Gmma<Arch>;
  using Mma=decltype(cute::make_tiled_mma(typename Gmma::template SS<M>{}));
  using Element=cutlass::bfloat16_t;

  template<class Accumulator>
  __device__ static __forceinline__ void Accumulate(
      Element const* weights,Element const* activation,Accumulator& accum) {
    using namespace cute;
    static_assert(sizeof(typename Gmma::Availability)>0);
    Mma mma;
    auto thread=mma.get_slice(int(threadIdx.x));
    auto w=make_tensor(make_smem_ptr(weights),Sm90KLayout<N,K>{});
    auto x=make_tensor(make_smem_ptr(activation),Sm90KLayout<M,K>{});
    auto rw=thread.partition_fragment_A(w);
    auto rx=thread.partition_fragment_B(x);
    Gmma::Arrive();
    CUTE_UNROLL
    for(int k=0;k<size<2>(rw);++k)gemm(mma,rw(_,_,k),rx(_,_,k),accum);
    Gmma::Commit();
  }
  template<class Accumulator>
  __device__ static __forceinline__ float* Materialize(Accumulator& accum,char* workspace,
      std::uint64_t* barriers=nullptr,int barrier_count=0) {
    using namespace cute;
    Gmma::template Wait<0>();Gmma::Fence(accum);
    codegen::executor::ComputeSync();
    // The FP32 tile aliases the pipeline header, including its live barriers.
    if(barrier_count) {
      if(codegen::executor::ComputeThread()==0)
        for(int i=0;i<barrier_count;++i)codegen::executor::Async<Arch>::Invalidate(barriers+i);
      codegen::executor::ComputeSync();
    }
    Mma mma;
    auto owned=mma.get_slice(int(threadIdx.x)).partition_C(
        make_identity_tensor(Shape<Int<N>,Int<M>>{}));
    float* tile=reinterpret_cast<float*>(workspace);
    CUTE_UNROLL
    for(int i=0;i<size(accum);++i)
      tile[get<1>(owned(i))*N+get<0>(owned(i))]=accum(i);
    codegen::executor::ComputeSync();
    return tile;
  }
};

template<class Arch,int M,int N,int K>
struct Sm90ForwardGmma {
  static_assert((M==64 || M==128) && (N==32 || N==64 || N==128 || N==256) &&
                K==64 && M*N<=16384,
                "forward WGMMA requires M64/128, N32/64/128/256, K64 and bounded accumulators");
  using Gmma=Sm90Gmma<Arch>;
  using Mma=decltype(cute::make_tiled_mma(typename Gmma::template SS<N>{}));
  using Element=cutlass::bfloat16_t;
  template<class Accumulator>
  __device__ static __forceinline__ void Accumulate(
      Element const* weights,Element const* activation,Accumulator& accum) {
    using namespace cute;
    static_assert(sizeof(typename Gmma::Availability)>0);
    auto thread=Mma{}.get_slice(int(threadIdx.x));
    auto a=make_tensor(make_smem_ptr(activation),Sm90KLayout<M,K>{});
    auto b=make_tensor(make_smem_ptr(weights),Sm90KLayout<N,K>{});
    auto ra=thread.partition_fragment_A(a);
    auto rb=thread.partition_fragment_B(b);
    Gmma::Arrive();
    CUTE_UNROLL
    for(int k=0;k<size<2>(ra);++k)gemm(Mma{},ra(_,_,k),rb(_,_,k),accum);
    Gmma::Commit();
  }
  template<class Accumulator>
  __device__ static __forceinline__ float* Materialize(Accumulator& accum,char* workspace,
      std::uint64_t* barriers=nullptr,int barrier_count=0) {
    using namespace cute;
    Gmma::template Wait<0>();Gmma::Fence(accum);
    codegen::executor::ComputeSync();
    if(barrier_count) {
      if(codegen::executor::ComputeThread()==0)
        for(int i=0;i<barrier_count;++i)codegen::executor::Async<Arch>::Invalidate(barriers+i);
      codegen::executor::ComputeSync();
    }
    auto owned=Mma{}.get_slice(int(threadIdx.x)).partition_C(
        make_identity_tensor(Shape<Int<M>,Int<N>>{}));
    float* tile=reinterpret_cast<float*>(workspace);
    CUTE_UNROLL
    for(int i=0;i<size(accum);++i)
      tile[get<0>(owned(i))*N+get<1>(owned(i))]=accum(i);
    codegen::executor::ComputeSync();
    return tile;
  }
};

} // namespace tilemega::backend
