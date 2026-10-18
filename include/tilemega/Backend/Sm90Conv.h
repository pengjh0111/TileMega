// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/DmDescriptors.h>

namespace tilemega::backend {
template<class Arch,int M,int N,int K,int Stages=2>
struct Sm90Conv {
  static_assert(M==64 && K==64,"TMA im2col convolution requires M64/K64");
  using Body=Sm90ForwardGmma<Arch,M,N,K>;
  using Gmma=typename Body::Gmma;
  using Async=codegen::executor::Async<Arch>;
  using Element=cutlass::bfloat16_t;
  static_assert(Stages>=2 && Stages<=4);
  static constexpr int kSharedBytes=1024+2*Stages*K*(M+N)>4*M*N?1024+2*Stages*K*(M+N):4*M*N;
  template<bool TiledB,class Operands>
#if TILEMEGA_FORWARD_WGMMA_OUTLINE > 1
  __device__ static __noinline__
#else
  __device__ static __forceinline__
#endif
  float* Run(Operands const& p,codegen::ConvDesc const& conv,
      codegen::DmBufferLayout const& layout,int tile_m,int tile_n,char* shared) {
    using namespace cute;
    using codegen::executor::ComputeThread;
    using codegen::executor::ComputeSync;
    if(!p.activation_tensor_map || (!TiledB && !p.tensor_map) ||
       (TiledB && !p.weight_base) || conv.c<64 || conv.c%64 ||
       p.k_begin%K || p.k_count<=0 || p.k_count%K || (Async::Shared(shared)&1023)) {
      asm volatile("trap;");return nullptr;
    }
    ComputeSync();
    auto* barriers=reinterpret_cast<std::uint64_t*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Stages*M*K;
    if(ComputeThread()==0)for(int slot=0;slot<Stages;++slot)Async::Init(barriers+slot,1);
    Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
    auto issue=[&](int it) {
      if(ComputeThread()!=0)return;
      auto* barrier=barriers+it%Stages;
      Async::ExpectTx(barrier,2*K*(M+N));
      int iteration=p.k_begin/K+it,blocks=conv.c/K;
      int tap=iteration/blocks,c=(iteration%blocks)*K;
      int row=tile_m*M,image=row/(conv.p*conv.q),pixel=row%(conv.p*conv.q);
      int w=(pixel%conv.q)*conv.stride_w+int(layout.halo_left)-int(conv.pad_w);
      int h=(pixel/conv.q)*conv.stride_h+int(layout.halo_top)-int(conv.pad_h);
      Async::TensorIm2col4D(a+(it%Stages)*M*K,p.activation_tensor_map,c,w,h,image,
          std::uint16_t((tap%conv.s)*conv.dilation_w),std::uint16_t((tap/conv.s)*conv.dilation_h),barrier);
      if constexpr(TiledB) {
        int kt=p.k_total_full/K;
        Async::Bulk(b+(it%Stages)*N*K,p.weight_base+(std::size_t(tile_n)*kt+iteration)*N*K,2*N*K,barrier);
      }else Async::Tensor2D(b+(it%Stages)*N*K,p.tensor_map,iteration*K,tile_n*N,barrier);
    };
    int iterations=p.k_count/K;
    for(int slot=0;slot<Stages && slot<iterations;++slot)issue(slot);
    auto accum=partition_fragment_C(typename Body::Mma{},Shape<Int<M>,Int<N>>{});
    clear(accum);Gmma::Fence(accum);
    for(int it=0;it<iterations;++it) {
      Async::Wait(barriers+it%Stages,(it/Stages)&1);ComputeSync();
      Body::Accumulate(b+(it%Stages)*N*K,a+(it%Stages)*M*K,accum);
      if constexpr(Stages>2) {
        Gmma::template Wait<1>();ComputeSync();
        if(it>0 && it-1+Stages<iterations)issue(it-1+Stages);
      }else {
        Gmma::template Wait<0>();ComputeSync();
        if(it+2<iterations)issue(it+2);
      }
    }
    return Body::Materialize(accum,shared,barriers,Stages);
  }
};
} // namespace tilemega::backend
