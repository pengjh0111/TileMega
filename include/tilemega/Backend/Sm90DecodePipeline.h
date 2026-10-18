// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/executor/Async.cuh>
#ifndef TILEMEGA_NONPAGED_TILED
#define TILEMEGA_NONPAGED_TILED 0
#endif
namespace tilemega::backend {
// Two operand slots are shared by the TMA/bulk producer and WGMMA consumer.
// A slot is reissued only after wait_group<1> has retired its previous reader.
template<class Arch,int M,int N,int K>
struct Sm90DecodePipeline {
  using Body=Sm90DecodeGmma<Arch,M,N,K>;
  using Gmma=typename Body::Gmma;
  using Async=codegen::executor::Async<Arch>;
  using Element=cutlass::bfloat16_t;
  static constexpr int kSharedBytes=1024+4*K*(M+N);
  template<class Operands>
  __device__ static __forceinline__ float* Run(Operands const& p,int tile_m,int tile_n,char* shared) {
    using namespace cute;
    using codegen::executor::ComputeThread;
    using codegen::executor::ComputeSync;
    using codegen::executor::kComputeThreads;
    auto layout_a=as_position_independent_swizzle_layout(Sm90KLayout<M,K>{});
    auto layout_b=as_position_independent_swizzle_layout(Sm90KLayout<N,K>{});
    if(!p.activation_tensor_map || !p.a || !p.b || p.k_count<=0 ||
        p.tensor_k_begin<0 || p.tensor_k_begin%64 || (Async::Shared(shared)&1023)) {
      asm volatile("trap;");return nullptr;
    }
#if TILEMEGA_NONPAGED_TILED
    if(!p.weight_base || p.k_begin%K || p.k_total_full<=0) {
      asm volatile("trap;");return nullptr;
    }
#else
    if(!p.tensor_map) {asm volatile("trap;");return nullptr;}
#endif
    int a_pitch=p.a_row_stride?p.a_row_stride:p.k_total;
    if(a_pitch%8 || p.k_begin<0 || p.k_begin+p.k_count>a_pitch) {
      asm volatile("trap;");return nullptr;
    }
    ComputeSync();
    auto* barriers=reinterpret_cast<std::uint64_t*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+2*M*K;
    if(ComputeThread()==0) {Async::Init(barriers,1);Async::Init(barriers+1,1);}
    Async::InitFence();ComputeSync();Async::ProxyAsyncGlobalFence();
    int iterations=(p.k_count+K-1)/K;
    auto issue=[&](int it) {
      auto* as=a+(it%2)*M*K;auto* bs=b+(it%2)*N*K;
      bool tma_a=it*K+K<=p.k_count;
      if(ComputeThread()==0) {
        auto* barrier=barriers+it%2;
        Async::ExpectTx(barrier,2*N*K+(tma_a?2*M*K:0));
        if(tma_a)for(int col=0;col<K;col+=64)
          Async::Tensor2D(as+layout_a(0,col),p.activation_tensor_map,
              p.tensor_k_begin+it*K+col,tile_m*M,barrier);
#if TILEMEGA_NONPAGED_TILED
        int kt=(p.k_total_full+K-1)/K;
        auto* src=p.weight_base+(std::size_t(tile_n)*kt+p.k_begin/K+it)*N*K;
        Async::Bulk(bs,src,2*N*K,barrier);
#else
        for(int col=0;col<K;col+=64)
          Async::Tensor2D(bs+layout_b(0,col),p.tensor_map,
              p.tensor_k_begin+it*K+col,tile_n*N,barrier);
#endif
      }
      if(!tma_a) {
        // A split ending inside a K tile must exclude the next split's values.
        // The BF16 epilogue therefore sees exactly the old zero-filled tail.
        for(int v=ComputeThread()*8;v<M*K;v+=kComputeThreads*8) {
          int row=tile_m*M+v/K,k=it*K+v%K;
          bool valid=row<p.m && k<p.k_count;
          auto* src=valid?p.a+std::int64_t(row)*a_pitch+p.k_begin+k:p.a;
          Async::Copy16Bytes(as+layout_a(v/K,v%K),src,valid?min(8,p.k_count-k)*2:0);
        }
        cp_async_fence();cp_async_wait<0>();ComputeSync();
        Async::FenceProxyAsyncShared();ComputeSync();
      }
    };
    issue(0);if(iterations>1)issue(1);
    typename Body::Mma mma;
    auto accum=partition_fragment_C(mma,Shape<Int<N>,Int<M>>{});
    clear(accum);Gmma::Fence(accum);
    for(int it=0;it<iterations;++it) {
      Async::Wait(barriers+it%2,(it/2)&1);ComputeSync();
      Body::Accumulate(b+(it%2)*N*K,a+(it%2)*M*K,accum);
      Gmma::template Wait<1>();ComputeSync();
      if(it>0 && it+1<iterations)issue(it+1);
    }
    return Body::Materialize(accum,shared,barriers,2);
  }
};
} // namespace tilemega::backend
