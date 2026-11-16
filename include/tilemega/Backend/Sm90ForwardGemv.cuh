// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90ForwardSmall.cuh>
#include <tilemega/Backend/Sm90ForwardResources.h>

namespace tilemega::backend {

// The original GEMV declaration is removed from Small.cuh when this body is
// integrated. K is static so codegen and resource filtering see the same bytes.
template<int Block,int K>
struct Sm90ForwardGemv {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Block==64 || Block==128 || Block==256);
  static_assert(K>0 && K%64==0);
  static constexpr int kThreads=160,kTileK=64,kOutputs=(Block+127)/128;
  static constexpr auto kLayout=Sm90ForwardGemvLayout(K,Block);
  static constexpr int kAOffset=kLayout.key,kABytes=2*K;
  static constexpr int kPageOffset=kLayout.value,kPageBytes=2*Block*kTileK;
  static constexpr int kSharedBytes=kLayout.end;

  __device__ static __forceinline__ int Swizzled(int row,int col) {
    return 512*(row/8+(Block/8)*(col/64))+64*(row%8)+
        8*(((col%64)/8)^(row%8))+col%8;
  }
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    if(threadIdx.x>=128)return;
    auto* activation=reinterpret_cast<Element*>(shared+kAOffset);
    auto* page=reinterpret_cast<Element*>(shared+kPageOffset);
    int blocks=(p.out_channels+Block-1)/Block,rows=p.width,panels=K/kTileK;
    int total=blocks*rows;
    unsigned ap=state.producer_generations[0],ac=state.consumer_generations[0];
    unsigned wp=state.producer_generations[1],wc=state.consumer_generations[1];
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);TM_FWD_WINDOW_ACC(store_trace);
    Async::FenceProxyAsyncShared();
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)Async::ProxyAsyncGlobalFence();
    for(int task=int(blockIdx.x);task<total;task+=int(gridDim.x)) {
      int row=task/blocks,block=task%blocks;
      float sums[kOutputs];
      #pragma unroll
      for(int i=0;i<kOutputs;++i) {
        int channel=block*Block+int(threadIdx.x)+i*128;
        sums[i]=(int(threadIdx.x)+i*128<Block && channel<p.out_channels && p.bias)?p.bias[channel]:0.0f;
      }
      TM_FWD_TRACE_BEGIN(load_trace);
      if(threadIdx.x==0) {
        Async::Wait(state.empty,ap);
        Async::ExpectTx(state.full,kABytes);
        Async::Bulk(activation,p.input+std::size_t(row)*K,kABytes,state.full);
        ap^=1;
      }
      Async::Wait(state.full,ac);ac^=1;
      TM_FWD_WINDOW_END(load_trace,false);
      for(int panel=0;panel<panels;++panel) {
        TM_FWD_TRACE_BEGIN(load_trace);
        if(threadIdx.x==0) {
          Async::Wait(state.empty+1,wp);
          Async::ExpectTx(state.full+1,kPageBytes);
          Async::Bulk(page,p.weight+(std::size_t(block)*panels+panel)*Block*kTileK,
              kPageBytes,state.full+1);
          wp^=1;
        }
        Async::Wait(state.full+1,wc);wc^=1;
        TM_FWD_WINDOW_END(load_trace,task+int(gridDim.x)>=total && panel+1==panels);
        if(task==int(blockIdx.x) && panel==0) {TM_FWD_TRACE_FIRST_READY();}
        TM_FWD_TRACE_BEGIN(compute_trace);
        #pragma unroll
        for(int i=0;i<kOutputs;++i) {
          int within=int(threadIdx.x)+i*128,channel=block*Block+within;
          if(within<Block && channel<p.out_channels) {
            // Preserve the old per-output order: bias, then ascending panel/c.
            for(int c=0;c<kTileK;++c)
              sums[i]+=float(page[Swizzled(within,c)])*float(activation[panel*kTileK+c]);
          }
        }
        codegen::executor::ComputeSync();
        TM_FWD_WINDOW_END(compute_trace,task+int(gridDim.x)>=total && panel+1==panels);
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+1);
      }
      TM_FWD_TRACE_BEGIN(store_trace);
      #pragma unroll
      for(int i=0;i<kOutputs;++i) {
        int within=int(threadIdx.x)+i*128,channel=block*Block+within;
        if(within<Block && channel<p.out_channels)
          p.output[std::size_t(row)*p.out_channels+channel]=Element(sums[i]);
      }
      codegen::executor::ComputeSync();
      TM_FWD_WINDOW_END(store_trace,task+int(gridDim.x)>=total);
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty);
    }
    if(threadIdx.x==0) {
      state.producer_generations[0]=ap;state.consumer_generations[0]=ac;
      state.producer_generations[1]=wp;state.consumer_generations[1]=wc;
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

}  // namespace tilemega::backend
