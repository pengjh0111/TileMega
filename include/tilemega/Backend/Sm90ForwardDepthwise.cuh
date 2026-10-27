// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>
#include <tilemega/Codegen/executor/Async.cuh>

// Native depthwise 3x3 body (prompt §10 N-1 item 3). A task owns one spatial
// tile of CT channels. The producer reads the tile's halo through a tiled 4D
// NHWC map, whose box covers ((TW-1) stride + 3) x ((TH-1) stride + 3) and
// whose start may be negative: out-of-range rows arrive as zeros, which is
// exactly the zero padding. The 9 x CT weights are read once per task; the
// consumers accumulate in fp32 from bf16x2 and run the epilogue chain.
namespace tilemega::backend {

struct Sm90ForwardDepthwiseOperands {
  CUtensorMap const* activation;   // 4D NHWC tiled map
  cutlass::bfloat16_t const* weight;   // [3][3][C]
  cutlass::bfloat16_t* output;
  Sm90ForwardEpilogueOperands epilogue;
  int width,height,channels,stride,output_width,output_height;
};

template<int TW,int TH,int CT,int Stride,int Slots,class Chain>
struct Sm90ForwardDepthwise {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert((Stride==1 || Stride==2) && TW>0 && TH>0 && CT%8==0);
  static_assert(Slots>=2 && Slots<=4);
  static constexpr int kThreads=160;
  static constexpr int kHaloW=(TW-1)*Stride+3,kHaloH=(TH-1)*Stride+3;
  static constexpr int kSlotElements=kHaloW*kHaloH*CT;
  static constexpr int kSlotBytes=2*kSlotElements;
  static constexpr int kWeightBytes=2*9*CT;
  static constexpr int kScratchBytes=4*TW*TH*CT;
  static constexpr Sm90DenseLayout kLayout{1024,1024,
      1024+Slots*kSlotBytes+kWeightBytes,
      (1024+Slots*kSlotBytes+kWeightBytes+kScratchBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  __device__ static __forceinline__ void Produce(Sm90ForwardDepthwiseOperands const& p,
      Sm90DenseRingState& state,Element* halo,Element* weights,int tasks,unsigned phase,
      int tiles_w) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x!=128) {__syncwarp();return;}
    Async::ProxyAsyncGlobalFence();
    // The weights outlive every slot: one bulk read before the first task.
    if(tasks) {
      Async::ExpectTx(state.side+0,kWeightBytes);
      Async::Bulk(weights,p.weight,kWeightBytes,state.side+0);
    }
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int tile_x=(task%tiles_w)*TW,tile_y=(task/tiles_w)*TH;
      int slot=local%Slots;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      Async::ExpectTx(state.full+slot,kSlotBytes);
      Async::Tensor4D(halo+slot*kSlotElements,p.activation,0,
          tile_x*Stride-1,tile_y*Stride-1,0,state.full+slot);
      phase^=1u<<slot;
    }
    if(tasks)for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
    __syncwarp();
  }

  __device__ static __forceinline__ void Consume(Sm90ForwardDepthwiseOperands const& p,
      Sm90DenseRingState& state,Element* halo,Element* weights,char* scratch,int tasks,
      unsigned phase,int tiles_w) {
    Sm90DenseRoles::Consumer();
    auto* tile=reinterpret_cast<float*>(scratch);
    unsigned side=state.side_generations[0];
    if(tasks) {Async::Wait(state.side+0,side&1);side^=1u;}
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int tile_x=(task%tiles_w)*TW,tile_y=(task/tiles_w)*TH;
      int slot=local%Slots;
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      auto const* source=halo+slot*kSlotElements;
      for(int i=int(threadIdx.x);i<TW*TH*CT;i+=128) {
        int channel=i%CT,pixel=i/CT,x=pixel%TW,y=pixel/TW;
        float sum=0;
        #pragma unroll
        for(int r=0;r<3;++r)
          #pragma unroll
          for(int s=0;s<3;++s) {
            int hx=x*Stride+s,hy=y*Stride+r;
            sum+=float(source[(hy*kHaloW+hx)*CT+channel])*float(weights[(r*3+s)*CT+channel]);
          }
        tile[pixel*CT+channel]=sum;
      }
      codegen::executor::ComputeSync();
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty+slot);
      // Rows of the epilogue tile are pixels; its channels start at zero.
      int rows=min(TW*TH,(p.output_width-tile_x)*(p.output_height-tile_y));
      Chain::Finish(tile,rows>0?rows:0,0,CT,p.epilogue,
          p.output+(std::size_t(tile_y)*p.output_width+tile_x)*p.channels,p.channels);
    }
    if(threadIdx.x==0 && tasks) {
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
      state.side_generations[0]=side&1;
    }
  }

  __device__ static __forceinline__ void Run(Sm90ForwardDepthwiseOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    auto* halo=reinterpret_cast<Element*>(shared+1024);
    auto* weights=halo+Slots*kSlotElements;
    auto* scratch=shared+kLayout.scratch;
    int tiles_w=(p.output_width+TW-1)/TW,tiles_h=(p.output_height+TH-1)/TH;
    int total=tiles_w*tiles_h*(p.channels/CT);
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    if(threadIdx.x<128)Consume(p,state,halo,weights,scratch,tasks,phase,tiles_w);
    else Produce(p,state,halo,weights,tasks,phase,tiles_w);
    __syncthreads();
  }
};

}  // namespace tilemega::backend
