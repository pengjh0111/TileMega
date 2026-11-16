// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90ForwardMark.cuh>
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
  // The input's padded channel block (what the halo map walks) and the
  // output row's pitch.
  int padded_channels,output_stride;
};

template<int TW,int TH,int CT,int Stride,int Slots,class Chain,int Pack=8,int Batch=1>
struct Sm90ForwardDepthwise {
  static_assert(Batch>=1);
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert((Stride==1 || Stride==2) && TW>0 && TH>0 && CT%8==0);
  static_assert(Slots>=2 && Slots<=8);
  static constexpr int kThreads=160;
  static constexpr int kHaloW=(TW-1)*Stride+3,kHaloH=(TH-1)*Stride+3;
  // A slot holds one channel block's halo and that block's filters, which
  // the plan packs KRSC: [C][3][3][Pack].
  static_assert(Pack>=1 && Pack<=8);
  static constexpr int kHaloElements=kHaloW*kHaloH*CT;
  static constexpr int kFilterElements=CT*9*Pack;
  static constexpr int kSlotElements=kHaloElements+kFilterElements;
  static constexpr int kSlotBytes=2*kSlotElements;
  static constexpr int kScratchBytes=4*TW*TH*CT+Chain::kScratchBytes;
  static constexpr Sm90DenseLayout kLayout{1024,1024,
      1024+Slots*kSlotBytes,
      (1024+Slots*kSlotBytes+kScratchBytes+1023)/1024*1024};
  static constexpr int kSharedBytes=kLayout.end;

  // Task t covers spatial tile t % tiles of channel block t / tiles.
  __device__ static __forceinline__ void Produce(Sm90ForwardDepthwiseOperands const& p,
      Sm90DenseRingState& state,Element* slots,int tasks,unsigned phase,int tiles_w,int tiles) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x!=128) {__syncwarp();return;}
    Async::ProxyAsyncGlobalFence();
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int blocks=(p.channels+CT-1)/CT,image=task/(tiles*blocks);
      int tile=task%tiles,block=(task/tiles)%blocks;
      int tile_x=(tile%tiles_w)*TW,tile_y=(tile/tiles_w)*TH;
      int slot=local%Slots;
      int channels=p.channels-block*CT;
      if(channels>CT)channels=CT;
      if(channels<0)channels=0;
      int filter_bytes=2*channels*9*Pack;
      auto* base=slots+slot*kSlotElements;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      Async::ExpectTx(state.full+slot,2*kHaloElements+filter_bytes);
      if constexpr(Chain::kGated) {
        // SimpleGate pairs the two logical C/2 halves, even when they belong
        // to different channel blocks. The two half-width halos keep the
        // same total slot bytes as an ordinary CT-channel halo.
        static_assert(CT%(2*Chain::kGateUnit)==0);
        Async::Tensor4D(base,p.activation,block*(CT/2),tile_x*Stride-1,tile_y*Stride-1,image,
            state.full+slot);
        Async::Tensor4D(base+kHaloElements/2,p.activation,p.channels/2+block*(CT/2),
            tile_x*Stride-1,tile_y*Stride-1,image,state.full+slot);
      }else {
        Async::Tensor4D(base,p.activation,block*CT,tile_x*Stride-1,tile_y*Stride-1,image,
            state.full+slot);
      }
      if(filter_bytes)
        Async::Bulk(base+kHaloElements,p.weight+std::size_t(block)*CT*9*Pack,filter_bytes,
            state.full+slot);
      phase^=1u<<slot;
    }
    if(tasks)for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
    __syncwarp();
  }

  __device__ static __forceinline__ void Consume(Sm90ForwardDepthwiseOperands const& p,
      Sm90DenseRingState& state,Element* slots,char* scratch,int tasks,unsigned phase,
      int tiles_w,int tiles) {
    Sm90DenseRoles::Consumer();
    auto* values=reinterpret_cast<float*>(scratch);
    TM_FWD_TRACE_ACC(wait_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(epilogue_trace);
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x);
      int blocks=(p.channels+CT-1)/CT,image=task/(tiles*blocks);
      int tile=task%tiles,block=(task/tiles)%blocks;
      int tile_x=(tile%tiles_w)*TW,tile_y=(tile/tiles_w)*TH;
      // An edge tile is narrower than TW: its pixels are decomposed by its
      // own width, and the store walks the same tile.
      int tile_w=min(TW,p.output_width-tile_x),tile_h=min(TH,p.output_height-tile_y);
      int rows=tile_w*tile_h;
      int slot=local%Slots;
      TM_FWD_TRACE_BEGIN(wait_trace);
      Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
      TM_FWD_TRACE_END(wait_trace);
      if(local==0) {TM_FWD_TRACE_FIRST_READY();}
      TM_FWD_TRACE_BEGIN(compute_trace);
      auto const* halo=slots+slot*kSlotElements;
      auto const* filters=halo+kHaloElements;
      for(int i=int(threadIdx.x);i<TW*TH*CT;i+=128) {
        int channel=i%CT,pixel=i/CT;
        float sum=0;
        if(pixel<rows && block*CT+channel<p.channels) {
          int x=pixel%tile_w,y=pixel/tile_w;
          auto const* filter=filters+std::size_t(channel)*9*Pack;
          #pragma unroll
          for(int r=0;r<3;++r)
            #pragma unroll
            for(int s=0;s<3;++s) {
              int hx=x*Stride+s,hy=y*Stride+r;
              int offset;
              if constexpr(Chain::kGated) {
                constexpr int unit=Chain::kGateUnit;
                int band=(channel/unit)%2,pair=(channel/(2*unit))*unit+channel%unit;
                offset=band*(kHaloElements/2)+(hy*kHaloW+hx)*(CT/2)+pair;
              }else offset=(hy*kHaloW+hx)*CT+channel;
              sum+=float(halo[offset])*float(filter[(r*3+s)*Pack]);
            }
        }
        values[pixel*CT+channel]=sum;
      }
      codegen::executor::ComputeSync();
      TM_FWD_TRACE_END(compute_trace);
      Async::FenceProxyAsyncShared();
      Async::Arrive(state.empty+slot);
      auto epilogue=p.epilogue;
      epilogue.tile_width=tile_w;epilogue.image_width=p.output_width;
      epilogue.row_base=(image*p.output_height+tile_y)*p.output_width+tile_x;
      if(epilogue.residual)
        epilogue.residual+=((std::size_t(image)*p.output_height+tile_y)*p.output_width+tile_x)*epilogue.residual_stride;
      // The block's channels in the producer's packed space; a gate pair in
      // the chain halves them, and Finish stores at the halved offset.
      int width=p.channels-block*CT;
      if(width>CT)width=CT;
      if(Chain::kGated)width/=2;
      TM_FWD_TRACE_BEGIN(epilogue_trace);
      Chain::Finish(values,rows,block*CT,width>0?width:0,epilogue,
          p.output+((std::size_t(image)*p.output_height+tile_y)*p.output_width+tile_x)*p.output_stride,p.output_stride);
      TM_FWD_TRACE_END(epilogue_trace);
    }
    TM_FWD_TRACE_FLUSH(wait_trace,kHaloWait);TM_FWD_TRACE_FLUSH(compute_trace,kDepthwiseCompute);
    TM_FWD_TRACE_FLUSH(epilogue_trace,kEpilogue);
    if(threadIdx.x==0 && tasks)
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
  }

  __device__ static __forceinline__ void Run(Sm90ForwardDepthwiseOperands const& p,char* shared,
      Sm90DenseRingState& state) {
    auto* slots=reinterpret_cast<Element*>(shared+1024);
    auto* scratch=shared+kLayout.scratch;
    int tiles_w=(p.output_width+TW-1)/TW,tiles_h=(p.output_height+TH-1)/TH;
    int tiles=tiles_w*tiles_h,blocks=(p.channels+CT-1)/CT;
    int total=Batch*tiles*blocks;
    int tasks=total/int(gridDim.x)+(int(blockIdx.x)<total%int(gridDim.x));
    unsigned phase=0;
    // Only the issuing producer reads its phase. Other producer lanes do not
    // use it and could otherwise read while lane 0 publishes its next phase.
    if(threadIdx.x<128) {
      #pragma unroll
      for(int i=0;i<Slots;++i)phase|=state.consumer_generations[i]<<i;
    }else if(threadIdx.x==128) {
      #pragma unroll
      for(int i=0;i<Slots;++i)phase|=state.producer_generations[i]<<i;
    }
    if(threadIdx.x<128)Consume(p,state,slots,scratch,tasks,phase,tiles_w,tiles);
    else Produce(p,state,slots,tasks,phase,tiles_w,tiles);
    __syncthreads();
  }
};

}  // namespace tilemega::backend
