// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90ForwardMark.cuh>
#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90ForwardResources.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Codegen/executor/Async.cuh>

// The small native forward bodies of prompt §10 N-1 item 7: the NCHW -> NHWC
// input conversion, max pooling, the global-average last arriver, the output
// GEMV, and the explicit LayerNorm kept only where a model's output needs it.
// Each is one stage of the forward program and uses no ring: their operands
// are read directly, so they add no slot bookkeeping to the boundary.
namespace tilemega::backend {

struct Sm90ForwardSmallOperands {
  cutlass::bfloat16_t const* input;
  cutlass::bfloat16_t const* weight;   // GEMV weights, or none
  float const* gamma;                  // explicit LayerNorm scale (FP32)
  cutlass::bfloat16_t const* table;    // EMB: the position embedding table
  float* stats;                        // EMB: per-row (sum, sum of squares)
  float const* bias;                   // per-channel vectors are FP32
  cutlass::bfloat16_t* output;
  float* accumulator;                  // GAP partials
  std::uint64_t* counter;              // GAP last arriver
  long long const* index0;             // EMB: token ids (the export's int64)
  long long const* index1;             // EMB: token types
  int width,height,channels,out_channels;
  float epsilon;
};

// NCHW -> NHWC. One task covers a 128-pixel block of one image;
// which every later body reads as its inner dimension. The image's own
// channels (`out_channels`) are copied into the padded 64-channel block
// (`channels`) every native A panel reads; the padding stays zero.
template<int Channels,int Block,int Batch=1>
struct Sm90ForwardLayout {
  using Element=cutlass::bfloat16_t;
  static_assert(Block==128 && Batch>=1);
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);TM_FWD_WINDOW_ACC(store_trace);
    int pixels=p.width*p.height,blocks=(pixels+Block-1)/Block;
    for(int task=int(blockIdx.x);task<Batch*blocks;task+=int(gridDim.x)) {
      int image=task/blocks,base=(task%blocks)*Block;
      for(int i=int(threadIdx.x);i<Block*p.out_channels;i+=128) {
        int pixel=base+i/p.out_channels,c=i%p.out_channels;
        if(pixel>=pixels)continue;
        TM_FWD_TRACE_BEGIN(load_trace);
        Element value=p.input[(std::size_t(image)*p.out_channels+c)*pixels+pixel];
        TM_FWD_WINDOW_END(load_trace,task+int(gridDim.x)>=Batch*blocks && i+128>=min(Block,pixels-base)*p.out_channels);TM_FWD_WINDOW_POINT(compute_trace,task+int(gridDim.x)>=Batch*blocks && i+128>=min(Block,pixels-base)*p.out_channels);
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[(std::size_t(image)*pixels+pixel)*p.channels+c]=value;
        TM_FWD_WINDOW_END(store_trace,task+int(gridDim.x)>=Batch*blocks && i+128>=min(Block,pixels-base)*p.out_channels);
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// 3x3 stride-2 max pool with one-pixel padding (ResNet18's stem pool).
template<int Window,int Stride,int Batch=1>
struct Sm90ForwardMaxPool {
  static_assert(Batch>=1 && Window==3 && Stride==2);
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);
    TM_FWD_WINDOW_ACC(store_trace);
    int out_w=(p.width+1)/Stride,out_h=(p.height+1)/Stride;
    int pixels=out_w*out_h,total=Batch*pixels;
    for(int task=int(blockIdx.x);task<total;task+=int(gridDim.x)) {
      int image=task/pixels,pixel=task%pixels;
      int x=(pixel%out_w)*Stride-1,y=(pixel/out_w)*Stride-1;
      for(int c=int(threadIdx.x);c<p.channels;c+=128) {
        float best=-INFINITY;
        #pragma unroll
        for(int r=0;r<Window;++r)
          #pragma unroll
          for(int s=0;s<Window;++s) {
            int sx=x+s,sy=y+r;
            if(sx<0 || sy<0 || sx>=p.width || sy>=p.height)continue;
#if TILEMEGA_TRACE_STAGE
            TM_FWD_TRACE_BEGIN(load_trace);
            float value=float(p.input[((std::size_t(image)*p.height+sy)*p.width+sx)*p.channels+c]);
            TM_FWD_WINDOW_END(load_trace,task+int(gridDim.x)>=total && c+128>=p.channels && r==min(Window-1,p.height-1-y) && s==min(Window-1,p.width-1-x));
            TM_FWD_TRACE_BEGIN(compute_trace);
            best=fmaxf(best,value);
            TM_FWD_WINDOW_END(compute_trace,task+int(gridDim.x)>=total && c+128>=p.channels && r==min(Window-1,p.height-1-y) && s==min(Window-1,p.width-1-x));
#else
            best=fmaxf(best,float(p.input[((std::size_t(image)*p.height+sy)*p.width+sx)*p.channels+c]));
#endif
          }
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(task)*p.channels+c]=Element(best);
        TM_FWD_WINDOW_END(store_trace,task+int(gridDim.x)>=total && c+128>=p.channels);
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// Global average pooling: the producing convolution's epilogue has already
// written this image's channel partial sums (prompt §10 N-1 item 7), so the
// stage is their reduction. `accumulator` is that fp32 [parts][channels]
// buffer, `width` its part count and `height` the pooled pixel count.
template<int Batch=1>
struct Sm90ForwardGlobalPool {
  static_assert(Batch>=1);
  using Element=cutlass::bfloat16_t;
  // Load 32 partial rows cooperatively, then retain the original part order.
  static constexpr int kParts=32,kColumns=128,kThreads=160;
  static constexpr auto kLayout=Sm90ForwardGlobalPoolLayout();
  static constexpr int kSharedBytes=kLayout.end;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char* shared,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);TM_FWD_WINDOW_ACC(store_trace);
    auto* tile=reinterpret_cast<float*>(shared+1024);
    int tid=int(threadIdx.x),blocks=(p.channels+127)/128,total=Batch*blocks;
    float inverse=1.0f/float(p.height);
    for(int task=int(blockIdx.x);task<total;task+=int(gridDim.x)) {
      int image=task/blocks,first=(task%blocks)*128,live=min(128,p.channels-first);
      float sum=0;
      // One or two partials have no latency chain to hide. This small-shape
      // specialization avoids the cooperative path's synchronization cost.
      if(p.width<=2) {
        if(tid<live)for(int part=0;part<p.width;++part) {
          auto* address=p.accumulator+(std::size_t(image)*p.width+part)*p.channels+first+tid;
          TM_FWD_TRACE_BEGIN(load_trace);float value=*address;
          TM_FWD_WINDOW_END(load_trace,task+int(gridDim.x)>=total && part+1==p.width);
          TM_FWD_TRACE_BEGIN(compute_trace);sum+=value;
          TM_FWD_WINDOW_END(compute_trace,task+int(gridDim.x)>=total && part+1==p.width);
          TM_FWD_TRACE_BEGIN(store_trace);*address=0.f;TM_FWD_WINDOW_END(store_trace,false);
        }
      }else for(int base=0;base<p.width;base+=kParts) {
        int count=min(kParts,p.width-base);
        TM_FWD_TRACE_BEGIN(load_trace);
        for(int i=tid;i<count*live;i+=128) {
          int row=i/live,col=i%live;
          tile[row*kColumns+col]=p.accumulator[(std::size_t(image)*p.width+base+row)*p.channels+first+col];
        }
        TM_FWD_WINDOW_END(load_trace,task+int(gridDim.x)>=total && base+count==p.width);
        TM_FWD_TRACE_BEGIN(store_trace);
        for(int i=tid;i<count*live;i+=128) {
          int row=i/live,col=i%live;
          p.accumulator[(std::size_t(image)*p.width+base+row)*p.channels+first+col]=0.f;
        }
        TM_FWD_WINDOW_END(store_trace,false);
        codegen::executor::ComputeSync();
        TM_FWD_TRACE_BEGIN(compute_trace);
        if(tid<live)for(int part=0;part<count;++part)sum+=tile[part*kColumns+tid];
        TM_FWD_WINDOW_END(compute_trace,task+int(gridDim.x)>=total && base+count==p.width);
        codegen::executor::ComputeSync();
      }
      TM_FWD_TRACE_BEGIN(store_trace);
      if(tid<live)p.output[std::size_t(image)*p.out_channels+first+tid]=Element(sum*inverse);
      TM_FWD_WINDOW_END(store_trace,task+int(gridDim.x)>=total);
      // The program's grid-barrier leader must follow every channel store.
      codegen::executor::ComputeSync();
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// EMB: one warp per row sums the word, token-type and position embeddings
// (prompt §10 N-1 item 7). A consumer that defers this row's LayerNorm reads
// the row's (sum, sum of squares), which the row's own warp writes once: a
// single writer per row keeps the statistics exact and launch-independent.
template<int Width>
struct Sm90ForwardEmbeddingSum {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160,kWarps=4;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);
    TM_FWD_WINDOW_ACC(store_trace);
    int rows=p.width,warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=int(blockIdx.x)*kWarps+warp;row<rows;row+=int(gridDim.x)*kWarps) {
      int token=int(p.index0[row]),type=p.index1?int(p.index1[row]):0;
      auto const* word=p.input+std::size_t(token)*Width;
      auto const* types=p.weight+std::size_t(type)*Width;
      auto const* position=p.table+std::size_t(row%p.height)*Width;
      float sum=0,squares=0;
      for(int c=lane;c<Width;c+=32) {
        // The embedding load envelope includes its original additions.
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=float(word[c])+float(types[c])+float(position[c]);
        TM_FWD_WINDOW_END(load_trace,row+int(gridDim.x)*kWarps>=rows && c+32>=Width);
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(row)*Width+c]=Element(value);
        TM_FWD_WINDOW_END(store_trace,!p.stats && row+int(gridDim.x)*kWarps>=rows && c+32>=Width);
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum+=value;squares+=value*value;
        TM_FWD_WINDOW_END(compute_trace,!p.stats && row+int(gridDim.x)*kWarps>=rows && c+32>=Width);
      }
      if(p.stats) {
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum=Sm90DenseAttentionMath::WarpSum(sum);
        squares=Sm90DenseAttentionMath::WarpSum(squares);
        TM_FWD_WINDOW_END(compute_trace,row+int(gridDim.x)*kWarps>=rows);
        TM_FWD_TRACE_BEGIN(store_trace);
        if(!lane) {p.stats[2*row]=sum;p.stats[2*row+1]=squares;}
        TM_FWD_WINDOW_END(store_trace,row+int(gridDim.x)*kWarps>=rows);
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// The explicit LayerNorm that stays a stage only for a model's own output
// (every consumed LayerNorm is deferred into its consumer's epilogue).
template<int Channels,int Stride=Channels>
struct Sm90ForwardLayerNorm {
  using Element=cutlass::bfloat16_t;
  static_assert(Channels>0 && Stride>=Channels);
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_WINDOW_ACC(load_trace);TM_FWD_WINDOW_ACC(compute_trace);
    TM_FWD_WINDOW_ACC(store_trace);
    int rows=p.width*p.height;
    // One warp per row: the row fits a single pass over Channels.
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=int(blockIdx.x)*4+warp;row<rows;row+=int(gridDim.x)*4) {
      auto const* source=p.input+std::size_t(row)*Stride;
      float sum=0,squares=0;
      for(int c=lane;c<Channels;c+=32) {
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=float(source[c]);
        TM_FWD_WINDOW_END(load_trace,false);
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum+=value;squares+=value*value;
        TM_FWD_WINDOW_END(compute_trace,false);
      }
      TM_FWD_TRACE_BEGIN(compute_trace);
      sum=Sm90DenseAttentionMath::WarpSum(sum);
      squares=Sm90DenseAttentionMath::WarpSum(squares);
      float mean=sum/Channels;
      float inverse=rsqrtf(squares/Channels-mean*mean+p.epsilon);
      TM_FWD_WINDOW_END(compute_trace,false);
      for(int c=lane;c<Channels;c+=32) {
#if TILEMEGA_TRACE_STAGE
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=float(source[c]);
        TM_FWD_WINDOW_END(load_trace,row+int(gridDim.x)*4>=rows && c+32>=Channels);
        // Affine parameter reads stay interleaved with the original expression.
        TM_FWD_TRACE_BEGIN(compute_trace);
        auto result=Element((value-mean)*inverse*
            float(p.weight[c])+(p.table?float(p.table[c]):0.0f));
        TM_FWD_WINDOW_END(compute_trace,row+int(gridDim.x)*4>=rows && c+32>=Channels);
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(row)*Stride+c]=result;
        TM_FWD_WINDOW_END(store_trace,row+int(gridDim.x)*4>=rows && c+32>=Channels);
#else
        p.output[std::size_t(row)*Stride+c]=Element((float(source[c])-mean)*inverse*
            float(p.weight[c])+(p.table?float(p.table[c]):0.0f));
#endif
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

}  // namespace tilemega::backend
