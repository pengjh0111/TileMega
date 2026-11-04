// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90ForwardMark.cuh>
#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
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

// NCHW -> NHWC. One task per pixel; the write is coalesced over channels,
// which every later body reads as its inner dimension. The image's own
// channels (`out_channels`) are copied into the padded 64-channel block
// (`channels`) every native A panel reads; the padding stays zero.
template<int Channels,int Block>
struct Sm90ForwardLayout {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
    int pixels=p.width*p.height;
    for(int task=int(blockIdx.x);task<pixels;task+=int(gridDim.x))
      for(int c=int(threadIdx.x);c<p.out_channels;c+=128) {
#if TILEMEGA_TRACE_STAGE
        TM_FWD_TRACE_BEGIN(load_trace);
        Element value=p.input[std::size_t(c)*pixels+task];
        TM_FWD_TRACE_END(load_trace);
        TM_FWD_TRACE_POINT(compute_trace);  // A layout copy has no arithmetic phase.
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(task)*p.channels+c]=value;
        TM_FWD_TRACE_END(store_trace);
#else
        p.output[std::size_t(task)*p.channels+c]=p.input[std::size_t(c)*pixels+task];
#endif
      }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// 3x3 stride-2 max pool with one-pixel padding (ResNet18's stem pool).
template<int Window,int Stride>
struct Sm90ForwardMaxPool {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
    int out_w=(p.width+1)/Stride,out_h=(p.height+1)/Stride;
    int total=out_w*out_h;
    for(int task=int(blockIdx.x);task<total;task+=int(gridDim.x)) {
      int x=(task%out_w)*Stride-1,y=(task/out_w)*Stride-1;
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
            float value=float(p.input[(std::size_t(sy)*p.width+sx)*p.channels+c]);
            TM_FWD_TRACE_END(load_trace);
            TM_FWD_TRACE_BEGIN(compute_trace);
            best=fmaxf(best,value);
            TM_FWD_TRACE_END(compute_trace);
#else
            best=fmaxf(best,float(p.input[(std::size_t(sy)*p.width+sx)*p.channels+c]));
#endif
          }
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(task)*p.channels+c]=Element(best);
        TM_FWD_TRACE_END(store_trace);
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
struct Sm90ForwardGlobalPool {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
    float inverse=1.0f/float(p.height);
    for(int c=int(blockIdx.x)*128+int(threadIdx.x);c<p.channels;c+=int(gridDim.x)*128) {
      float sum=0;
      for(int part=0;part<p.width;++part) {
        auto* address=p.accumulator+std::size_t(part)*p.channels+c;
#if TILEMEGA_TRACE_STAGE
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=*address;
        TM_FWD_TRACE_END(load_trace);
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum+=value;
        TM_FWD_TRACE_END(compute_trace);
#else
        sum+=*address;
#endif
        // The producing epilogue accumulates into these partials, so each
        // launch must start from zero: the only reader clears them.
        TM_FWD_TRACE_BEGIN(store_trace);
        *address=0.0f;
        TM_FWD_TRACE_END(store_trace);
      }
      TM_FWD_TRACE_BEGIN(store_trace);
      p.output[c]=Element(sum*inverse);
      TM_FWD_TRACE_END(store_trace);
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// Output GEMV: one task per output block of one row (ResNet18's 1000 classes
// give 125 blocks). The weight is the same K-major tile page layout every
// other native body reads, with the block as its N tile.
template<int Block>
struct Sm90ForwardGemv {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160,kTileK=64;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  // The element's position inside one K-major tile page, which the packer
  // writes in the same 128-byte swizzle the WGMMA bodies read.
  __device__ static __forceinline__ int Swizzled(int row,int col) {
    return 512*(row/8+(Block/8)*(col/64))+64*(row%8)+
        8*(((col%64)/8)^(row%8))+col%8;
  }
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
    int blocks=(p.out_channels+Block-1)/Block,panels=p.channels/kTileK;
    int rows=p.width>0?p.width:1,total=blocks*rows;
    for(int task=int(blockIdx.x);task<total;task+=int(gridDim.x)) {
      int row=task/blocks,block=task%blocks;
      auto const* source=p.input+std::size_t(row)*p.channels;
      for(int i=int(threadIdx.x);i<Block;i+=128) {
        int channel=block*Block+i;
        if(channel>=p.out_channels)continue;
        float sum=p.bias?float(p.bias[channel]):0.0f;
        for(int panel=0;panel<panels;++panel) {
          auto const* page=p.weight+(std::size_t(block)*panels+panel)*Block*kTileK;
          for(int c=0;c<kTileK;++c)
#if TILEMEGA_TRACE_STAGE
          {
            TM_FWD_TRACE_BEGIN(load_trace);
            float weight=float(page[Swizzled(i,c)]),value=float(source[panel*kTileK+c]);
            TM_FWD_TRACE_END(load_trace);
            TM_FWD_TRACE_BEGIN(compute_trace);
            sum+=weight*value;
            TM_FWD_TRACE_END(compute_trace);
          }
#else
            sum+=float(page[Swizzled(i,c)])*float(source[panel*kTileK+c]);
#endif
        }
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(row)*p.out_channels+channel]=Element(sum);
        TM_FWD_TRACE_END(store_trace);
      }
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
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
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
        TM_FWD_TRACE_END(load_trace);
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(row)*Width+c]=Element(value);
        TM_FWD_TRACE_END(store_trace);
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum+=value;squares+=value*value;
        TM_FWD_TRACE_END(compute_trace);
      }
      if(p.stats) {
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum=Sm90DenseAttentionMath::WarpSum(sum);
        squares=Sm90DenseAttentionMath::WarpSum(squares);
        TM_FWD_TRACE_END(compute_trace);
        TM_FWD_TRACE_BEGIN(store_trace);
        if(!lane) {p.stats[2*row]=sum;p.stats[2*row+1]=squares;}
        TM_FWD_TRACE_END(store_trace);
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

// The explicit LayerNorm that stays a stage only for a model's own output
// (every consumed LayerNorm is deferred into its consumer's epilogue).
template<int Channels>
struct Sm90ForwardLayerNorm {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=160;
  static constexpr Sm90DenseLayout kLayout{1024,1024,1024,1024};
  static constexpr int kSharedBytes=1024;
  __device__ static __forceinline__ void Run(Sm90ForwardSmallOperands const& p,char*,
      Sm90DenseRingState&) {
    if(threadIdx.x>=128)return;
    TM_FWD_TRACE_ACC(load_trace);TM_FWD_TRACE_ACC(compute_trace);
    TM_FWD_TRACE_ACC(store_trace);
    int rows=p.width*p.height;
    // One warp per row: the row fits a single pass over Channels.
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=int(blockIdx.x)*4+warp;row<rows;row+=int(gridDim.x)*4) {
      auto const* source=p.input+std::size_t(row)*Channels;
      float sum=0,squares=0;
      for(int c=lane;c<Channels;c+=32) {
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=float(source[c]);
        TM_FWD_TRACE_END(load_trace);
        TM_FWD_TRACE_BEGIN(compute_trace);
        sum+=value;squares+=value*value;
        TM_FWD_TRACE_END(compute_trace);
      }
      TM_FWD_TRACE_BEGIN(compute_trace);
      sum=Sm90DenseAttentionMath::WarpSum(sum);
      squares=Sm90DenseAttentionMath::WarpSum(squares);
      float mean=sum/Channels;
      float inverse=rsqrtf(squares/Channels-mean*mean+p.epsilon);
      TM_FWD_TRACE_END(compute_trace);
      for(int c=lane;c<Channels;c+=32) {
#if TILEMEGA_TRACE_STAGE
        TM_FWD_TRACE_BEGIN(load_trace);
        float value=float(source[c]);
        TM_FWD_TRACE_END(load_trace);
        // Affine parameter reads stay interleaved with the original expression.
        TM_FWD_TRACE_BEGIN(compute_trace);
        auto result=Element((value-mean)*inverse*
            float(p.weight[c])+(p.table?float(p.table[c]):0.0f));
        TM_FWD_TRACE_END(compute_trace);
        TM_FWD_TRACE_BEGIN(store_trace);
        p.output[std::size_t(row)*Channels+c]=result;
        TM_FWD_TRACE_END(store_trace);
#else
        p.output[std::size_t(row)*Channels+c]=Element((float(source[c])-mean)*inverse*
            float(p.weight[c])+(p.table?float(p.table[c]):0.0f));
#endif
      }
    }
    TM_FWD_TRACE_FLUSH(load_trace,kSmallLoad);TM_FWD_TRACE_FLUSH(compute_trace,kSmallCompute);
    TM_FWD_TRACE_FLUSH(store_trace,kSmallStore);
  }
};

}  // namespace tilemega::backend
