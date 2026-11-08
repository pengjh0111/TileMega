// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <cutlass/bfloat16.h>
#include <cuda_bf16.h>
#include <cstddef>

// The F3 small bodies (DNN3 prompt §8): the NCHW to NHWC8 conversion, a
// generalized max pool, the global-average last arriver, the explicit
// LayerNorm a model's own output needs, and BERT's embedding sum.
//
// All five use the 256 consumer threads only; their `Produce` is empty and
// they take no ring slot, so they add no bookkeeping to the stage gate.
namespace tilemega::backend {

struct Sm90F3SmallOperands {
  cutlass::bfloat16_t const* input;
  cutlass::bfloat16_t const* weight;    // LN gamma / embedding token types
  cutlass::bfloat16_t const* table;     // LN beta / embedding positions
  float const* vector;                  // FP32 affine, when the plan folds one
  float* stats;                         // embedding: per-row (sum, sum of squares)
  float* accumulator;                   // GAP partial sums, [image][part][channel]
  cutlass::bfloat16_t* output;
  long long const* index0;              // embedding: token ids (the export's int64)
  long long const* index1;              // embedding: token types
  int width,height,channels,out_channels,parts,rows;
  float epsilon;
};

// One 16 B line of eight BF16 values. A plain `uint4` with reinterpreting
// accessors keeps the type trivially copyable, which a union holding
// `__nv_bfloat162` is not.
struct Sm90F3Oct {
  uint4 raw;
  __device__ __forceinline__ cutlass::bfloat16_t* value() {
    return reinterpret_cast<cutlass::bfloat16_t*>(&raw);
  }
  __device__ __forceinline__ cutlass::bfloat16_t const* value() const {
    return reinterpret_cast<cutlass::bfloat16_t const*>(&raw);
  }
  __device__ __forceinline__ __nv_bfloat162* pair() {
    return reinterpret_cast<__nv_bfloat162*>(&raw);
  }
  __device__ __forceinline__ __nv_bfloat162 const* pair() const {
    return reinterpret_cast<__nv_bfloat162 const*>(&raw);
  }
};
__device__ __forceinline__ Sm90F3Oct Sm90F3LoadOct(void const* address) {
  Sm90F3Oct octet;
  octet.raw=*reinterpret_cast<uint4 const*>(address);
  return octet;
}
__device__ __forceinline__ void Sm90F3StoreOct(void* address,Sm90F3Oct const& octet) {
  *reinterpret_cast<uint4*>(address)=octet.raw;
}

// NCHW -> NHWC with an eight-channel block. A work item is one image's block
// of eight pixels: each thread reads a 16 B run of eight pixels from every
// input channel plane, transposes in registers and writes eight 16 B lines.
// The channels past the image's own stay zero, which is what the gather stem's
// taps and the eight-channel A panel expect.
template<int Channels,int Batch>
struct Sm90F3Layout {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=384;
  static constexpr int kBlock=8;
  static_assert(Channels>=1 && Channels<=8,"an F3 layout stage fills one eight-channel block");
  static constexpr int kSharedBytes=1024;
  static constexpr int kSideOffset=0;

  __device__ static __forceinline__ void Produce(Sm90F3SmallOperands const&,
      Sm90F3Ring&,char*,Sm90F3Phase&,int) {}

  __device__ static __forceinline__ void Consume(Sm90F3SmallOperands const& p,
      Sm90F3Ring&,char*,Sm90F3Phase&,Sm90F3Header*,int) {
    int pixels=p.width*p.height;
    int blocks=pixels/kBlock,tail=pixels%kBlock;
    long long items=1LL*Batch*blocks;
    for(long long item=blockIdx.x*256+threadIdx.x;item<items;
        item+=1LL*gridDim.x*256) {
      int image=int(item/blocks),base=int(item%blocks)*kBlock;
      Sm90F3Oct plane[Channels];
      #pragma unroll
      for(int c=0;c<Channels;++c)
        plane[c]=Sm90F3LoadOct(p.input+
            (std::size_t(image)*Channels+c)*pixels+base);
      #pragma unroll
      for(int j=0;j<kBlock;++j) {
        Sm90F3Oct out;
        #pragma unroll
        for(int c=0;c<8;++c)out.value()[c]=c<Channels?plane[c].value()[j]:Element(0.f);
        Sm90F3StoreOct(p.output+(std::size_t(image)*pixels+base+j)*8,out);
      }
    }
    // The pixels past the last whole block, one pixel per thread.
    if(tail) {
      long long items_tail=1LL*Batch*tail;
      for(long long item=blockIdx.x*256+threadIdx.x;item<items_tail;
          item+=1LL*gridDim.x*256) {
        int image=int(item/tail),pixel=blocks*kBlock+int(item%tail);
        Sm90F3Oct out;
        #pragma unroll
        for(int c=0;c<8;++c)
          out.value()[c]=c<Channels?
              p.input[(std::size_t(image)*Channels+c)*pixels+pixel]:Element(0.f);
        Sm90F3StoreOct(p.output+(std::size_t(image)*pixels+pixel)*8,out);
      }
    }
  }
};

// Max pooling over any window, stride and padding the plan names, so VGG16's
// 2x2/s2/p0 and ResNet's 3x3/s2/p1 share one body (§4 F-2). A work item is one
// output pixel's block of eight channels, and each in-bounds tap is a 16 B
// read reduced with `__hmax2` over four BF16 pairs.
template<int Window,int WindowW,int Stride,int Pad,int Channels,int Batch,int Width=0,int Height=0>
struct Sm90F3MaxPool {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=384;
  static constexpr int kSharedBytes=1024;
  static constexpr int kSideOffset=0;
  static_assert(Window>=1 && WindowW>=1 && Stride>=1 && Pad>=0);
  static_assert(Channels%8==0,"a pooled image carries whole eight-channel lines");

  __device__ static __forceinline__ void Produce(Sm90F3SmallOperands const&,
      Sm90F3Ring&,char*,Sm90F3Phase&,int) {}

  __device__ static __forceinline__ void Consume(Sm90F3SmallOperands const& p,
      Sm90F3Ring&,char*,Sm90F3Phase&,Sm90F3Header*,int) {
    int input_w=Width?Width:p.width,input_h=Height?Height:p.height;
    int out_w=(input_w+2*Pad-WindowW)/Stride+1;
    int out_h=(input_h+2*Pad-Window)/Stride+1;
    constexpr int lines=Channels/8;
    unsigned items=unsigned(Batch*out_w*out_h*lines);
    for(unsigned first=4*(blockIdx.x*256+threadIdx.x);first<items;
        first+=4*gridDim.x*256) {
      Sm90F3Oct taps[4][Window*WindowW];
      #pragma unroll
      for(int j=0;j<4;++j) {
        unsigned item=first+j,pixel=item/lines;
        unsigned line=item%lines,x=pixel%out_w,rest=pixel/out_w;
        unsigned y=rest%out_h,image=rest/out_h;
        #pragma unroll
        for(int r=0;r<Window;++r) {
          int sy=int(y)*Stride-Pad+r;
          #pragma unroll
          for(int s=0;s<WindowW;++s) {
            int sx=int(x)*Stride-Pad+s;
            auto& tap=taps[j][r*WindowW+s];
            #pragma unroll
            for(int pair=0;pair<4;++pair)tap.pair()[pair]=__nv_bfloat162(-INFINITY,-INFINITY);
            if(item<items && sy>=0 && sy<input_h && sx>=0 && sx<input_w)
              tap=Sm90F3LoadOct(p.input+((image*input_h+sy)*input_w+sx)*Channels+line*8);
          }
        }
      }
      #pragma unroll
      for(int j=0;j<4;++j) {
        Sm90F3Oct out;
        #pragma unroll
        for(int pair=0;pair<4;++pair) {
          auto best=__nv_bfloat162(-INFINITY,-INFINITY);
          #pragma unroll
          for(int tap=0;tap<Window*WindowW;++tap)best=__hmax2(best,taps[j][tap].pair()[pair]);
          out.pair()[pair]=best;
        }
        if(first+j<items)Sm90F3StoreOct(p.output+(first+j)*8,out);
      }
    }
  }
};

// Global average pooling: the producing GEMM's epilogue already wrote this
// image's per-(64-row block) channel partials, so the stage reduces them in
// ascending part order and divides by the pixel count, exactly as the legacy
// body does, which keeps repeated launches bitwise equal.
template<int Batch>
struct Sm90F3GlobalPool {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=384;
  static constexpr int kSharedBytes=1024;
  static constexpr int kSideOffset=0;

  __device__ static __forceinline__ void Produce(Sm90F3SmallOperands const&,
      Sm90F3Ring&,char*,Sm90F3Phase&,int) {}

  __device__ static __forceinline__ void Consume(Sm90F3SmallOperands const& p,
      Sm90F3Ring&,char*,Sm90F3Phase&,Sm90F3Header*,int) {
    float inverse=Sm90F3Async::DivideNormalization(1.0f,float(p.rows));
    long long items=1LL*Batch*p.channels;
    for(long long item=blockIdx.x*256+threadIdx.x;item<items;
        item+=1LL*gridDim.x*256) {
      int channel=int(item%p.channels),image=int(item/p.channels);
      float sum=0;
      int first=(image*p.rows)/64,last=((image+1)*p.rows-1)/64;
      for(int part=0;part<=last-first;++part)
        sum+=p.accumulator[(std::size_t(image)*p.parts+part)*p.channels+channel];
      p.output[std::size_t(image)*p.out_channels+channel]=Element(sum*inverse);
    }
  }
};

// The explicit LayerNorm a model's own output keeps: one warp per row, 16 B
// reads, and the two passes in the legacy order.
template<int Channels,int Stride>
struct Sm90F3LayerNorm {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=384;
  static constexpr int kSharedBytes=1024;
  static constexpr int kSideOffset=0;
  static_assert(Channels%8==0 && Stride>=Channels);
  static constexpr int kWarps=8,kLines=Channels/8;

  __device__ static __forceinline__ void Produce(Sm90F3SmallOperands const&,
      Sm90F3Ring&,char*,Sm90F3Phase&,int) {}

  __device__ static __forceinline__ void Consume(Sm90F3SmallOperands const& p,
      Sm90F3Ring&,char*,Sm90F3Phase&,Sm90F3Header*,int) {
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=int(blockIdx.x)*kWarps+warp;row<p.rows;row+=int(gridDim.x)*kWarps) {
      auto const* source=p.input+std::size_t(row)*Stride;
      float sum=0,squares=0;
      for(int line=lane;line<kLines;line+=32) {
        auto octet=Sm90F3LoadOct(source+line*8);
        #pragma unroll
        for(int i=0;i<8;++i) {
          float value=float(octet.value()[i]);
          sum+=value;squares+=value*value;
        }
      }
      #pragma unroll
      for(int offset=16;offset;offset>>=1) {
        sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
        squares+=__shfl_xor_sync(0xffffffffu,squares,offset);
      }
      float mean=Sm90F3Async::DivideNormalization(sum,float(Channels));
      float inverse=rsqrtf(Sm90F3Async::DivideNormalization(squares,float(Channels))-mean*mean+p.epsilon);
      for(int line=lane;line<kLines;line+=32) {
        auto octet=Sm90F3LoadOct(source+line*8);
        Sm90F3Oct out;
        #pragma unroll
        for(int i=0;i<8;++i) {
          int c=line*8+i;
          out.value()[i]=Element((float(octet.value()[i])-mean)*inverse*
              float(p.weight[c])+(p.table?float(p.table[c]):0.0f));
        }
        Sm90F3StoreOct(p.output+std::size_t(row)*Stride+line*8,out);
      }
    }
  }
};

// BERT's embedding sum: one warp per token row adds the word, token-type and
// position rows in FP32 and publishes the row's (sum, sum of squares) for the
// first encoder GEMM's deferred LayerNorm. One writer per row keeps those
// statistics exact and launch independent.
template<int Width>
struct Sm90F3EmbeddingSum {
  using Element=cutlass::bfloat16_t;
  static constexpr int kThreads=384;
  static constexpr int kSharedBytes=1024;
  static constexpr int kSideOffset=0;
  static_assert(Width%8==0);
  static constexpr int kWarps=8,kLines=Width/8;

  __device__ static __forceinline__ void Produce(Sm90F3SmallOperands const&,
      Sm90F3Ring&,char*,Sm90F3Phase&,int) {}

  __device__ static __forceinline__ void Consume(Sm90F3SmallOperands const& p,
      Sm90F3Ring&,char*,Sm90F3Phase&,Sm90F3Header*,int) {
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=int(blockIdx.x)*kWarps+warp;row<p.rows;row+=int(gridDim.x)*kWarps) {
      int token=int(p.index0[row]),type=p.index1?int(p.index1[row]):0;
      auto const* word=p.input+std::size_t(token)*Width;
      auto const* types=p.weight+std::size_t(type)*Width;
      auto const* position=p.table+std::size_t(row%p.height)*Width;
      float sum=0,squares=0;
      for(int line=lane;line<kLines;line+=32) {
        auto a=Sm90F3LoadOct(word+line*8);
        auto b=Sm90F3LoadOct(types+line*8);
        auto c=Sm90F3LoadOct(position+line*8);
        Sm90F3Oct out;
        #pragma unroll
        for(int i=0;i<8;++i) {
          float value=float(a.value()[i])+float(b.value()[i])+float(c.value()[i]);
          out.value()[i]=Element(value);
          sum+=value;squares+=value*value;
        }
        Sm90F3StoreOct(p.output+std::size_t(row)*Width+line*8,out);
      }
      if(p.stats) {
        #pragma unroll
        for(int offset=16;offset;offset>>=1) {
          sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
          squares+=__shfl_xor_sync(0xffffffffu,squares,offset);
        }
        if(!lane) {
          Sm90F3Async::StoreRelaxed(p.stats+2*row,sum);
          Sm90F3Async::StoreRelaxed(p.stats+2*row+1,squares);
        }
      }
    }
  }
};

}  // namespace tilemega::backend
