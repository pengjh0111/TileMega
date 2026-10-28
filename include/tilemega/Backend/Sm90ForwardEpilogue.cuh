// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cutlass/bfloat16.h>
#include <cstddef>
#include <cmath>

// Native DNN forward epilogue chain (prompt §10 N-1 item 5). The plan's
// DmEpilogueChain is emitted as this compile-time op list: every op reads the
// accumulator tile in fp32 and writes it back, so a chain costs no extra
// round trip. Parameters are bound by index into the stage's operand table.
namespace tilemega::backend {

struct Sm90ForwardEpilogueOperands {
  // Per-channel vectors (bias, BN fold, residual scales, gate scales) and the
  // residual source; `stats` holds per-row (sum, sumsq) for deferred norms.
  // The plan keeps every per-channel vector in FP32, as the folds that build
  // them are exact only in that precision.
  float const* vectors[4];
  cutlass::bfloat16_t const* residual;
  float const* stats;
  float* row_stats;        // side output: per-row (sum, sumsq)
  float* channel_partial;  // side output: per-channel partial sums
  // The row statistics of a tiled stage are reduced in a fixed tile order by
  // the row's last arriver, so repeated launches agree bitwise.
  float* row_partials;
  unsigned long long* row_counters;
  int row_tiles,row_tile;
  float epsilon;
  int residual_stride;
  // The normalization width of a deferred or reconstructed LayerNorm: its
  // statistics arrive as the producer's raw (sum, sum of squares) per row.
  int norm_width;
  // Spatial write mapping: a body whose rows are the pixels of a TW x TH tile
  // sets `tile_width` and the image's row pitch, and the store walks the tile
  // instead of a dense run of rows. Zero keeps the dense mapping.
  int tile_width,image_width;
  // This tile's first row in the producing body's own row space, which the
  // NCHW and shuffled writebacks need to locate the pixel.
  int row_base;
};

// A row's offset inside its buffer: a body whose rows are the pixels of a
// spatial tile addresses them through that tile, while a GEMM's rows are a
// dense run.
__device__ __forceinline__ std::size_t Sm90ForwardRowOffset(
    Sm90ForwardEpilogueOperands const& p,int row,int stride) {
  return p.tile_width>0?
      (std::size_t(row/p.tile_width)*p.image_width+row%p.tile_width)*std::size_t(stride):
      std::size_t(row)*std::size_t(stride);
}

// (mean, 1/sigma) of one row from the producer's raw sums. The producer ran
// on other SMs, whose L1 caches are not coherent, so the sums are read at GPU
// scope exactly as the split-K partials are.
__device__ __forceinline__ void Sm90ForwardRowNorm(float const* stats,int row,int width,
    float epsilon,float& mean,float& inverse) {
  float sum,squares;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(sum):"l"(stats+2*row):"memory");
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(squares)
      :"l"(stats+2*row+1):"memory");
  mean=sum/float(width);
  inverse=rsqrtf(squares/float(width)-mean*mean+epsilon);
}

enum : int {kDenseWrite=0,kNchwWrite=1,kShuffleWrite=2};

enum class Sm90ForwardOp : unsigned {
  kBias, kScale, kActivation, kResidual, kGatePair, kDeferredLayerNorm, kResidualLN
};

// One chain entry. `vector` selects the operand vector; `activation` and
// `gate` name the arithmetic; `unit` is the gate pair's interleave.
// `Vector2` is the second per-channel vector a normalization step needs: the
// deferred LayerNorm's b' and the residual LayerNorm's beta.
template<Sm90ForwardOp Kind,unsigned Vector=0,
         codegen::DmActivation Activation=codegen::DmActivation::kRelu,
         codegen::DmGatePair Gate=codegen::DmGatePair::kSimpleGate,unsigned Unit=16,
         unsigned Vector2=0>
struct Sm90ForwardEpilogueStep {
  static constexpr Sm90ForwardOp kKind=Kind;
  static constexpr unsigned kVector=Vector,kVector2=Vector2,kUnit=Unit;
  static constexpr codegen::DmActivation kActivation=Activation;
  static constexpr codegen::DmGatePair kGate=Gate;
};

__device__ __forceinline__ float Sm90ForwardActivate(codegen::DmActivation kind,float x) {
  switch(kind) {
    case codegen::DmActivation::kRelu:return x>0?x:0.0f;
    case codegen::DmActivation::kRelu6:return fminf(fmaxf(x,0.0f),6.0f);
    case codegen::DmActivation::kGeluErf:return 0.5f*x*(1.0f+erff(x*0.70710678118654752f));
    case codegen::DmActivation::kGeluTanh: {
      float inner=0.7978845608028654f*(x+0.044715f*x*x*x);return 0.5f*x*(1.0f+tanhf(inner));
    }
    case codegen::DmActivation::kTanh:return tanhf(x);
    case codegen::DmActivation::kSilu:return x/(1.0f+__expf(-x));
  }
  return x;
}

// A chain for a result still in registers: the fused body's depthwise, whose
// value never reaches a shared tile. Only the channel-wise operations a
// depthwise epilogue carries are implemented; the build rejects the rest.
template<class... Step>
struct Sm90ForwardElementChain {
  template<class S>
  __device__ static __forceinline__ void ApplyOne(float& value,int channel,
      float const* const (&vectors)[4]) {
    if constexpr(S::kKind==Sm90ForwardOp::kScale)
      value*=vectors[S::kVector][channel];
    else if constexpr(S::kKind==Sm90ForwardOp::kBias)
      value+=vectors[S::kVector][channel];
    else if constexpr(S::kKind==Sm90ForwardOp::kActivation)
      value=Sm90ForwardActivate(S::kActivation,value);
    else
      static_assert(S::kKind==Sm90ForwardOp::kScale,
          "a register-resident chain implements scale, bias and activation");
  }
  __device__ static __forceinline__ float Apply(float value,int channel,
      float const* const (&vectors)[4]) {
    (ApplyOne<Step>(value,channel,vectors),...);
    return value;
  }
};

// The chain runs over one M x N fp32 tile in shared memory. Rows are the
// task's M output rows; columns are its N channels starting at `channel`.
// `Write` names the writeback mapping and `Shuffle`, `Width`, `Height` and
// `Channels` its image geometry; all four are compile-time properties of the
// stage, so they travel with the chain rather than in the operand bank.
template<int M,int N,int Write=kDenseWrite,int Shuffle=1,int Width=0,int Height=0,
         int Channels=0,class... Step>
struct Sm90ForwardEpilogueChain {
  using Element=cutlass::bfloat16_t;
  static constexpr int kSteps=int(sizeof...(Step));
  // Gate pairs halve the written width; everything else keeps N.
  static constexpr int kWrite=Write;
  static constexpr bool kGated=((Step::kKind==Sm90ForwardOp::kGatePair) || ...);
  static constexpr int kWidth=kGated?N/2:N;
  static_assert(!kGated || N%2==0);
  // A deferred or reconstructed LayerNorm needs the producing body's per-row
  // (mu, 1/sigma) over the A tile; the body computes them only when asked.
  static constexpr bool kNeedsStats=
      ((Step::kKind==Sm90ForwardOp::kDeferredLayerNorm ||
        Step::kKind==Sm90ForwardOp::kResidualLN) || ...);

  template<class S>
  __device__ static __forceinline__ void Apply(float* tile,int rows,int channel,
      Sm90ForwardEpilogueOperands const& p) {
    int tid=int(threadIdx.x);
    if constexpr(S::kKind==Sm90ForwardOp::kGatePair) {
      // SimpleGate/SwiGLU over the interleaved halves; writes N/2 columns.
      for(int i=tid;i<rows*(N/2);i+=128) {
        int row=i/(N/2),col=i%(N/2);
        int a=S::kGate==codegen::DmGatePair::kSwiGLU?2*int(S::kUnit)*(col/int(S::kUnit))+col%int(S::kUnit):col;
        int b=S::kGate==codegen::DmGatePair::kSwiGLU?a+int(S::kUnit):col+N/2;
        float g=tile[row*N+a],u=tile[row*N+b];
        tile[row*N+col]=S::kGate==codegen::DmGatePair::kSwiGLU?
            Sm90ForwardActivate(codegen::DmActivation::kSilu,g)*u:g*u;
      }
      codegen::executor::ComputeSync();
      return;
    }else {
      for(int i=tid;i<rows*kWidth;i+=128) {
        int row=i/kWidth,col=i%kWidth;float value=tile[row*N+col];
        if constexpr(S::kKind==Sm90ForwardOp::kBias)
          value+=float(p.vectors[S::kVector][channel+col]);
        else if constexpr(S::kKind==Sm90ForwardOp::kScale)
          value*=float(p.vectors[S::kVector][channel+col]);
        else if constexpr(S::kKind==Sm90ForwardOp::kActivation)
          value=Sm90ForwardActivate(S::kActivation,value);
        else if constexpr(S::kKind==Sm90ForwardOp::kResidual) {
          float add=float(p.residual[Sm90ForwardRowOffset(p,row,p.residual_stride)+channel+col]);
          if(p.vectors[S::kVector])add*=float(p.vectors[S::kVector][channel+col]);
          value+=add;
        }else if constexpr(S::kKind==Sm90ForwardOp::kDeferredLayerNorm) {
          // (acc - mu cs_n)/sigma + b', with (mu, sigma) from the row stats
          // the producing stage wrote (prompt §10 N-1 item 6).
          float mean,inverse;
          Sm90ForwardRowNorm(p.stats,row,p.norm_width,p.epsilon,mean,inverse);
          value=(value-mean*float(p.vectors[S::kVector][channel+col]))*inverse+
              float(p.vectors[S::kVector2][channel+col]);
        }else if constexpr(S::kKind==Sm90ForwardOp::kResidualLN) {
          float mean,inverse;
          Sm90ForwardRowNorm(p.stats,row,p.norm_width,p.epsilon,mean,inverse);
          float source=float(
              p.residual[Sm90ForwardRowOffset(p,row,p.residual_stride)+channel+col]);
          value+=(source-mean)*inverse*float(p.vectors[S::kVector][channel+col])+
              float(p.vectors[S::kVector2][channel+col]);
        }
        tile[row*N+col]=value;
      }
      codegen::executor::ComputeSync();
    }
  }

  // Side outputs read the finished tile; row stats feed a later deferred norm.
  __device__ static __forceinline__ void Finish(float* tile,int rows,int channel,int width,
      Sm90ForwardEpilogueOperands const& p,Element* output,int stride) {
    (Apply<Step>(tile,rows,channel,p),...);
    if(p.row_stats)
      for(int row=int(threadIdx.x);row<rows;row+=128) {
        float sum=0,squares=0;
        for(int col=0;col<width;++col) {float v=tile[row*N+col];sum+=v;squares+=v*v;}
        if(p.row_tiles<=1) {p.row_stats[2*row]=sum;p.row_stats[2*row+1]=squares;continue;}
        // This tile's share of the row, then the row's last arriver sums the
        // shares in tile order.
        auto* slot=p.row_partials+(std::size_t(row)*p.row_tiles+p.row_tile)*2;
        asm volatile("st.relaxed.gpu.global.f32 [%0], %1;"::"l"(slot),"f"(sum):"memory");
        asm volatile("st.relaxed.gpu.global.f32 [%0], %1;"::"l"(slot+1),"f"(squares):"memory");
        unsigned long long old;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(old)
            :"l"(p.row_counters+row):"memory");
        if(int(old+1)!=p.row_tiles)continue;
        p.row_counters[row]=0;
        float total=0,total_squares=0;
        for(int tile_index=0;tile_index<p.row_tiles;++tile_index) {
          auto const* other=p.row_partials+(std::size_t(row)*p.row_tiles+tile_index)*2;
          float a,b;
          asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(a):"l"(other):"memory");
          asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(b):"l"(other+1):"memory");
          total+=a;total_squares+=b;
        }
        p.row_stats[2*row]=total;p.row_stats[2*row+1]=total_squares;
      }
    if(p.channel_partial)
      for(int col=int(threadIdx.x);col<kWidth;col+=128) {
        float sum=0;
        for(int row=0;row<rows;++row)sum+=tile[row*N+col];
        atomicAdd(p.channel_partial+channel+col,sum);
      }
    for(int i=int(threadIdx.x);i<rows*width;i+=128) {
      int row=i/width,col=i%width;
      float value=tile[row*N+col];
      if constexpr(Write==kDenseWrite) {
        output[Sm90ForwardRowOffset(p,row,stride)+channel+col]=Element(value);
      }else {
        // Both remaining mappings address the whole image, so they locate the
        // producing body's own row inside it.
        constexpr int kPixels=Width*Height;
        int source=p.row_base+row,image=source/kPixels,pixel=source%kPixels;
        int y=pixel/Width,x=pixel%Width;
        if constexpr(Write==kNchwWrite) {
          int c=channel+col;
          output[((std::size_t(image)*Channels+c)*Height+y)*Width+x]=Element(value);
        }else {
          // Channel g*C + c of pixel (y, x) is output pixel
          // (y*f + g/f, x*f + g%f), channel c.
          int group=(channel+col)/Channels;
          std::size_t destination=(std::size_t(image)*Height*Shuffle+y*Shuffle+group/Shuffle)*
              (std::size_t(Width)*Shuffle)+std::size_t(x)*Shuffle+group%Shuffle;
          output[destination*Channels+(channel+col)%Channels]=Element(value);
        }
      }
    }
    codegen::executor::ComputeSync();
  }
};

}  // namespace tilemega::backend
