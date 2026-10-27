// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <cutlass/bfloat16.h>
#include <cmath>

// Native DNN forward epilogue chain (prompt §10 N-1 item 5). The plan's
// DmEpilogueChain is emitted as this compile-time op list: every op reads the
// accumulator tile in fp32 and writes it back, so a chain costs no extra
// round trip. Parameters are bound by index into the stage's operand table.
namespace tilemega::backend {

struct Sm90ForwardEpilogueOperands {
  // Per-channel vectors (bias, BN fold, residual scales, gate scales) and the
  // residual source; `stats` holds per-row (sum, sumsq) for deferred norms.
  cutlass::bfloat16_t const* vectors[4];
  cutlass::bfloat16_t const* residual;
  float const* stats;
  float* row_stats;        // side output: per-row (sum, sumsq)
  float* channel_partial;  // side output: per-channel partial sums
  float epsilon;
  int residual_stride;
};

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

// The chain runs over one M x N fp32 tile in shared memory. Rows are the
// task's M output rows; columns are its N channels starting at `channel`.
template<int M,int N,class... Step>
struct Sm90ForwardEpilogueChain {
  using Element=cutlass::bfloat16_t;
  static constexpr int kSteps=int(sizeof...(Step));
  // Gate pairs halve the written width; everything else keeps N.
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
          float add=float(p.residual[std::size_t(row)*p.residual_stride+channel+col]);
          if(p.vectors[S::kVector])add*=float(p.vectors[S::kVector][channel+col]);
          value+=add;
        }else if constexpr(S::kKind==Sm90ForwardOp::kDeferredLayerNorm) {
          // (acc - mu cs_n)/sigma + b', with (mu, sigma) from the row stats
          // the producing GEMM's consumer wrote (prompt §10 N-1 item 6).
          float mean=p.stats[2*row],inverse=p.stats[2*row+1];
          value=(value-mean*float(p.vectors[S::kVector][channel+col]))*inverse+
              float(p.vectors[S::kVector2][channel+col]);
        }else if constexpr(S::kKind==Sm90ForwardOp::kResidualLN) {
          float mean=p.stats[2*row],inverse=p.stats[2*row+1];
          float source=float(p.residual[std::size_t(row)*p.residual_stride+channel+col]);
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
        for(int col=0;col<kWidth;++col) {float v=tile[row*N+col];sum+=v;squares+=v*v;}
        atomicAdd(p.row_stats+2*row,sum);atomicAdd(p.row_stats+2*row+1,squares);
      }
    if(p.channel_partial)
      for(int col=int(threadIdx.x);col<kWidth;col+=128) {
        float sum=0;
        for(int row=0;row<rows;++row)sum+=tile[row*N+col];
        atomicAdd(p.channel_partial+channel+col,sum);
      }
    for(int i=int(threadIdx.x);i<rows*width;i+=128) {
      int row=i/width,col=i%width;
      output[std::size_t(row)*stride+channel+col]=Element(tile[row*N+col]);
    }
    codegen::executor::ComputeSync();
  }
};

}  // namespace tilemega::backend
