// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <cutlass/bfloat16.h>
#include <cstddef>

#ifndef TILEMEGA_F3_GELU_OUTLINE
#define TILEMEGA_F3_GELU_OUTLINE 0
#endif
#if TILEMEGA_F3_GELU_OUTLINE != 0 && TILEMEGA_F3_GELU_OUTLINE != 1
#error "F3 GELU quartet must be0 or1"
#endif

// The F3 epilogue chain (DNN3 prompt §6.4 step 3). The accumulator never
// leaves the registers: the producer has already prefetched this tile's
// residual, per-column vectors and row statistics into the stage's side area,
// so each step is one shared-memory read and one FP32 operation per element.
//
// Every step's arithmetic is the legacy chain's, operation for operation and
// rounding except the compensated FP32 normalization division,
// so only the accumulation order of the contraction itself differs (§13 U9).
namespace tilemega::backend {

// Reuse the legacy operation enum and step shape: codegen emits one list that
// both skeletons can read, and the activation set is shared.
using Sm90F3Op=Sm90ForwardOp;

// DT-GELU: A&S7.1.26, measured max normalized error1.972e-7 on the full grid
// and finite BF16 domain;132-CTA resident epilogue ratio0.7487 (gelu_v1).
__device__ __forceinline__ float Sm90F3GeluErf(float x) {
  float z=fabsf(x*.70710678118654752f),denominator=__fmaf_rn(.3275911f,z,1.f),t;
  asm("rcp.approx.ftz.f32 %0, %1;":"=f"(t):"f"(denominator));
  t=__fmaf_rn(__fmaf_rn(-denominator,t,1.f),t,t);
  float poly=__fmaf_rn(t,1.061405429f,-1.453152027f);
  poly=__fmaf_rn(t,poly,1.421413741f);
  poly=__fmaf_rn(t,poly,-.284496736f);
  poly=t*__fmaf_rn(t,poly,.254829592f);
  float erf=1.f-poly*Sm90F3Async::Exp2(-z*z*1.4426950408889634f);
  return .5f*x*(1.f+copysignf(erf,x));
}

#if TILEMEGA_F3_GELU_OUTLINE
// Four independent columns/rows expose the same scalar recurrence's ILP;
// each value retains the original operation order and rounding points.
__device__ __noinline__ float4 Sm90F3GeluErfFour(float4 input) {
  float x[4]={input.x,input.y,input.z,input.w};
  float z[4],denominator[4],t[4],poly[4];
  #pragma unroll
  for(int j=0;j<4;++j) {
    z[j]=fabsf(x[j]*.70710678118654752f);
    denominator[j]=__fmaf_rn(.3275911f,z[j],1.f);
    asm("rcp.approx.ftz.f32 %0, %1;":"=f"(t[j]):"f"(denominator[j]));
  }
  #pragma unroll
  for(int j=0;j<4;++j)
    t[j]=__fmaf_rn(__fmaf_rn(-denominator[j],t[j],1.f),t[j],t[j]);
  #pragma unroll
  for(int j=0;j<4;++j)poly[j]=__fmaf_rn(t[j],1.061405429f,-1.453152027f);
  #pragma unroll
  for(int j=0;j<4;++j)poly[j]=__fmaf_rn(t[j],poly[j],1.421413741f);
  #pragma unroll
  for(int j=0;j<4;++j)poly[j]=__fmaf_rn(t[j],poly[j],-.284496736f);
  #pragma unroll
  for(int j=0;j<4;++j)poly[j]=t[j]*__fmaf_rn(t[j],poly[j],.254829592f);
  #pragma unroll
  for(int j=0;j<4;++j) {
    float erf=1.f-poly[j]*Sm90F3Async::Exp2(-z[j]*z[j]*1.4426950408889634f);
    x[j]=.5f*x[j]*(1.f+copysignf(erf,x[j]));
  }
  return make_float4(x[0],x[1],x[2],x[3]);
}
#endif

template<Sm90ForwardOp Kind,unsigned Vector=0,
         codegen::DmActivation Activation=codegen::DmActivation::kRelu,
         unsigned Vector2=0>
struct Sm90F3Step {
  static constexpr Sm90ForwardOp kKind=Kind;
  static constexpr unsigned kVector=Vector,kVector2=Vector2;
  static constexpr codegen::DmActivation kActivation=Activation;
};

// The pointers a stage's epilogue binds. Per-column vectors stay FP32, as the
// folds that build them are exact only in that precision.
struct Sm90F3EpilogueOperands {
  float const* vectors[4];
  cutlass::bfloat16_t const* residual;
  float const* stats;            // the producer's per-row (sum, sum of squares)
  float* row_stats;              // small-body side output, one pair per row
  float* row_partials;           // GEMM side output: [N/64][M][2]
  unsigned long long* row_counters;
  float* channel_partial;        // GAP side output: [image][part][channel]
  float* partial;                // split-K: [tile][slice][thread][fragment] fp32
  unsigned long long* counters;  // split-K: one per output tile
  cutlass::bfloat16_t* output;
  int output_stride,residual_stride,norm_width;
  float epsilon;
};

// Where the producer put this task's side data inside the stage's side area
// (§6.4 step 1). The offsets are compile-time, so a step's read is a single
// shared load from a known base.
template<int TM,int TN,bool Residual,int Vectors,bool RowStats,int StatParts=1>
struct Sm90F3SideLayout {
  static_assert(StatParts>=1);
  static constexpr int kStatParts=StatParts;
  static constexpr int kResidualBytes=Residual?Sm90F3ResidualBytes(TM,TN):0;
  static constexpr int kVectorBytes=Sm90F3VectorBytes(TN,Vectors);
  static constexpr int kRowStatBytes=RowStats?StatParts*Sm90F3RowStatBytes(TM):0;
  static constexpr int kResidualBase=0;
  static constexpr int kVectorBase=kResidualBytes;
  static constexpr int kRowStatBase=kVectorBase+kVectorBytes;
  static constexpr int kBytes=kRowStatBase+kRowStatBytes;

  // The residual tile is stored exactly as an A panel is: SW128 K-major rows
  // of 128 B, one box per 64 columns (§6.4 step 3, last bullet).
  __device__ static __forceinline__ unsigned ResidualOffset(int row,int col) {
    return unsigned(kResidualBase+(col/64)*TM*128+row*128+
        ((((col%64)/8)^(row%8))*16)+(col%8)*2);
  }
  __device__ static __forceinline__ float const* Vector(char const* side,unsigned slot) {
    return reinterpret_cast<float const*>(side+kVectorBase)+std::size_t(slot)*TN;
  }
  __device__ static __forceinline__ float const* RowStat(char const* side,int part=0) {
    return reinterpret_cast<float const*>(side+kRowStatBase)+part*TM*2;
  }
};

// (mean, 1/sigma) of one row from the producer's raw sums, in the legacy
// order: mean = sum/width, inv = rsqrt(sumsq/width - mean^2 + eps).
template<int TM,int Parts>
__device__ __forceinline__ void Sm90F3RowNorm(float const* stats,int row,int width,
    float epsilon,float& mean,float& inverse) {
  float sum=0,squares=0;
  #pragma unroll
  for(int part=0;part<Parts;++part) {
    sum+=stats[(part*TM+row)*2];squares+=stats[(part*TM+row)*2+1];
  }
  mean=Sm90F3Async::DivideNormalization(sum,float(width));
  inverse=rsqrtf(Sm90F3Async::DivideNormalization(squares,float(width))-mean*mean+epsilon);
}

// One element's chain. `row` and `col` are the element's coordinates inside
// the CTA tile; `side` is the stage's side area and `norm` the row's
// (mean, 1/sigma), which the caller computed once per row it owns.
template<class Side,class... Step>
struct Sm90F3Chain {
  using Layout=Side;
  static constexpr bool kWritesStats=false;
  static constexpr int kSteps=int(sizeof...(Step));
  static constexpr bool kNeedsResidual=
      ((Step::kKind==Sm90ForwardOp::kResidual ||
        Step::kKind==Sm90ForwardOp::kResidualLN) || ...);
  static constexpr bool kNeedsStats=
      ((Step::kKind==Sm90ForwardOp::kDeferredLayerNorm ||
        Step::kKind==Sm90ForwardOp::kResidualLN) || ...);
#if TILEMEGA_F3_GELU_OUTLINE
  static constexpr bool kQuartetGelu=
      ((Step::kKind==Sm90ForwardOp::kActivation &&
        Step::kActivation==codegen::DmActivation::kGeluErf) || ...);

  template<class S>
  __device__ static __forceinline__ void ApplyFourOne(float (&values)[4],int row,int col,
      char const* side,float const (&means)[2],float const (&inverses)[2],
      Sm90F3EpilogueOperands const& p,float const (&columns)[2][4]) {
    if constexpr(S::kKind==Sm90ForwardOp::kActivation &&
        S::kActivation==codegen::DmActivation::kGeluErf) {
      auto result=Sm90F3GeluErfFour(make_float4(values[0],values[1],values[2],values[3]));
      values[0]=result.x;values[1]=result.y;values[2]=result.z;values[3]=result.w;
    }
    else {
      #pragma unroll
      for(int half=0;half<2;++half) {
        ApplyOne<S>(values[half*2],row+half*8,col,side,means[half],inverses[half],p,columns[0]);
        ApplyOne<S>(values[half*2+1],row+half*8,col+1,side,means[half],inverses[half],p,columns[1]);
      }
    }
  }

  __device__ static __forceinline__ void ApplyFour(float (&values)[4],int row,int col,
      char const* side,float const (&means)[2],float const (&inverses)[2],
      Sm90F3EpilogueOperands const& p,float const (&columns)[2][4]) {
    (ApplyFourOne<Step>(values,row,col,side,means,inverses,p,columns),...);
  }
#endif
  // A gate pair halves the width and only NAFNet uses it; F3 rejects it, and
  // a residual only ever takes the dense mapping (§6.4 step 3, kGatePair).
  static_assert(((Step::kKind!=Sm90ForwardOp::kGatePair) && ...),
      "F3 does not implement a gate pair; the four DNN3 models never use one");

  template<unsigned Slot>
  __device__ static __forceinline__ float VectorAt(char const* side,int col,
      float const* columns) {
    return columns?columns[Slot]:Side::Vector(side,Slot)[col];
  }

  template<class S>
  __device__ static __forceinline__ void ApplyOne(float& value,int row,int col,
      char const* side,float mean,float inverse,Sm90F3EpilogueOperands const& p,
      float const* columns) {
    if constexpr(S::kKind==Sm90ForwardOp::kBias)
      value=__fadd_rn(value,VectorAt<S::kVector>(side,col,columns));
    else if constexpr(S::kKind==Sm90ForwardOp::kScale)
      value=__fmul_rn(value,VectorAt<S::kVector>(side,col,columns));
    else if constexpr(S::kKind==Sm90ForwardOp::kActivation) {
      if constexpr(S::kActivation==codegen::DmActivation::kGeluErf)
        value=Sm90F3GeluErf(value);
      else value=Sm90ForwardActivate(S::kActivation,value);
    }
    else if constexpr(S::kKind==Sm90ForwardOp::kResidual) {
      // One 32-bit shared read takes the (col, col+1) pair; this element keeps
      // its own half. The scale, when the plan named one, rounds before the add.
      float add=float(ResidualAt(side,row,col));
      if constexpr(S::kVector<4)
        if(p.vectors[S::kVector])value=__fmul_rn(value,VectorAt<S::kVector>(side,col,columns));
      value=__fadd_rn(value,add);
    }else if constexpr(S::kKind==Sm90ForwardOp::kDeferredLayerNorm) {
      // (acc - mu u_n) * inv + v_n, with (mu, inv) from the producer's sums.
      value=(value-mean*VectorAt<S::kVector>(side,col,columns))*inverse+
          VectorAt<S::kVector2>(side,col,columns);
    }else if constexpr(S::kKind==Sm90ForwardOp::kResidualLN) {
      float source=float(ResidualAt(side,row,col));
      value+=(source-mean)*inverse*VectorAt<S::kVector>(side,col,columns)+
          VectorAt<S::kVector2>(side,col,columns);
    }
  }

  __device__ static __forceinline__ cutlass::bfloat16_t ResidualAt(char const* side,
      int row,int col) {
    auto const* address=reinterpret_cast<cutlass::bfloat16_t const*>(
        side+Side::ResidualOffset(row,col));
    return *address;
  }

  __device__ static __forceinline__ float Apply(float value,int row,int col,
      char const* side,float mean,float inverse,Sm90F3EpilogueOperands const& p,
      float const* columns=nullptr) {
    (ApplyOne<Step>(value,row,col,side,mean,inverse,p,columns),...);
    return value;
  }
};

template<class Chain>
struct Sm90F3WriteRowStats:Chain {
  static constexpr bool kWritesStats=true;
};

}  // namespace tilemega::backend
