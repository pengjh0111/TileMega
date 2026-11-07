// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <tilemega/Backend/Sm90F3Async.cuh>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <cutlass/bfloat16.h>
#include <cstddef>

// The F3 epilogue chain (DNN3 prompt §6.4 step 3). The accumulator never
// leaves the registers: the producer has already prefetched this tile's
// residual, per-column vectors and row statistics into the stage's side area,
// so each step is one shared-memory read and one FP32 operation per element.
//
// Every step's arithmetic is the legacy chain's, operation for operation and
// rounding for rounding (`Sm90ForwardEpilogue.cuh:60-68,92-104,251-262,284-309`),
// so only the accumulation order of the contraction itself differs (§13 U9).
namespace tilemega::backend {

// Reuse the legacy operation enum and step shape: codegen emits one list that
// both skeletons can read, and the activation set is shared.
using Sm90F3Op=Sm90ForwardOp;

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
  float* row_stats;              // side output: per-row (sum, sum of squares)
  float* row_partials;           // per (n_tile, row) share when NT > 1
  unsigned long long* row_counters;
  float* channel_partial;        // GAP side output: [image][part][channel]
  float* partial;                // split-K: [tile][slice][TM][TN] fp32
  unsigned long long* counters;  // split-K: one per output tile
  cutlass::bfloat16_t* output;
  int output_stride,residual_stride,norm_width;
  float epsilon;
};

// Where the producer put this task's side data inside the stage's side area
// (§6.4 step 1). The offsets are compile-time, so a step's read is a single
// shared load from a known base.
template<int TM,int TN,bool Residual,int Vectors,bool RowStats>
struct Sm90F3SideLayout {
  static constexpr int kResidualBytes=Residual?Sm90F3ResidualBytes(TM,TN):0;
  static constexpr int kVectorBytes=Sm90F3VectorBytes(TN,Vectors);
  static constexpr int kRowStatBytes=RowStats?Sm90F3RowStatBytes(TM):0;
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
  __device__ static __forceinline__ float const* RowStat(char const* side) {
    return reinterpret_cast<float const*>(side+kRowStatBase);
  }
};

// (mean, 1/sigma) of one row from the producer's raw sums, in the legacy
// order: mean = sum/width, inv = rsqrt(sumsq/width - mean^2 + eps).
__device__ __forceinline__ void Sm90F3RowNorm(float const* stats,int row,int width,
    float epsilon,float& mean,float& inverse) {
  float sum=stats[2*row],squares=stats[2*row+1];
  mean=sum/float(width);
  inverse=rsqrtf(squares/float(width)-mean*mean+epsilon);
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
  // A gate pair halves the width and only NAFNet uses it; F3 rejects it, and
  // a residual only ever takes the dense mapping (§6.4 step 3, kGatePair).
  static_assert(((Step::kKind!=Sm90ForwardOp::kGatePair) && ...),
      "F3 does not implement a gate pair; the four DNN3 models never use one");

  template<class S>
  __device__ static __forceinline__ void ApplyOne(float& value,int row,int col,
      char const* side,float mean,float inverse,Sm90F3EpilogueOperands const& p) {
    if constexpr(S::kKind==Sm90ForwardOp::kBias)
      value=__fadd_rn(value,Side::Vector(side,S::kVector)[col]);
    else if constexpr(S::kKind==Sm90ForwardOp::kScale)
      value=__fmul_rn(value,Side::Vector(side,S::kVector)[col]);
    else if constexpr(S::kKind==Sm90ForwardOp::kActivation)
      value=Sm90ForwardActivate(S::kActivation,value);
    else if constexpr(S::kKind==Sm90ForwardOp::kResidual) {
      // One 32-bit shared read takes the (col, col+1) pair; this element keeps
      // its own half. The scale, when the plan named one, rounds before the add.
      float add=float(ResidualAt(side,row,col));
      if constexpr(S::kVector<4)
        if(p.vectors[S::kVector])value=__fmul_rn(value,Side::Vector(side,S::kVector)[col]);
      value=__fadd_rn(value,add);
    }else if constexpr(S::kKind==Sm90ForwardOp::kDeferredLayerNorm) {
      // (acc - mu u_n) * inv + v_n, with (mu, inv) from the producer's sums.
      value=(value-mean*Side::Vector(side,S::kVector)[col])*inverse+
          Side::Vector(side,S::kVector2)[col];
    }else if constexpr(S::kKind==Sm90ForwardOp::kResidualLN) {
      float source=float(ResidualAt(side,row,col));
      value+=(source-mean)*inverse*Side::Vector(side,S::kVector)[col]+
          Side::Vector(side,S::kVector2)[col];
    }
  }

  __device__ static __forceinline__ cutlass::bfloat16_t ResidualAt(char const* side,
      int row,int col) {
    auto const* address=reinterpret_cast<cutlass::bfloat16_t const*>(
        side+Side::ResidualOffset(row,col));
    return *address;
  }

  __device__ static __forceinline__ float Apply(float value,int row,int col,
      char const* side,float mean,float inverse,Sm90F3EpilogueOperands const& p) {
    (ApplyOne<Step>(value,row,col,side,mean,inverse,p),...);
    return value;
  }
};

template<class Chain>
struct Sm90F3WriteRowStats:Chain {
  static constexpr bool kWritesStats=true;
};

}  // namespace tilemega::backend
