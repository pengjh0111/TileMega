// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cutlass/bfloat16.h>
#include <cuda_bf16.h>
#include <climits>
#include <cmath>

namespace tilemega::backend {

// Native decode epilogue (prompt §5 D4). InputNorm rows are scaled by the
// inverse RMS the consumers accumulated from their A fragments; residual
// tiles were prefetched into shared memory by the producer. Only the
// compile-time Batch rows are touched. Rounding matches ServingEpilogueValue.
template<ServingEpilogueOp Op,int Batch,int N,int K,int TN,int U,bool InputNorm>
struct Sm90DenseEpilogue {
  using Element=cutlass::bfloat16_t;
  static_assert(Op==ServingEpilogueOp::kStore || Op==ServingEpilogueOp::kResidual ||
                Op==ServingEpilogueOp::kSwiGLU || Op==ServingEpilogueOp::kArgmaxPartial);
  static_assert(Op!=ServingEpilogueOp::kSwiGLU || TN%(2*U)==0);
  static_assert(TN%2==0);
  static constexpr int kOutput=Op==ServingEpilogueOp::kSwiGLU?TN/2:TN;
  static constexpr int kStride=Op==ServingEpilogueOp::kSwiGLU?N/2:
      Op==ServingEpilogueOp::kArgmaxPartial?N/TN:N;

  __device__ static __forceinline__ float Inverse(float squares,float epsilon) {
    return rsqrtf(squares/float(K)+epsilon);
  }
  __device__ static __forceinline__ __nv_bfloat162 Pair(float x,float y) {
    __nv_bfloat162 v;v.x=__float2bfloat16_rn(x);v.y=__float2bfloat16_rn(y);return v;
  }
  __device__ static __forceinline__ void Store2(Element* output,int row,int col,float x,float y,
      Element const* residual) {
    if constexpr(Op==ServingEpilogueOp::kResidual) {
      auto r=*reinterpret_cast<__nv_bfloat162 const*>(residual+row*TN+col);
      x=float(Element(x))+__bfloat162float(r.x);y=float(Element(y))+__bfloat162float(r.y);
    }
    *reinterpret_cast<__nv_bfloat162*>(output+std::size_t(row)*kStride+col)=Pair(x,y);
  }

  // WK == 1: each warp owns complete accumulators; store/residual write them
  // directly. owned(i) are the (row, col) of accum(i); even/odd pairs share a row.
  template<class Accumulator,class Owned>
  __device__ static __forceinline__ void Registers(Accumulator const& accum,Owned const& owned,
      float const (&inv)[2],Element const* residual,Element* output,int task) {
    static_assert(Op==ServingEpilogueOp::kStore || Op==ServingEpilogueOp::kResidual);
    output+=task*TN;
    #pragma unroll
    for(int i=0;i<int(cute::size(accum));i+=2) {
      int row=cute::get<0>(owned(i)),col=cute::get<1>(owned(i));
      if(row<Batch) {
        float scale=InputNorm?inv[row>=8]:1.0f;
        Store2(output,row,col,accum(i)*scale,accum(i+1)*scale,residual);
      }
    }
  }

  // Shared path: WK partial tiles [WK][16][TN] and per-split squares
  // norms[WK][16]; one pass merges the reduction with the op.
  template<int WK>
  __device__ static __forceinline__ void Shared(float const* tile,float const* norms,
      Element const* residual,Element* output,int* index,int task,float epsilon) {
    int tid=threadIdx.x;
    auto value=[&](int row,int col) {
      float sum=tile[row*TN+col];
      #pragma unroll
      for(int s=1;s<WK;++s)sum+=tile[s*16*TN+row*TN+col];
      return sum;
    };
    auto inverse=[&](int row) {
      if constexpr(!InputNorm)return 1.0f;
      else {
        float squares=0;
        #pragma unroll
        for(int s=0;s<WK;++s)squares+=norms[s*16+row];
        return Inverse(squares,epsilon);
      }
    };
    if constexpr(Op==ServingEpilogueOp::kArgmaxPartial) {
      // Warp w owns rows w, w+4, ...; ties keep the smaller vocabulary index.
      int warp=tid>>5,lane=tid&31;
      auto* values=reinterpret_cast<float*>(output);
      for(int row=warp;row<Batch;row+=4) {
        float scale=inverse(row),best=-INFINITY;int arg=INT_MAX;
        for(int col=lane;col<TN;col+=32) {
          float v=float(Element(value(row,col)*scale));
          if(v>best) {best=v;arg=task*TN+col;}
        }
        #pragma unroll
        for(int d=16;d;d>>=1) {
          float other=__shfl_xor_sync(0xffffffff,best,d);int oi=__shfl_xor_sync(0xffffffff,arg,d);
          if(other>best || (other==best && oi<arg)) {best=other;arg=oi;}
        }
        if(lane==0) {values[row*kStride+task]=best;index[row*kStride+task]=arg;}
      }
    }else {
      output+=task*kOutput;
      for(int i=tid;i<Batch*kOutput/2;i+=128) {
        int row=i/(kOutput/2),col=2*(i%(kOutput/2));
        float scale=inverse(row);
        if constexpr(Op==ServingEpilogueOp::kSwiGLU) {
          float out[2];
          #pragma unroll
          for(int e=0;e<2;++e) {
            int gate=2*U*((col+e)/U)+(col+e)%U;
            out[e]=float(ServingEpilogueValue<Op>::Apply(value(row,gate)*scale,value(row,gate+U)*scale));
          }
          Store2(output,row,col,out[0],out[1],residual);
        }else Store2(output,row,col,value(row,col)*scale,value(row,col+1)*scale,residual);
      }
    }
  }
};
}  // namespace tilemega::backend
