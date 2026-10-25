// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cutlass/bfloat16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

namespace tilemega::backend {

template<int Width,int Vocab,int Capacity>
struct Sm90DenseEmbedding {
  using Element=cutlass::bfloat16_t;
  static_assert(Width%32==0 && Vocab>0 && Capacity>=1088);
  __device__ static void Run(int const* tokens,Element const* table,Element* output,
                            int row,int past) {
    int token=tokens[row*Capacity+past];
    auto* source=table+std::size_t(token)*Width;
    for(int col=threadIdx.x*8;col<Width;col+=128*8)
      *reinterpret_cast<uint4*>(output+row*Width+col)=*reinterpret_cast<uint4 const*>(source+col);
  }
};

template<int Parts,int Capacity>
struct Sm90DenseArgmax {
  struct Storage {float value[4];int index[4];};
  static_assert(Parts>0 && Capacity>=1088);
  __device__ static void Run(float const* values,int const* indices,int* tokens,
                            Storage& s,int row,int past) {
    float best=-INFINITY;int index=INT_MAX;
    for(int part=threadIdx.x;part<Parts;part+=128) {
      float value=values[row*Parts+part];int id=indices[row*Parts+part];
      if(value>best || (value==best && id<index)) {best=value;index=id;}
    }
    for(int d=16;d;d>>=1) {
      float value=__shfl_down_sync(0xffffffff,best,d);
      int id=__shfl_down_sync(0xffffffff,index,d);
      if(value>best || (value==best && id<index)) {best=value;index=id;}
    }
    if((threadIdx.x&31)==0) {s.value[threadIdx.x>>5]=best;s.index[threadIdx.x>>5]=index;}
    codegen::executor::ComputeSync();
    if(threadIdx.x==0) {
      best=s.value[0];index=s.index[0];
      #pragma unroll
      for(int warp=1;warp<4;++warp) {
        float value=s.value[warp];int id=s.index[warp];
        if(value>best || (value==best && id<index)) {best=value;index=id;}
      }
      tokens[row*Capacity+past+1]=index;
    }
  }
};

}  // namespace tilemega::backend
