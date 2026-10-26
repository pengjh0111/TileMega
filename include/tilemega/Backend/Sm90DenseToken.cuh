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

}  // namespace tilemega::backend
