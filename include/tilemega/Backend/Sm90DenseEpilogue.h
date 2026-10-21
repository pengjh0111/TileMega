// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingEpilogue.h>

namespace tilemega::backend {
template<ServingEpilogueOp Op,int TN,int U>
struct Sm90DenseEpilogue {
  __device__ static void Run(float* tile,int task,int batch,int n,int stride,
      cutlass::bfloat16_t* output,cutlass::bfloat16_t const* residual,int* index,
      float const* norm,float* sums,int norm_k,float epsilon) {
    if constexpr(Op==ServingEpilogueOp::kSwiGLU && TN%16) {
      static_assert(TN%(2*U)==0);
      if(norm) {
        float* inverse=tile+16*TN;
        int row=threadIdx.x/8,part=threadIdx.x%8;
        float sum=0;
        if(row<batch)for(int j=part;j<norm_k/32;j+=8)
          sum+=DeferredNormBlock(norm,row,norm_k,j);
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        sum+=__shfl_xor_sync(0xffffffff,sum,4);
        if(part==0)inverse[row]=row<batch?rsqrtf(sum/float(norm_k)+epsilon):0;
        codegen::executor::ComputeSync();
        for(int i=threadIdx.x;i<16*TN;i+=128)tile[i]*=inverse[i/TN];
        codegen::executor::ComputeSync();
      }
      // TN=8 mod 16 has a half-width that is not an integral eight-element
      // vector. Every logical output must still be written, including its tail.
      for(int i=threadIdx.x;i<16*(TN/2);i+=128) {
        int row=i/(TN/2),col=i%(TN/2),gate=2*U*(col/U)+col%U;
        if(row<batch)output[row*stride+task*(TN/2)+col]=
            ServingEpilogueValue<Op>::Apply(tile[row*TN+gate],tile[row*TN+gate+U]);
      }
    }else ServingEpilogue<Op,16,TN,U>::template RunFromTile<false>(tile,0,task,batch,n,
        stride,n,output,residual,nullptr,reinterpret_cast<float*>(output),index,
        norm,sums,norm_k,epsilon);
  }
};
}  // namespace tilemega::backend
