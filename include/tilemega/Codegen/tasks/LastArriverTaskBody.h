// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <tilemega/Codegen/tasks/ServingGemmCombineTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>
namespace tilemega::codegen {
template<int TileM,int TileN,backend::ServingEpilogueOp Op>
struct LastArriverGemmTaskBody {
  template<class... Args>
  __device__ static bool Run(unsigned* ticket,unsigned split_count,unsigned* shared_last,
                            Args... args) {
    return executor::LastArriver::Run(ticket,split_count,shared_last,[&]{
      ServingGemmCombineTaskBody<TileM,TileN,Op>::Run(args...);
    });
  }
};
template<int D,int Q,int S>
struct LastArriverAttentionTaskBody {
  __device__ static bool Run(unsigned* ticket,unsigned* shared_last,int block,
      float const* partial,float const* lse,cutlass::bfloat16_t* output,
      int batch,int group,int heads,int capacity,int extent,int past) {
    unsigned live=(past+S+extent-1)/extent;
    if(block*extent>=past+S)return false;
    return executor::LastArriver::Run(ticket,live,shared_last,[&]{
      AttentionMergeTaskBody<D,Q,S>::Run(partial,lse,output,batch,group,heads,capacity,extent,past);
    });
  }
};
} // namespace tilemega::codegen
