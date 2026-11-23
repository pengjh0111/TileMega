// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/GemmTaskBody.h>

#if TILEMEGA_SM80_SPLIT_K
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cstddef>
#include <cstdint>
#endif

namespace tilemega::codegen {
template <class Arch, int Stages>
struct GemmSplitKTaskBody : GemmTaskBody<Arch, Stages> {};

#if TILEMEGA_SM80_SPLIT_K
template<int Tiles,int SplitK,int Rows,int Columns,bool Norm>
struct Sm80SplitKTaskBody {
  static_assert(SplitK==2 || SplitK==4);
  static constexpr int kElements=Rows*Columns;
  static constexpr std::size_t kValues=std::size_t(Tiles)*SplitK*(kElements+(Norm?Rows:0));
  static constexpr std::size_t kPartialBytes=kValues*sizeof(float);

  __device__ static __forceinline__ void Store(float* dst,float value) {
    asm volatile("st.relaxed.gpu.global.f32 [%0], %1;"::"l"(dst),"f"(value):"memory");
  }
  __device__ static __forceinline__ float Load(float const* src) {
    float value;
    asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];":"=f"(value):"l"(src):"memory");
    return value;
  }

  template<int WarpSplits>
  __device__ static __forceinline__ bool Reduce(float* workspace,std::uint64_t* counters,
      int tile,int slice,float* values,float* norms,int* last,int live_rows=Rows) {
    constexpr int kStride=kElements+(Norm?Rows:0);
    auto* partial=workspace+std::size_t(tile)*SplitK*kStride;
    auto* own=partial+slice*kStride;
    // The partials cross SMs, whose L1 caches are not coherent. GPU-scope
    // relaxed accesses are published and collected by the last-arriver RMW.
    for(int i=int(threadIdx.x);i<live_rows*Columns;i+=128) {
      float sum=values[i];
      #pragma unroll
      for(int split=1;split<WarpSplits;++split)sum+=values[split*16*Columns+i];
      Store(own+i,sum);
    }
    if constexpr(Norm)if(threadIdx.x<unsigned(live_rows)) {
      float sum=norms[threadIdx.x];
      #pragma unroll
      for(int split=1;split<WarpSplits;++split)sum+=norms[split*16+threadIdx.x];
      Store(own+kElements+threadIdx.x,sum);
    }
    executor::ComputeSync();
    if(threadIdx.x==0) {
      unsigned long long old;
      asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;"
          :"=l"(old):"l"(counters+tile):"memory");
      *last=int(old+1)==SplitK;
      if(*last)counters[tile]=0;
    }
    executor::ComputeSync();
    bool finish=*last!=0;
    if(finish) {
      for(int i=int(threadIdx.x);i<live_rows*Columns;i+=128) {
        float sum=0;
        #pragma unroll
        for(int part=0;part<SplitK;++part)sum+=Load(partial+part*kStride+i);
        values[i]=sum;
      }
      if constexpr(Norm)if(threadIdx.x<unsigned(live_rows)) {
        float sum=0;
        #pragma unroll
        for(int part=0;part<SplitK;++part)sum+=Load(partial+part*kStride+kElements+threadIdx.x);
        norms[threadIdx.x]=sum;
      }
      executor::ComputeSync();
    }
    return finish;
  }
};
#endif
}  // namespace tilemega::codegen
