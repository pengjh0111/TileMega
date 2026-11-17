// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseSequence.cuh>
#include <tilemega/Codegen/tasks/Sm90DenseStaticPlan.cuh>
namespace tilemega::codegen::native_dense {
template<class P>
__global__ __launch_bounds__(160,1)
void sm90_static_sequence_unit(typename P::Binding const* b,SequenceState const* state,
    int const* pasts,int unit,float epsilon,unsigned long long* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  if(unit==0 && blockIdx.x==0 && threadIdx.x==0)stamps[state->step]=Now();
  P::Unit(unit,*b,pasts[state->step],shared,epsilon,state->step);
  rings[0].Invalidate();rings[1].Invalidate();
}
template<class P>
struct StaticTaskSequence {
  static void Check(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
  static void Configure() {
    auto kernel=sm90_static_sequence_unit<P>;
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,P::kSharedBytes));
    int resident;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,160,P::kSharedBytes));
    if(resident<1)throw std::invalid_argument("static sequence cannot keep one track per SM resident");
  }
  static void Capture(typename P::Binding const* b,SequenceState* state,int const* pasts,
      float epsilon,unsigned long long* stamps,int grid,cudaStream_t stream,bool=true) {
    for(unsigned unit=0;unit<P::kStages;++unit)
      sm90_static_sequence_unit<P><<<grid,160,P::kSharedBytes,stream>>>(b,state,pasts,unit,epsilon,stamps);
    sm90_dense_sequence_advance<<<1,1,0,stream>>>(state,stamps);Check(cudaGetLastError());
  }
};
} // namespace tilemega::codegen::native_dense
