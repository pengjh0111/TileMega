// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>

#if TILEMEGA_TRACE_STAGE != 1 || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY || TILEMEGA_NATIVE_RELEASE_ENABLED
#error "native family probes require only the level-one stage ledger"
#endif

namespace tilemega::codegen::native_dense {
// No predecessor preissues into a probe. Its operand publication is outside
// the measured interval, and every instance ends with the same V3 grid join.
template<class P,class Body,unsigned Index,bool Measured>
__device__ __forceinline__ void ProbeInstance(typename P::Binding const& binding,
    int past,char* shared,float epsilon,GridCounter* events,
    unsigned long long iteration,TraceState trace,int layer) {
  using Stage=Placement<Body,0,0>;
  static_assert(Stage::kSharedBytes<=P::kSharedBytes);
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  auto& operands=backend::Sm90DenseOperandHeader::Bank(shared,0);
  if(threadIdx.x==0) {
    if constexpr(Stage::kUsesOperandRecord)
      Stage::PublishOperands(binding,layer,past,epsilon,operands,iteration);
  }
  __syncthreads();
  StageTrace* record=Measured && trace.stages && trace.slots?
      trace.stages+(iteration%trace.slots)*P::kStages*gridDim.x+Index*gridDim.x+blockIdx.x:nullptr;
  if(threadIdx.x==0 && record) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    *record={iteration,static_cast<unsigned long long>(past),Now(),0,0,
        Stage::kTasks/gridDim.x+(blockIdx.x<Stage::kTasks%gridDim.x),sm,Index};
  }
  Stage::Run(binding,past,shared,epsilon,rings[0],backend::Sm90DenseNoTail{},operands,false);
  if(threadIdx.x==0) {
    if(record)record->arrive=Now();
    auto* counter=&events[Index].arrivals;
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
    unsigned long long seen;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
    while(seen<(iteration+1)*gridDim.x);
  }
  __syncthreads();
  if(threadIdx.x==0 && record)record->release=Now();
}
} // namespace tilemega::codegen::native_dense
