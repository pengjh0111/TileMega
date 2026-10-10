// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>

using Arch=typename tilemega::arch::ArchFromId<TILEMEGA_ARCH_ID>::type;
using Operands=tilemega::codegen::ServingGemmOperands;
using Paged=tilemega::codegen::PagedGemmTaskBody<Arch,16,32,64,8192,2>;

template<int N>
__global__ void GemvDense(Operands p,float* output) {
  extern __shared__ char shared[];
  using Body=tilemega::backend::ServingGemv<Arch,16,N,64>;
  auto* tile=Body::Dense(p,0,0,shared,true);
  for(int i=threadIdx.x;i<16*N;i+=blockDim.x)output[i]=tile[i];
}
template __global__ void GemvDense<8>(Operands,float*);
template __global__ void GemvDense<16>(Operands,float*);
template __global__ void GemvDense<32>(Operands,float*);

__global__ void GemvPages(Operands p,float* output) {
  using namespace tilemega::codegen::executor;
  extern __shared__ char shared[];
  Paged::Ring ring{reinterpret_cast<Paged::Ring::Slot*>(shared),shared+8192};
  ring.Initialize();std::uint64_t sequence=0;
  if(IsCompute()) {
    using Body=tilemega::backend::ServingGemv<Arch,16,32,64>;
    auto* tile=Body::Paged(p,0,0,ring,sequence,shared+1024,Paged::NoPhaseGate{});
    for(int i=ComputeThread();i<16*32;i+=kComputeThreads)output[i]=tile[i];
  }else Paged::Load(p,0,ring,sequence);
}
