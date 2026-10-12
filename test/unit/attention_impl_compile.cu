// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>
using Arch=tilemega::arch::CurrentArch;
using namespace tilemega::codegen;
template<int D,int Q>
__global__ void AttentionImplCompile(ServingAttentionOperands p) {
  using Body=PagedAttentionTaskBody<Arch,D,Q,false,16384,5>;
  using Ring=typename Body::Ring;
  extern __shared__ __align__(1024) char memory[];
  constexpr unsigned offset=1024+(sizeof(typename Body::SharedStorage)+1023)/1024*1024;
  Ring ring{reinterpret_cast<typename Ring::Slot*>(memory),memory+offset};
  ring.Initialize();std::uint64_t sequence=0;
  if(executor::IsCompute())Body::Run(p,0,0,0,ring,sequence,
      *reinterpret_cast<typename Body::SharedStorage*>(memory+1024));
  else Body::Load(p,0,0,0,ring,sequence);
}
template __global__ void AttentionImplCompile<64,4>(ServingAttentionOperands);
template __global__ void AttentionImplCompile<128,3>(ServingAttentionOperands);
#if TILEMEGA_ATTENTION_IMPL!=3
template __global__ void AttentionImplCompile<128,8>(ServingAttentionOperands);
#endif
