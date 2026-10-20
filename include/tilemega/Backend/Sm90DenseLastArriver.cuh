// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cstdint>

namespace tilemega::backend {

struct Sm90DenseLastArriver {
  template<unsigned Chunks>
  __device__ static bool Publish(std::uint64_t* counter,unsigned* shared_last) {
    static_assert(Chunks>=1 && Chunks<=64);
    if constexpr(Chunks==1)return true;
    else {
      // Generic stores reach the elected thread through the CTA barrier;
      // acq_rel RMWs then chain all producers to the last observer. PTX 8.7
      // sections 8.8, 8.9.4 and 8.9.5 define this transitive causality path.
      // https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model
      codegen::executor::ComputeSync();
      if(threadIdx.x==0) {
        unsigned long long old;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
            :"=l"(old):"l"(counter),"l"(1ull):"memory");
        *shared_last=old%Chunks==Chunks-1;
      }
      codegen::executor::ComputeSync();
      return *shared_last!=0;
    }
  }
};

}  // namespace tilemega::backend
