// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace tilemega::backend {

struct Sm80CpAsyncMainloop {
  template<int Pending>
  __device__ static __forceinline__ void Wait(int groups) {
    if(groups>=Pending)
      asm volatile("cp.async.wait_group %0;"::"n"(Pending):"memory");
    else if constexpr(Pending>0)
      Wait<Pending-1>(groups);
  }

  template<int Slots,class Issue,class Consume>
  __device__ static __forceinline__ void Run(int iterations,Issue issue,Consume consume) {
    static_assert(Slots>=2 && Slots<=8);
    auto enqueue=[&](int iteration) {
      issue(iteration%Slots,iteration);
      asm volatile("cp.async.commit_group;":::"memory");
    };
    for(int i=0;i<Slots-1 && i<iterations;++i)enqueue(i);
    for(int i=0;i<iterations;++i) {
      if(i+Slots-1<iterations)enqueue(i+Slots-1);
      Wait<Slots-1>(iterations-i-1);
      // A warp's wait retires only its own copies, while each MMA can read
      // operands supplied by every warp. Join before reading and reusing a slot.
      __syncthreads();
      consume(i%Slots,i);
      __syncthreads();
    }
  }
};

}  // namespace tilemega::backend
