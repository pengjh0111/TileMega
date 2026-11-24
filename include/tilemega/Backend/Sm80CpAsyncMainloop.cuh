// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace tilemega::backend {

struct Sm80CpAsyncMainloop {
  struct NoObserver {
    __device__ void operator()(bool) const {}
  };

  template<int Pending>
  __device__ static __forceinline__ void Wait(int groups) {
    if(groups>=Pending)
      asm volatile("cp.async.wait_group %0;"::"n"(Pending):"memory");
    else if constexpr(Pending>0)
      Wait<Pending-1>(groups);
  }

  template<int Slots,class Issue,class Consume,class Observe=NoObserver>
  __device__ static __forceinline__ void Run(int iterations,Issue issue,Consume consume,
      Observe observe={}) {
    static_assert(Slots>=2 && Slots<=8);
    auto enqueue=[&](int iteration) {
      issue(iteration%Slots,iteration);
      asm volatile("cp.async.commit_group;":::"memory");
    };
#if defined(TILEMEGA_SM80_MULTISTAGE_SINGLE_JOIN)
    // With one fewer pending group, the slot written at iteration i was
    // last read at i-2. The preceding CTA join has retired all those reads.
    constexpr int ahead=Slots-2;
#else
    constexpr int ahead=Slots-1;
#endif
    for(int i=0;i<ahead && i<iterations;++i)enqueue(i);
    for(int i=0;i<iterations;++i) {
      if(i+ahead<iterations)enqueue(i+ahead);
      observe(true);
      Wait<ahead>(iterations-i-1);
      // A warp's wait retires only its own copies, while each MMA can read
      // operands supplied by every warp. Join before reading and reusing a slot.
      __syncthreads();
      observe(false);
      consume(i%Slots,i);
#if !defined(TILEMEGA_SM80_MULTISTAGE_SINGLE_JOIN)
      __syncthreads();
#endif
    }
#if defined(TILEMEGA_SM80_MULTISTAGE_SINGLE_JOIN)
    // The epilogue and the next task can reuse the current operand arena.
    __syncthreads();
#endif
  }
};

}  // namespace tilemega::backend
