// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace tilemega::codegen::executor {
// Sequential launches share a monotonic counter. Saturation consumes exactly
// count tickets, including empty virtual tasks, so idle workers cannot move
// the next iteration's origin. No clearing kernel or stage barrier is needed.
__device__ inline std::uint32_t ClaimDynamicTask(unsigned long long* counter,
    std::uint32_t count,unsigned long long iteration) {
  auto begin=iteration*static_cast<unsigned long long>(count),end=begin+count;
  auto observed=atomicAdd(counter,0ull);
  while(true) {
    if(observed<begin) {
      auto prior=atomicCAS(counter,observed,begin);
      observed=prior==observed?begin:prior;
    }else if(observed>=end)return count;
    else {
      auto prior=atomicCAS(counter,observed,observed+1);
      if(prior==observed)return std::uint32_t(observed-begin);
      observed=prior;
    }
  }
}
} // namespace tilemega::codegen::executor
