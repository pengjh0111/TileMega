// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define TILEMEGA_CURSOR_HD __host__ __device__
#else
#define TILEMEGA_CURSOR_HD
#endif

namespace tilemega::codegen {
struct DynamicStageRange {
  std::uint32_t first=0,past=0,task_begin=0,task_count=0,dynamic=0;
};

// Each worker advances through the DAG in stage order. Static stages retain
// their owners; virtual stages drain a shared saturating claim counter. There
// is no inter-worker stage barrier: each task still waits on its own edges.
struct DynamicTaskCursor {
  std::uint32_t stage=0,local=0;
  template<class Claim>
  TILEMEGA_CURSOR_HD std::uint32_t Next(DynamicStageRange const* ranges,
      std::uint32_t stages,std::uint32_t const* canonical,Claim claim) {
    while(stage<stages) {
      auto const& range=ranges[stage];
      if(range.dynamic && range.task_count) {
        auto task=claim(stage,range.task_count);
        if(task<range.task_count)return canonical[range.task_begin+task];
      }else if(range.first+local<range.past)return range.first+local++;
      ++stage;local=0;
    }
    return ~std::uint32_t(0);
  }
};
} // namespace tilemega::codegen
#undef TILEMEGA_CURSOR_HD
