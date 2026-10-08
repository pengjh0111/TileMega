// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeCountedThresholds.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::counted_threshold_test {
int TestCountedThreshold(int,char**) {
  using namespace tilemega::codegen;
  std::uint32_t counts[]={999,8,8,2,777};
  RuntimeCountedThresholdView table{counts,5};
  constexpr auto absent=std::numeric_limits<std::uint32_t>::max();
  std::uint32_t count;std::uint64_t target;
  for(unsigned task=0;task<3;++task) {
    for(std::uint64_t iteration:{0ull,1ull,17ull,10000000000ull}) {
      assert(ReadCountedThreshold(table,1,3,task,0,&count) && count==counts[task+1]);
      assert(CountedThresholdTarget(table,1,3,task,0,iteration,&target));
      assert(target==counts[task+1]*(iteration+1));
      assert(CountedThresholdTarget({},absent,0,task,7,iteration,&target));
      assert(target==7*(iteration+1));
    }
  }
  assert(!ReadCountedThreshold(table,1,3,3,8,&count));
  assert(!ReadCountedThreshold(table,4,3,0,8,&count));
  assert(!ReadCountedThreshold(table,1,0,0,8,&count));
  assert(!ReadCountedThreshold({},1,3,0,8,&count));
  assert(!ReadCountedThreshold(table,1,3,0,8,nullptr));
  assert(!ReadCountedThreshold({},absent,0,0,0,&count));
  assert(!ReadCountedThreshold(table,absent-1,3,0,8,&count));
  counts[3]=0;assert(!ReadCountedThreshold(table,1,3,2,8,&count));counts[3]=2;
  assert(!CountedThresholdTarget(table,1,3,0,0,UINT64_MAX,&target));
  assert(!CountedThresholdTarget(table,1,3,0,0,UINT64_MAX/8,&target));
  assert(!CountedThresholdTarget(table,1,3,0,0,0,nullptr));
  std::cout<<"Counted thresholds: offsets, consumer tails, uniform compatibility and epoch bounds PASS\n";
  return 0;
}
} // namespace tilemega::tests::counted_threshold_test
