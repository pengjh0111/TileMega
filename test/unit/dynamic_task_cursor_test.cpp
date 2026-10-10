// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/DynamicTaskCursor.h>
#include <algorithm>
#include <cassert>
#include <vector>
namespace tilemega::tests::dynamic_task_cursor_test {
int TestDynamicTaskCursor(int,char**) {
  using namespace codegen;
  // Static queue slots 1..7 belong to different workers, but the fast worker
  // may claim more than its old share. Static entry/exit owners are retained.
  DynamicStageRange ranges[3][3]={
    {{0,1,0,3,0},{1,3,3,7,1},{3,4,10,3,0}},
    {{4,5,0,3,0},{5,8,3,7,1},{8,9,10,3,0}},
    {{9,10,0,3,0},{10,12,3,7,1},{12,13,10,3,0}}};
  std::uint32_t canonical[]={0,4,9,1,2,5,6,7,10,11,3,8,12};
  DynamicTaskCursor cursors[3];std::uint32_t next=0;
  auto claim=[&](std::uint32_t stage,std::uint32_t count) {
    assert(stage==1 && count==7);return next<count?next++:count;
  };
  std::vector<unsigned> seen;
  for(unsigned w=0;w<3;++w)seen.push_back(cursors[w].Next(ranges[w],3,canonical,claim));
  for(unsigned i=0;i<5;++i)seen.push_back(cursors[0].Next(ranges[0],3,canonical,claim));
  assert(next==5);
  for(unsigned w:{1,2,0})while(true) {
    auto slot=cursors[w].Next(ranges[w],3,canonical,claim);
    if(slot==~std::uint32_t(0))break;
    seen.push_back(slot);
  }
  assert(next==7);std::sort(seen.begin(),seen.end());
  assert(seen.size()==13);
  for(unsigned i=0;i<seen.size();++i)assert(seen[i]==i);
  DynamicStageRange empty{0,0,0,0,1};DynamicTaskCursor cursor;
  assert(cursor.Next(&empty,1,canonical,[](auto,auto){assert(false);return 0;})==~std::uint32_t(0));
  return 0;
}
} // namespace tilemega::tests::dynamic_task_cursor_test
