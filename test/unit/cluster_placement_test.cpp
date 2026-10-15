// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/ClusterPlacement.h>
#include <cassert>
#include <cstdio>
#include <functional>
struct Task { unsigned stage, logical_task; };
int main() {
  using namespace tilemega::codegen::executor;
  unsigned checks=0;
  auto reject=[&](std::function<void()> fn) {
    bool rejected=false;try { fn(); }catch(std::invalid_argument const&) { rejected=true; }
    assert(rejected);++checks;
  };
  std::vector<unsigned> counts={4,2,0};
  std::vector<std::uint32_t> offsets={0,2,4,5,6};
  std::vector<Task> tasks={{0,0},{1,0},{0,1},{1,1},{0,2},{0,3}};
  for(unsigned c:{1u,2u,4u}) {
    auto owners=ValidateClusterPlacement(c,4,counts,offsets,tasks);
    assert(owners[0][3].worker==3 && owners[1][1].slot==1);++checks;
  }
  reject([&]{ValidateClusterPlacement(3,4,counts,offsets,tasks);});
  reject([&]{ValidateClusterPlacement(4,2,counts,offsets,tasks);});
  reject([&]{ValidateClusterPlacement(2,0,counts,offsets,tasks);});
  reject([&]{auto o=offsets;o.pop_back();ValidateClusterPlacement(2,4,counts,o,tasks);});
  reject([&]{auto o=offsets;o[0]=1;ValidateClusterPlacement(2,4,counts,o,tasks);});
  reject([&]{auto o=offsets;o[1]=7;ValidateClusterPlacement(2,4,counts,o,tasks);});
  reject([&]{auto t=tasks;t.back().logical_task=0;ValidateClusterPlacement(2,4,counts,offsets,t);});
  reject([&]{auto t=tasks;t.back().stage=3;ValidateClusterPlacement(2,4,counts,offsets,t);});
  reject([&]{auto t=tasks;t.back().logical_task=4;ValidateClusterPlacement(2,4,counts,offsets,t);});
  reject([&]{auto c=counts;++c[0];ValidateClusterPlacement(2,4,c,offsets,tasks);});
  ValidateClusterCooperation(2,{{2,0},{3,1}});++checks;
  ValidateClusterCooperation(4,{{7,0},{4,3},{6,1},{5,2}});++checks;
  reject([&]{ValidateClusterCooperation(2,{{1,0},{2,0}});});
  reject([&]{ValidateClusterCooperation(4,{{0,0},{0,1}});});
  reject([&]{ValidateClusterCooperation(1,{{0,0}});});
  reject([&]{ValidateClusterCooperation(2,{});});
  reject([&]{ValidateClusterCooperation(2,{{0,0},{1,0},{2,0}});});
  std::printf("PASS cluster placement %u checks\n",checks);
}
