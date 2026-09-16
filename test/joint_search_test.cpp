// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/JointSearch.h>
#include <cassert>
#include <iostream>
using namespace tilemega::solver;
int main(){
  JointCandidate a,b,c;a.key="a";b.key="b";c.key="c";
  a.priority_ns=0;b.priority_ns=1;c.priority_ns=2;c.cp_lb_ns=30;
  JointSearchStats stats;
  auto out=SearchL2Configurations({a,b,c},1,[](auto const&){
    JointEvaluation r;r.status="ok";r.simulated=true;r.makespan_ns=20;r.floor_ns=10;return std::vector<JointEvaluation>{r};
  },&stats);
  assert(out.size()==1 && stats.evaluated==1 && stats.pruned==1 && stats.capacity_deferred==1);
  a.queue_lb_lb_ns=100;a.priority_ns=2;
  auto ranked=SearchL2Configurations({b,a},2,[](auto const&){
    std::vector<JointEvaluation> values;
    for (int i=0;i<6;++i) {
      JointEvaluation r;r.status="ok";r.simulated=true;r.placement=std::to_string(i);
      r.floor_ns=10+i;r.makespan_ns=100-i;values.push_back(r);
    }
    return values;
  },&stats);
  assert(ranked.size()==6 && stats.pruned==1 && stats.evaluated==1);
  assert(ranked.front().placement=="0" && ranked.back().placement=="5");
  std::cout<<"JOINT_SEARCH PASS admissible-pruning explicit-capacity\n";
}
