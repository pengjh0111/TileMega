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
    JointEvaluation r;r.status="ok";r.simulated=true;r.makespan_ns=20;return std::vector<JointEvaluation>{r};
  },&stats);
  assert(out.size()==1 && stats.evaluated==1 && stats.pruned==1 && stats.capacity_deferred==1);
  std::cout<<"JOINT_SEARCH PASS admissible-pruning explicit-capacity\n";
}
