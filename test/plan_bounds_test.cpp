// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ExecutionSimulator.h>
#include <cassert>
#include <cmath>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;
int main() {
  codegen::RuntimeTaskGraph graph{{0, 2, 4}, {{2}, {3}, {}, {}}, {}, 0};
  SimulatorInput in; in.graph=&graph; in.task_ns={3, 5, 7, 11};
  PreparedPlanBounds prepared; std::string error;
  assert(PreparePlanBounds(in,&prepared,&error));
  assert(prepared.critical_path_ns==16 && prepared.work_ns==26);
  MaterializedPlan one, two;
  one.queue={{{0,0},{0,1},{1,0},{1,1}}};
  two.queue={{{0,0},{1,0}},{{0,1},{1,1}}};
  PlanBounds a,b;
  assert(EvaluatePlanBounds(prepared,one,&a,&error));
  assert(EvaluatePlanBounds(prepared,two,&b,&error));
  assert(a.lower_bound_ns==26 && b.lower_bound_ns==16);
  // A zero-hop same-worker dependency retains both node weights.
  SimulatorOptions options; HopCurve hop;
  std::vector<RankedPlan> ranked;
  assert(RankPlans(in,prepared,{&one,&two},options,hop,1,&ranked,&error));
  assert(ranked[0].index==1 && ranked[0].simulated && !ranked[1].simulated);
  assert(ranked[0].makespan_ns==16);
  one.queue[0].pop_back();
  assert(!EvaluatePlanBounds(prepared,one,&a,&error));
  graph.successors[2].push_back(0);
  assert(!PreparePlanBounds(in,&prepared,&error));
  std::cout << "PLAN_BOUNDS PASS dependencies queue omissions cycles top-k\n";
}
