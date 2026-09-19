// SPDX-License-Identifier: BSD-3-Clause
// The interval bound must answer exactly what the materialized edge set
// answers; only the storage in between differs.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/RelationBounds.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tilemega::solver::PrepareRelationBounds;
using tilemega::solver::RelationBounds;

tilemega::codegen::RuntimeTaskGraph Materialize(std::vector<int> const& counts,
    std::string const& relation) {
  auto graph=tilemega::codegen::MaterializeRuntimeTaskGraph(counts,{},1);
  auto node=[&](long stage,long task) {
    assert(stage>=0 && stage<long(counts.size()) && task>=0 && task<counts[stage]);
    return graph.stage_offsets[stage]+int(task);
  };
  tilemega::analysis::VisitFiniteRelation(tilemega::analysis::SharedIslContext(),relation,4,
      [&](long const* e){graph.successors[node(e[2],e[3])].push_back(node(e[0],e[1]));});
  for (auto& row:graph.successors) {
    std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
  }
  return graph;
}

// Prices differ per node so a wrong relaxation cannot hide behind a uniform
// cost: every longest path here has its own length.
std::vector<double> Prices(std::vector<int> const& counts) {
  std::vector<double> prices;
  for (std::size_t s=0;s<counts.size();++s)
    for (int t=0;t<counts[s];++t) prices.push_back(1.0+double(s)*7.0+double(t)*0.5);
  return prices;
}

void Agrees(std::vector<int> const& counts,std::string const& relation) {
  auto graph=Materialize(counts,relation);
  auto prices=Prices(counts);
  tilemega::solver::SimulatorInput input;input.graph=&graph;input.task_ns=prices;
  tilemega::solver::PreparedPlanBounds dense;std::string dense_error;
  bool const dense_ok=PreparePlanBounds(input,&dense,&dense_error);
  RelationBounds intervals;std::string interval_error;
  bool const interval_ok=PrepareRelationBounds(graph.stage_offsets,counts,prices,relation,
      &intervals,&interval_error);
  assert(dense_ok==interval_ok);
  if (!dense_ok) {assert(interval_error.find("cyclic")!=std::string::npos);return;}
  assert(std::abs(dense.work_ns-intervals.work_ns)<1e-9);
  assert(std::abs(dense.critical_path_ns-intervals.critical_path_ns)<1e-9);
}
}  // namespace

int main() {
  tilemega::analysis::IslContext context;
  // A complete bipartite block, the case the intervals exist for.
  Agrees({4,4},"{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<4 and 0<=j<4 }");
  // Wide dense slices: one box per fixed consumer coordinate.
  Agrees({4,2048},"{ [s,i] -> [t,j] : s=0 and t=1 and 0<=i<4 and 512*i<=j<512*i+512 }");
  // Modular holes keep the point route on both sides.
  Agrees({8,8},"{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<8 and 0<=j<8 and (i+j)%2=0 }");
  // Overlapping pieces repeat edges; the interval walk must still terminate.
  Agrees({6,6},"{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<4 and 0<=j<3; "
               "[s,i] -> [t,j] : s=1 and t=0 and 2<=i<6 and 1<=j<4 }");
  // A chain of three stages, and a diagonal inside one block.
  Agrees({3,3,3},"{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<3 and 0<=j<3; "
                 "[s,i] -> [t,j] : s=2 and t=1 and 0<=i<3 and j=i }");
  Agrees({5,5},"{ [s,i] -> [t,j] : false }");
  // A block that reaches back into its own producers is a cycle either way.
  Agrees({4},"{ [s,i] -> [t,j] : s=0 and t=0 and 0<=i<4 and 0<=j<4 }");

  {
    RelationBounds bounds;std::string error;
    assert(!PrepareRelationBounds({0,4},{4},{1,1,1,1},
        "{ [s,i] -> [t,j] : s=0 and t=0 and 0<=i<4 and 4<=j<8 }",&bounds,&error));
    assert(error.find("outside task domain")!=std::string::npos);
    assert(!PrepareRelationBounds({0,2},{2},{1,std::nan("")},
        "{ [s,i] -> [t,j] : false }",&bounds,&error));
    assert(error.find("finite nonnegative")!=std::string::npos);
  }

  std::cout<<"relation bounds ok\n";
  return 0;
}
