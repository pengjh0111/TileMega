// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace tilemega::tests::runtime_dependency_table_test {
int TestRuntimeDependencyTable(int, char**) {
  using namespace tilemega;
  analysis::IslContext context;
  auto f = [](long x) { return analysis::ClosedForm::Constant(x); };
  analysis::OperatorNode p, c;
  p.output = {"p", {{"row", f(12)}}}; p.tile = {f(1)};
  c.output = {"c", {{"row", f(7)}}}; c.tile = {f(1)};
  auto relation = analysis::CouplingRelation::FromIslText(
      "{ [c] -> [p] : 0 <= c < 6 and 0 <= p < 12 and (p+2*c) % 3 != 1 }");
  auto table = analysis::BuildDependencyTable(relation, p, c, {});
  std::vector<codegen::RuntimeDependencyInterval> intervals;
  for (auto const& i : table.intervals) intervals.push_back({i.first, i.count});
  codegen::RuntimeDependencyTableView view{intervals.data(), table.consumers, table.stride};
  for (int workers : {1, 4, 16}) {
    auto graph = codegen::MaterializeRuntimeTaskGraphTables({12, 7}, {}, {{0, 1, view}}, workers);
    for (int source = 0; source < 12; ++source) {
      std::vector<int> expected;
      for (int sink = 0; sink < 6; ++sink) if ((source + 2*sink) % 3 != 1) expected.push_back(12+sink);
      assert(graph.successors[source] == expected);
    }
    auto empty = codegen::MaterializeRuntimeTaskGraphTables({12, 7}, {}, {{0, 1, {nullptr, 7, 0}}}, workers);
    for (auto const& row : empty.successors) assert(row.empty());
    auto duplicate = codegen::MaterializeRuntimeTaskGraphTables({12, 7}, {}, {{0, 1, view}, {0, 1, view}}, workers);
    assert(duplicate.successors == graph.successors);
  }
  auto reject = [](codegen::RuntimeTaskTableDependency edge) {
    bool failed = false;
    try { (void)codegen::MaterializeRuntimeTaskGraphTables({12, 7}, {}, {edge}, 4); }
    catch (std::invalid_argument const&) { failed = true; }
    assert(failed);
  };
  reject({1, 0, view}); reject({0, 2, view}); reject({0, 1, {nullptr, 7, 1}});
  reject({0, 1, {intervals.data(), 8, table.stride}});
  intervals[0] = {11, 2}; reject({0, 1, view});
  auto periodic=analysis::CouplingRelation::FromIslText(
      "{ [c] -> [p] : 0<=c<132 and 0<=p<1160 and p%4=c%4 }");
  auto large=analysis::BuildDependencyTableLinear(periodic,1160,132);
  assert(large.stride==290 && large.intervals.size()==132*290);
  assert(large.encoded_relation.ToString().size()<2000);
  assert(analysis::Contains(periodic,large.encoded_relation) &&
      analysis::Contains(large.encoded_relation,periodic));
  for(unsigned consumer=0;consumer<132;++consumer)for(unsigned entry=0;entry<290;++entry) {
    auto interval=large.intervals[consumer*290+entry];
    assert(interval.first==4*entry+consumer%4 && interval.count==1);
  }
  auto dense=analysis::CouplingRelation::FromIslText(
      "{ [c] -> [p] : 0<=c<4096 and 0<=p<8192 }");
  auto dense_table=analysis::BuildDependencyTableLinear(dense,8192,4097);
  assert(dense_table.stride==1 && dense_table.intervals.size()==4097);
  for(unsigned c=0;c<4096;++c)
    assert(dense_table.intervals[c].first==0 && dense_table.intervals[c].count==8192);
  assert(dense_table.intervals.back().count==0);
  auto overlap=analysis::CouplingRelation::FromIslText(
      "{ [c] -> [p] : c=0 and 0<=p<=4; [c] -> [p] : c=0 and 3<=p<8 }");
  auto merged=analysis::BuildDependencyTableLinear(overlap,8,1);
  assert(merged.stride==1 && merged.intervals[0].first==0 && merged.intervals[0].count==8);
  bool bad_range=false;
  try{(void)analysis::BuildDependencyTableLinear(dense,8191,4096);}
  catch(std::invalid_argument const&){bad_range=true;}
  assert(bad_range);
  analysis::ValidateDependencyTableLinear(large);
  analysis::ValidateDependencyTableLinear(dense_table);
  analysis::ValidateDependencyTableLinear(merged);
  unsigned corruptions=0;
  auto reject_table=[&](analysis::DependencyTable broken) {
    bool caught=false;
    try{analysis::ValidateDependencyTableLinear(broken);}catch(std::invalid_argument const&){caught=true;}
    assert(caught);++corruptions;
  };
  auto corrupt=large;corrupt.intervals[17].first+=1;reject_table(corrupt);
  corrupt=dense_table;corrupt.intervals.back()={1,0};reject_table(corrupt);
  corrupt=dense_table;corrupt.intervals[0].count-=1;reject_table(corrupt);
  corrupt=dense_table;corrupt.linear_relation=dense.Subtract(dense);reject_table(corrupt);
  corrupt=merged;corrupt.stride=2;corrupt.intervals={{0,4},{4,4}};reject_table(corrupt);
  corrupt=merged;corrupt.stride=2;corrupt.intervals.push_back({0,0});reject_table(corrupt);
  assert(corruptions==6);
  std::uint64_t target;
  assert(codegen::CountedDependencyTarget(8, 17, &target) && target == 144);
  assert(!codegen::CountedDependencyTarget(0, 0, &target));
  assert(!codegen::CountedDependencyTarget(8, std::numeric_limits<std::uint64_t>::max()/8, &target));
  return 0;
}
}
