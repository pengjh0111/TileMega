// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/StorageHazards.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <set>

namespace tilemega::tests::storage_hazards_test {
namespace {
using namespace analysis;
ClosedForm F(long x) { return ClosedForm::Constant(x); }
OperatorNode Task(char const* name, long tasks) {
  OperatorNode node; node.name = name;
  node.output = {"ownership", {{"t", F(tasks)}}}; node.tile = {F(1)};
  return node;
}
using Points = std::set<std::pair<std::vector<long>, std::vector<long>>>;
Points Listed(CouplingRelation const& relation) {
  auto points = relation.Points(); return {points.begin(), points.end()};
}
void Reuse() {
  auto old = Task("old", 20), first = Task("first", 6), second = Task("second", 4), next = Task("next", 7);
  StorageTaskAccess writes{&old, CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 20 and 4*t <= i < 4*t+4 }")};
  StorageTaskAccess read1{&first, CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 6 and 8*t <= i < 8*t+8 }")};
  StorageTaskAccess read2{&second, CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 4 and (i=64+4*t or i=66+4*t) }")};
  StorageTaskAccess replacement{&next, CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 7 and 13*t <= i < 13*t+13 and i < 80 }")};
  auto hazards = DeriveStorageReuseHazards(writes, {read1, read2}, replacement);
  assert(hazards.size() == 3);
  Points expected1, expected2, expected_waw;
  for (long element = 0; element < 80; ++element) {
    if (element < 48) expected1.insert({{element / 13}, {element / 8}});
    else if (element >= 64 && element % 2 == 0) expected2.insert({{element / 13}, {(element - 64) / 4}});
    else expected_waw.insert({{element / 13}, {element / 4}});
  }
  for (unsigned edge = 0; edge < hazards.size(); ++edge) {
    auto const& hazard = hazards[edge];
    assert(hazard.kind == (edge < 2 ? StorageHazardKind::kWAR : StorageHazardKind::kWAW));
    assert(Listed(hazard.coupling.C) == (edge == 0 ? expected1 : edge == 1 ? expected2 : expected_waw));
    assert(hazard.coupling.exact && hazard.coupling.tier == Tier::kAffine);
    auto const& producer = edge == 0 ? first : edge == 1 ? second : old;
    auto table = BuildDependencyTable(hazard.coupling.C, producer, next, {});
    assert(Contains(table.encoded_relation, hazard.coupling.C));
  }
  auto no_read = DeriveStorageReuseHazards(writes, {}, replacement);
  assert(no_read.size() == 1 && no_read[0].kind == StorageHazardKind::kWAW);
  auto full_reader = read1;
  full_reader.elements = CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 6 and 0 <= i < 80 }");
  auto all_read = DeriveStorageReuseHazards(writes, {full_reader}, replacement);
  assert(all_read.size() == 1 && all_read[0].kind == StorageHazardKind::kWAR);
  bool rejected = false;
  try {
    auto overlapping = replacement;
    overlapping.elements = CouplingRelation::FromIslText("{ [t] -> [i] : 0 <= t < 7 and 0 <= i < 80 }");
    (void)DeriveStorageReuseHazards(writes, {}, overlapping);
  } catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected);
}
}
int TestStorageHazards(int, char**) { IslContext context; Reuse(); return 0; }
}
