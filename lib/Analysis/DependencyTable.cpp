// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
std::string Tuple(std::vector<std::string> const& names) {
  std::string out;
  for (auto const& name : names) { if (!out.empty()) out += ','; out += name; }
  return "[" + out + "]";
}
std::uint32_t Count(OperatorNode const& node, ParamBinding const& known) {
  auto count = node.Count().Eval(known, {});
  if (count <= 0 || count > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("dependency table task count outside 32-bit range");
  return static_cast<std::uint32_t>(count);
}
CouplingRelation Linearization(OperatorNode const& node, std::vector<std::string> const& names,
                              ParamBinding const& known, char const* id) {
  std::vector<long> extents;
  for (unsigned axis = 0; axis < node.output.axes.size(); ++axis)
    if (node.IsTiled(axis)) extents.push_back(node.CoordinateExtent(axis).Eval(known, {}));
  if (names.size() != extents.size())
    throw std::invalid_argument("dependency table coordinate rank mismatch");
  std::string expression = "0", bounds;
  for (unsigned axis = 0; axis < names.size(); ++axis) {
    if (extents[axis] <= 0) throw std::invalid_argument("dependency table has nonpositive extent");
    expression = "(" + expression + ") * " + std::to_string(extents[axis]) + " + " + names[axis];
    bounds += " and 0 <= " + names[axis] + " < " + std::to_string(extents[axis]);
  }
  return CouplingRelation::FromIslText("{ " + Tuple(names) + " -> [" + id + "] : " +
      id + " = " + expression + bounds + " }");
}
}
CouplingRelation LinearizeTaskCoordinates(OperatorNode const& node,
    std::vector<std::string> const& coordinates, ParamBinding const& known, char const* id) {
  return Linearization(node, coordinates, known, id);
}
CouplingRelation LinearizeTaskCoupling(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  if (relation.empty()) throw std::invalid_argument("dependency table requires a typed relation");
  auto concrete = relation.BindParams(known);
  auto c = Linearization(consumer, concrete.DomainDimNames(), known, "_tm_c");
  auto p = Linearization(producer, concrete.RangeDimNames(), known, "_tm_p");
  return c.Reverse().ApplyRange(concrete).ApplyRange(p);
}
DependencyTable BuildDependencyTable(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  return BuildDependencyTableLinear(LinearizeTaskCoupling(relation, producer, consumer, known),
                                   Count(producer, known), Count(consumer, known));
}
DependencyTable BuildDependencyTableLinear(CouplingRelation const& relation,
    std::uint32_t producers, std::uint32_t consumers) {
  IslReferenceAudit audit(__func__);
  if (!producers || !consumers || relation.DomainDimNames().size() != 1 ||
      relation.RangeDimNames().size() != 1)
    throw std::invalid_argument("invalid linear dependency table domain");
  DependencyTable result;
  result.consumers = consumers; result.producers = producers;
  result.linear_relation = relation;
  std::vector<std::set<std::uint32_t>> rows(result.consumers);
  for (auto const& [to, from] : result.linear_relation.Points()) {
    if (to.size() != 1 || from.size() != 1 || to[0] < 0 || from[0] < 0 ||
        to[0] >= result.consumers || from[0] >= result.producers)
      throw std::invalid_argument("dependency table includes an out-of-range task");
    rows[to[0]].insert(static_cast<std::uint32_t>(from[0]));
  }
  std::vector<std::vector<TaskInterval>> intervals(result.consumers);
  for (unsigned task = 0; task < result.consumers; ++task) {
    for (auto source : rows[task]) {
      auto& row = intervals[task];
      if (!row.empty() && row.back().first + row.back().count == source) ++row.back().count;
      else row.push_back({source, 1});
    }
    result.stride = std::max(result.stride, static_cast<std::uint32_t>(intervals[task].size()));
  }
  if (std::uint64_t(result.consumers) * result.stride > std::numeric_limits<std::size_t>::max() / sizeof(TaskInterval))
    throw std::invalid_argument("dependency table storage size overflows");
  result.intervals.resize(std::size_t(result.consumers) * result.stride);
  std::string encoded = "{ ";
  bool first = true;
  for (unsigned task = 0; task < result.consumers; ++task) {
    std::copy(intervals[task].begin(), intervals[task].end(),
              result.intervals.begin() + std::size_t(task) * result.stride);
    for (auto const& interval : intervals[task]) {
      if (!first) encoded += "; "; first = false;
      encoded += "[_tm_c] -> [_tm_p] : _tm_c = " + std::to_string(task) +
          " and " + std::to_string(interval.first) + " <= _tm_p < " +
          std::to_string(std::uint64_t(interval.first) + interval.count);
    }
  }
  if (first) encoded += "[_tm_c] -> [_tm_p] : false";
  result.encoded_relation = CouplingRelation::FromIslText(encoded + " }");
  if (!Contains(result.encoded_relation, result.linear_relation) ||
      !Contains(result.linear_relation, result.encoded_relation))
    throw std::logic_error("dependency table failed exact containment proof");
  return result;
}
}  // namespace tilemega::analysis
