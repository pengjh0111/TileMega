// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tilemega::analysis {
CountedDependencyForm BindCountedTaskDependency(OperatorNode const& consumer,
    CouplingRelation const& reads, std::vector<unsigned> const& unit_axes,
    std::string binding_source, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  if(reads.empty())throw std::invalid_argument("counted dependency needs a typed read relation");
  auto count=consumer.Count().Eval(known,{});
  auto rank=reads.RangeDimNames().size();
  if(reads.empty() || count<=0 || count>std::numeric_limits<std::uint32_t>::max() ||
     binding_source.empty() || unit_axes.empty() || !std::is_sorted(unit_axes.begin(),unit_axes.end()) ||
     std::adjacent_find(unit_axes.begin(),unit_axes.end())!=unit_axes.end() || unit_axes.back()>=rank)
    throw std::invalid_argument("invalid counted dependency contribution domain");
  auto units=reads.BindParams(known);
  for(unsigned axis=rank;axis-- >0;)
    if(!std::binary_search(unit_axes.begin(),unit_axes.end(),axis))units=units.ProjectRange(axis,1);
  auto linear=LinearizeTaskCoordinates(consumer,units.DomainDimNames(),known,"_tm_target");
  CountedDependencyForm result;
  result.binding_source=std::move(binding_source);
  result.target_units=linear.Reverse().ApplyRange(units);
  if(!reads.BindParams(known).IsSubset(linear.ApplyRange(linear.Reverse()).ApplyRange(reads.BindParams(known))))
    throw std::invalid_argument("counted dependency reads exceed consumer task domain");
  std::vector<ParamBinding> coordinates(count);
  for(long target=0;target<count;++target)coordinates[target].Bind("_tm_target",target);
  auto cardinalities=result.target_units.BoundTaskCard().EvalPoints({},coordinates);
  result.expected.reserve(count);
  for(auto value:cardinalities) {
    if(value<0 || value>std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("counted dependency threshold exceeds runtime range");
    result.expected.push_back(std::uint32_t(value));
  }
  return result;
}
} // namespace tilemega::analysis
