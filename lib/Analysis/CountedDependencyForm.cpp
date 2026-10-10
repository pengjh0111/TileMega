// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
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

CountedDependencyForm BindAlignedCountedScatterDependency(
    OperatorNode const& producer, OperatorNode const& consumer,
    std::string const& tensor, std::vector<unsigned> const& unit_axes,
    std::string const& binding_source, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  if(!producer.element_access || !consumer.element_access || tensor.empty())
    throw std::invalid_argument("counted scatter requires exact task accesses");
  auto const& pa=*producer.element_access;auto const& ca=*consumer.element_access;
  CouplingRelation writes,reads;unsigned stores=0;
  auto add_write=[&](TensorSpace const& target,IndexingMap const& map,
                     std::vector<IndexResult> const& predicates) {
    if(target.name!=tensor)return;
    if(++stores!=1)throw std::invalid_argument("counted scatter needs one declared store");
    std::vector<unsigned> indexed;
    for(unsigned axis=0;axis<map.results.size();++axis)
      if(map.results[axis].kind==IndexResult::Kind::kDataDependent) {
        if(map.results[axis].binding_source!=binding_source || binding_source.empty())
          throw std::invalid_argument("counted scatter uses a different binding source");
        indexed.push_back(axis);
      }
    if(indexed.empty() || indexed!=unit_axes)
      throw std::invalid_argument("counted scatter units must be its bound store axes");
    writes=writes.Union(ProjectTaskWrite(pa.semantic,producer,pa.partition,target,map,predicates,known));
  };
  add_write(pa.semantic.result,pa.semantic.result_map,{});
  for(auto const& side:pa.semantic.additional_writes)add_write(side.tensor,side.map,side.nonnegative);
  auto add_read=[&](TensorSpace const& target,IndexingMap const& map,
                    std::vector<IndexResult> const& predicates) {
    if(target.name!=tensor)return;
    if(std::any_of(map.results.begin(),map.results.end(),[](auto const& axis) {
         return axis.kind==IndexResult::Kind::kDataDependent;}))
      throw std::invalid_argument("counted scatter consumer units must be static");
    reads=reads.Union(ProjectTaskRead(ca.semantic,consumer,ca.partition,target,map,predicates,known));
  };
  if(!ca.semantic.element_reads.empty()) {
    for(auto const& read:ca.semantic.element_reads)add_read(read.tensor,read.map,read.nonnegative);
  } else for(auto const& read:ca.semantic.operands)
    // A split combiner owns the original logical tensor; its physical stage
    // name differs from the producer retained in the source L-sem.
    add_read(read.tensor,read.map,{});
  if(writes.empty() || reads.empty())
    throw std::invalid_argument("counted scatter has no shared tensor access");
  auto result=BindCountedTaskDependency(consumer,reads,unit_axes,binding_source,known);
  auto coupling=reads.ApplyRange(writes.Reverse());
  if(!Contains(coupling.ApplyRange(writes),reads))
    throw std::invalid_argument("counted scatter leaves consumer contributions unwritten");
  auto columns=[](CouplingRelation relation,std::vector<unsigned> const& units) {
    for(auto axis=units.rbegin();axis!=units.rend();++axis)relation=relation.ProjectRange(*axis,1);
    return relation;
  };
  auto read_columns=columns(reads,unit_axes),write_columns=columns(writes,unit_axes);
  auto producer_ids=writes.Reverse().Image().ImageIdentity();
  auto required=coupling.RangeProduct(read_columns);
  auto provided=coupling.ApplyRange(producer_ids.RangeProduct(write_columns));
  if(!Contains(required,provided) || !Contains(provided,required))
    throw std::invalid_argument("counted scatter producer and consumer columns are not aligned: " +
        producer.name + " -> " + consumer.name);
  return result;
}
} // namespace tilemega::analysis
