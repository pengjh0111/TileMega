// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include "IslUtil.h"
#include <isl/ilp.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

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
using Runs=std::vector<std::pair<std::uint32_t,std::uint32_t>>;
std::vector<TaskInterval> ProducerIntervals(isl_set* sources) {
  struct Collect {std::vector<TaskInterval> intervals;std::string error;} collected;
  auto component=[](isl_basic_set* raw,void* user)->isl_stat {
    auto& into=*static_cast<Collect*>(user);
    auto set=isl_util::Set(isl_set_from_basic_set(raw));
    try {
      if(isl_set_is_empty(set.get())==isl_bool_true)return isl_stat_ok;
      auto lo=isl_util::Val(isl_set_dim_min_val(isl_set_copy(set.get()),0));
      auto hi=isl_util::Val(isl_set_dim_max_val(isl_set_copy(set.get()),0));
      if(!lo || !hi || isl_val_is_int(lo.get())!=isl_bool_true ||
          isl_val_is_int(hi.get())!=isl_bool_true)
        throw std::invalid_argument("unbounded dependency table producer interval");
      auto box=isl_util::Set(isl_set_universe(isl_set_get_space(set.get())));
      box=isl_util::Set(isl_set_lower_bound_val(box.release(),isl_dim_set,0,isl_val_copy(lo.get())));
      box=isl_util::Set(isl_set_upper_bound_val(box.release(),isl_dim_set,0,isl_val_copy(hi.get())));
      if(isl_set_is_equal(set.get(),box.get())==isl_bool_true) {
        auto first=isl_val_get_num_si(lo.get()),last=isl_val_get_num_si(hi.get());
        into.intervals.push_back({std::uint32_t(first),std::uint32_t(last-first+1)});
        return isl_stat_ok;
      }
      // Periodic holes require separate runtime intervals. Only these sparse
      // one-dimensional components need point enumeration.
      return isl_set_foreach_point(set.get(),[](isl_point* raw,void* data)->isl_stat {
        auto& into=*static_cast<Collect*>(data);isl_util::Point point(raw);
        auto value=isl_util::Val(isl_point_get_coordinate_val(point.get(),isl_dim_set,0));
        if(!value || isl_val_is_int(value.get())!=isl_bool_true)return isl_stat_error;
        into.intervals.push_back({std::uint32_t(isl_val_get_num_si(value.get())),1});
        return isl_stat_ok;
      },user);
    }catch(std::exception const& e){into.error=e.what();return isl_stat_error;}
  };
  if(isl_set_foreach_basic_set(sources,component,&collected)!=isl_stat_ok)
    throw std::runtime_error(collected.error.empty()?"dependency producer interval scan failed":collected.error);
  std::sort(collected.intervals.begin(),collected.intervals.end(),
      [](auto const& a,auto const& b){return std::tie(a.first,a.count)<std::tie(b.first,b.count);});
  std::vector<TaskInterval> result;
  for(auto const& interval:collected.intervals) {
    if(!result.empty() && std::uint64_t(result.back().first)+result.back().count>=interval.first) {
      auto past=std::max(std::uint64_t(result.back().first)+result.back().count,
                         std::uint64_t(interval.first)+interval.count);
      result.back().count=std::uint32_t(past-result.back().first);
    }else result.push_back(interval);
  }
  return result;
}
isl_util::Set EncodeRuns(Runs const& runs,isl_space* space,char const* coordinate) {
  std::vector<isl_util::Set> pieces;
  for(std::size_t begin=0;begin<runs.size();) {
    std::size_t end=begin+1;std::uint64_t step=0;
    if(end<runs.size() && runs[end].second==runs[begin].second) {
      step=std::uint64_t(runs[end].first)-runs[begin].first;
      ++end;
      while(end<runs.size() && runs[end].second==runs[begin].second &&
          std::uint64_t(runs[end].first)-runs[end-1].first==step)++end;
    }
    auto text="("+std::to_string(runs[begin].first)+" <= "+coordinate+" < "+
        std::to_string(std::uint64_t(runs[end-1].first)+runs[begin].second);
    if(end>begin+1 && step>runs[begin].second)
      text+=" and ("+std::string(coordinate)+"-"+std::to_string(runs[begin].first)+")%"+
          std::to_string(step)+" < "+std::to_string(runs[begin].second);
    text+=")";begin=end;
    auto piece=isl_util::ReadSet(isl_space_get_ctx(space),
        "{ ["+std::string(coordinate)+"] : "+text+" }");
    pieces.emplace_back(isl_set_reset_space(piece.release(),isl_space_copy(space)));
  }
  // Keep independent interval disjuncts out of the parser's left-folded
  // conjunction/disjunction expansion. Ordinary union preserves overlap.
  while(pieces.size()>1) {
    std::vector<isl_util::Set> next;
    for(std::size_t i=0;i<pieces.size();i+=2)
      if(i+1==pieces.size())next.push_back(std::move(pieces[i]));
      else next.emplace_back(isl_set_union(pieces[i].release(),pieces[i+1].release()));
    pieces=std::move(next);
  }
  return pieces.empty()?isl_util::Set(isl_set_empty(isl_space_copy(space))):std::move(pieces.front());
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
  auto* ctx=SharedIslContext().raw();auto map=isl_util::ReadMap(ctx,relation.ToString());
  if(isl_map_dim(map.get(),isl_dim_param))
    throw std::invalid_argument("dependency table requires every parameter bound");
  auto bounds=isl_util::Map(isl_map_universe(isl_map_get_space(map.get())));
  bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_in,0,0));
  bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_in,0,isl_val_int_from_ui(ctx,consumers-1)));
  bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_out,0,0));
  bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_out,0,isl_val_int_from_ui(ctx,producers-1)));
  if(isl_map_is_subset(map.get(),bounds.get())!=isl_bool_true)
    throw std::invalid_argument("dependency table includes an out-of-range task");
  std::vector<std::vector<TaskInterval>> intervals(result.consumers);
  std::map<std::string,std::vector<TaskInterval>> row_cache;
  for (unsigned task = 0; task < result.consumers; ++task) {
    auto row=isl_util::Map(isl_map_fix_val(isl_map_copy(map.get()),isl_dim_in,0,isl_val_int_from_ui(ctx,task)));
    auto sources=isl_util::Set(isl_map_range(row.release()));
    char* text=isl_set_to_str(sources.get());
    if(!text)throw std::runtime_error("cannot serialize dependency producer row");
    std::string key(text);free(text);
    auto found=row_cache.find(key);
    if(found==row_cache.end()) {
      auto row_intervals=ProducerIntervals(sources.get());
      Runs runs;
      for(auto const& interval:row_intervals)runs.emplace_back(interval.first,interval.count);
      auto space=isl_util::Space(isl_set_get_space(sources.get()));
      auto encoded=EncodeRuns(runs,space.get(),"_tm_p");
      if(isl_set_is_subset(encoded.get(),sources.get())!=isl_bool_true ||
          isl_set_is_subset(sources.get(),encoded.get())!=isl_bool_true)
        throw std::logic_error("dependency table row failed exact containment proof");
      found=row_cache.emplace(std::move(key),std::move(row_intervals)).first;
    }
    intervals[task]=found->second;
    result.stride = std::max(result.stride, static_cast<std::uint32_t>(intervals[task].size()));
  }
  if (std::uint64_t(result.consumers) * result.stride > std::numeric_limits<std::size_t>::max() / sizeof(TaskInterval))
    throw std::invalid_argument("dependency table storage size overflows");
  result.intervals.resize(std::size_t(result.consumers) * result.stride);
  for(unsigned task=0;task<result.consumers;++task) {
    std::copy(intervals[task].begin(), intervals[task].end(),
              result.intervals.begin() + std::size_t(task) * result.stride);
  }
  // Bounds exclude every other consumer and both inclusions were proved for
  // every row above. Thus the original compact relation is also an exact
  // description of these intervals. Rebuilding a disjunction of consumer
  // rows would lose its affine factoring and make the global proof enormous.
  result.encoded_relation=result.linear_relation;
  return result;
}
}  // namespace tilemega::analysis
