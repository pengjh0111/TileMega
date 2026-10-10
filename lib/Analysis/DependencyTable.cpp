// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include "IslUtil.h"
#include <isl/ilp.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <optional>
#include <isl/options.h>

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
template<class Proof>
bool MemoTableProof(DependencyTable const& table,Proof proof) {
  auto dimensions=std::to_string(table.producers)+","+std::to_string(table.consumers)+","+
      std::to_string(table.stride);
  std::string rows;
  for(auto interval:table.intervals)
    rows+=std::to_string(interval.first)+","+std::to_string(interval.count)+";";
  auto relation=table.linear_relation.ToString();
  return MemoExact({"dependency-table-proof-v1",dimensions,relation,rows},proof);
}
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
      // Exact extrema already prove set is inside box. A direct reverse
      // inclusion avoids equality's expensive normalization of projected divs.
      auto covered=isl_set_is_subset(box.get(),set.get());
      if(covered==isl_bool_error)
        throw std::runtime_error("dependency interval containment proof failed");
      if(covered==isl_bool_true) {
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
std::optional<std::vector<std::vector<TaskInterval>>> SymbolicProducerIntervals(
    std::string const& relation,std::uint32_t producers,std::uint32_t consumers) {
  // Prove each component's interval fibers once, then enumerate only their
  // endpoints. A failed/quota-limited optimization retains the exact row path.
  IslContext proof;
  auto* ctx=proof.raw();isl_options_set_on_error(ctx,ISL_ON_ERROR_CONTINUE);
  isl_ctx_set_max_operations(ctx,1000000);
  auto map=isl_util::Map(isl_map_read_from_str(ctx,relation.c_str()));
  struct State {
    std::vector<std::vector<TaskInterval>> rows;
    std::uint32_t producers;
    bool unavailable=false;
    std::string error;
  } state{std::vector<std::vector<TaskInterval>>(consumers),producers};
  auto component=[](isl_basic_map* raw,void* user)->isl_stat {
    auto& state=*static_cast<State*>(user);
    auto part=isl_util::Map(isl_map_from_basic_map(raw));
    auto* ctx=isl_map_get_ctx(part.get());
    isl_ctx_reset_operations(ctx);isl_ctx_set_max_operations(ctx,1000000);
    auto lo=isl_util::Map(isl_map_lexmin(isl_map_copy(part.get())));
    auto hi=isl_util::Map(isl_map_lexmax(isl_map_copy(part.get())));
    if(!lo || !hi)return isl_stat_error;
    auto space=isl_util::Space(isl_space_range(isl_map_get_space(part.get())));
    auto lower=isl_util::Map(isl_map_apply_range(isl_map_copy(lo.get()),
        isl_map_lex_le(isl_space_copy(space.get()))));
    auto upper=isl_util::Map(isl_map_apply_range(isl_map_copy(hi.get()),
        isl_map_lex_ge(isl_space_copy(space.get()))));
    auto band=isl_util::Map(isl_map_intersect(lower.release(),upper.release()));
    if(!band)return isl_stat_error;
    auto forward=isl_map_is_subset(part.get(),band.get());
    auto reverse=isl_map_is_subset(band.get(),part.get());
    if(forward==isl_bool_error || reverse==isl_bool_error)return isl_stat_error;
    if(forward!=isl_bool_true || reverse!=isl_bool_true) {
      state.unavailable=true;return isl_stat_error;
    }
    // Bounds have already been checked. Endpoint enumeration is finite and
    // visits at most two points per consumer, rather than every RAW pair.
    isl_ctx_set_max_operations(ctx,0);
    auto endpoints=[&](isl_map* source,std::map<std::uint32_t,std::uint32_t>& into) {
      auto wrapped=isl_util::Set(isl_map_wrap(isl_map_copy(source)));
      struct Sink {std::map<std::uint32_t,std::uint32_t>* values;State* state;};
      Sink sink{&into,&state};
      return isl_set_foreach_point(wrapped.get(),[](isl_point* raw,void* user)->isl_stat {
        auto& sink=*static_cast<Sink*>(user);isl_util::Point point(raw);
        auto c=isl_util::Val(isl_point_get_coordinate_val(point.get(),isl_dim_set,0));
        auto p=isl_util::Val(isl_point_get_coordinate_val(point.get(),isl_dim_set,1));
        if(!c || !p || isl_val_is_int(c.get())!=isl_bool_true || isl_val_is_int(p.get())!=isl_bool_true)
          return isl_stat_error;
        auto consumer=isl_val_get_num_si(c.get()),producer=isl_val_get_num_si(p.get());
        if(consumer<0 || std::uint64_t(consumer)>=sink.state->rows.size() ||
           producer<0 || std::uint64_t(producer)>=sink.state->producers ||
           !sink.values->emplace(consumer,producer).second) {
          sink.state->error="invalid symbolic dependency interval endpoint";return isl_stat_error;
        }
        return isl_stat_ok;
      },&sink);
    };
    std::map<std::uint32_t,std::uint32_t> first,last;
    if(endpoints(lo.get(),first)!=isl_stat_ok || endpoints(hi.get(),last)!=isl_stat_ok)
      return isl_stat_error;
    if(first.size()!=last.size())return isl_stat_error;
    for(auto const& [c,p]:first) {
      auto end=last.find(c);
      if(end==last.end() || end->second<p)return isl_stat_error;
      state.rows[c].push_back({p,end->second-p+1});
    }
    return isl_stat_ok;
  };
  auto status=map?isl_map_foreach_basic_map(map.get(),component,&state):isl_stat_error;
  if(status!=isl_stat_ok) {
    if(state.unavailable || isl_ctx_last_error(ctx)==isl_error_quota)return std::nullopt;
    throw std::runtime_error(state.error.empty()?"symbolic dependency interval proof failed":state.error);
  }
  for(auto& row:state.rows) {
    std::sort(row.begin(),row.end(),[](auto const& a,auto const& b) {
      return std::tie(a.first,a.count)<std::tie(b.first,b.count);
    });
    std::vector<TaskInterval> merged;
    for(auto interval:row) {
      if(!merged.empty() && std::uint64_t(merged.back().first)+merged.back().count>=interval.first) {
        auto past=std::max(std::uint64_t(merged.back().first)+merged.back().count,
            std::uint64_t(interval.first)+interval.count);
        merged.back().count=std::uint32_t(past-merged.back().first);
      }else merged.push_back(interval);
    }
    row=std::move(merged);
  }
  return std::move(state.rows);
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
void ValidateLinearTaskBounds(CouplingRelation const& relation,
    std::uint32_t producers,std::uint32_t consumers) {
  IslReferenceAudit audit(__func__);
  auto source=relation.ToString(),p=std::to_string(producers),c=std::to_string(consumers);
  (void)MemoExact({"linear-task-domain-bounds-v1",source,p,c},[&] {
    if(!producers || !consumers || relation.DomainDimNames().size()!=1 || relation.RangeDimNames().size()!=1)
      throw std::invalid_argument("invalid linear task bounds");
    auto* ctx=SharedIslContext().raw();auto map=isl_util::ReadMap(ctx,source);
    if(isl_map_dim(map.get(),isl_dim_param))
      throw std::invalid_argument("linear task bounds require bound parameters");
    auto bounds=isl_util::Map(isl_map_universe(isl_map_get_space(map.get())));
    bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_in,0,0));
    bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_in,0,isl_val_int_from_ui(ctx,consumers-1)));
    bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_out,0,0));
    bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_out,0,isl_val_int_from_ui(ctx,producers-1)));
    if(isl_map_is_subset(map.get(),bounds.get())!=isl_bool_true)
      throw std::invalid_argument("linear task relation exceeds its task bounds");
    return true;
  });
}
void ValidateDependencyTableLinear(DependencyTable const& table) {
  IslReferenceAudit audit(__func__);
  (void)MemoTableProof(table,[&] {
  if(!table.producers || !table.consumers || table.linear_relation.DomainDimNames().size()!=1 ||
      table.linear_relation.RangeDimNames().size()!=1 ||
      std::uint64_t(table.consumers)*table.stride!=table.intervals.size())
    throw std::invalid_argument("invalid retained dependency table dimensions");
  auto* ctx=SharedIslContext().raw();auto map=isl_util::ReadMap(ctx,table.linear_relation.ToString());
  if(isl_map_dim(map.get(),isl_dim_param))
    throw std::invalid_argument("retained dependency table has unbound parameters");
  auto bounds=isl_util::Map(isl_map_universe(isl_map_get_space(map.get())));
  bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_in,0,0));
  bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_in,0,isl_val_int_from_ui(ctx,table.consumers-1)));
  bounds=isl_util::Map(isl_map_lower_bound_si(bounds.release(),isl_dim_out,0,0));
  bounds=isl_util::Map(isl_map_upper_bound_val(bounds.release(),isl_dim_out,0,isl_val_int_from_ui(ctx,table.producers-1)));
  if(isl_map_is_subset(map.get(),bounds.get())!=isl_bool_true)
    throw std::invalid_argument("retained dependency table contains out-of-range tasks");
  unsigned maximum=0;
  for(unsigned c=0;c<table.consumers;++c) {
    Runs runs;bool padding=false;std::uint64_t past=0;
    for(unsigned i=0;i<table.stride;++i) {
      auto interval=table.intervals[std::size_t(c)*table.stride+i];
      if(!interval.count) {
        if(interval.first)throw std::invalid_argument("nonzero retained dependency padding");
        padding=true;continue;
      }
      if(padding || (i && interval.first<=past) ||
          std::uint64_t(interval.first)+interval.count>table.producers)
        throw std::invalid_argument("retained dependency intervals are not canonical");
      runs.emplace_back(interval.first,interval.count);past=std::uint64_t(interval.first)+interval.count;
    }
    maximum=std::max(maximum,unsigned(runs.size()));
    auto source=isl_util::Map(isl_map_fix_val(isl_map_copy(map.get()),isl_dim_in,0,isl_val_int_from_ui(ctx,c)));
    auto space=isl_util::Space(isl_space_range(isl_map_get_space(map.get())));
    auto encoded_set=EncodeRuns(runs,space.get(),"_tm_p");
    auto encoded=isl_util::Map(isl_map_from_range(encoded_set.release()));
    encoded=isl_util::Map(isl_map_add_dims(encoded.release(),isl_dim_in,1));
    if(isl_map_has_tuple_id(map.get(),isl_dim_in)==isl_bool_true)
      encoded=isl_util::Map(isl_map_set_tuple_id(encoded.release(),isl_dim_in,
          isl_map_get_tuple_id(map.get(),isl_dim_in)));
    encoded=isl_util::Map(isl_map_fix_val(encoded.release(),isl_dim_in,0,isl_val_int_from_ui(ctx,c)));
    // Keeping the fixed consumer coordinate avoids projecting a complicated
    // source row merely to recover endpoints that are already serialized.
    if(isl_map_is_subset(source.get(),encoded.get())!=isl_bool_true ||
        isl_map_is_subset(encoded.get(),source.get())!=isl_bool_true)
      throw std::invalid_argument("retained dependency intervals differ from their relation");
  }
  if(maximum!=table.stride)throw std::invalid_argument("retained dependency stride is not the maximum row size");
  return true;
  });
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
  auto symbolic=SymbolicProducerIntervals(relation.ToString(),producers,consumers);
  if(symbolic)intervals=std::move(*symbolic);
  else for (unsigned task = 0; task < result.consumers; ++task) {
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
  }
  for(auto const& row:intervals)
    result.stride=std::max(result.stride,static_cast<std::uint32_t>(row.size()));
  if (std::uint64_t(result.consumers) * result.stride > std::numeric_limits<std::size_t>::max() / sizeof(TaskInterval))
    throw std::invalid_argument("dependency table storage size overflows");
  result.intervals.resize(std::size_t(result.consumers) * result.stride);
  for(unsigned task=0;task<result.consumers;++task) {
    std::copy(intervals[task].begin(), intervals[task].end(),
              result.intervals.begin() + std::size_t(task) * result.stride);
  }
  // Bounds exclude every other consumer and both inclusions were proved for
  // each symbolic component or each row. The compact relation is an exact
  // description of these intervals. Rebuilding a disjunction of consumer
  // rows would lose its affine factoring and make the global proof enormous.
  result.encoded_relation=result.linear_relation;
  // Construction proved bounds and both inclusions for the canonical rows.
  // Only this complete immutable value may reuse that proof in later readers.
  (void)MemoTableProof(result,[]{return true;});
  return result;
}
}  // namespace tilemega::analysis
