// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Codegen/DmReductionProof.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <set>
#include <tuple>

namespace tilemega::solver {
namespace {
struct ProofStage {
  codegen::TaskKind kind=codegen::TaskKind::kAdd;
  codegen::DmMoeStage moe{};
  bool handoff_elided=false;
  unsigned dm_reduce_stage=codegen::kDmNoIndex;
};
struct ProofDependency {
  enum class Map {kTable,kCounted,kPhase};
  unsigned producer=0,consumer=0;
  Map map=Map::kTable;
  unsigned table_rows=0,table_stride=0,table_offset=0;
};
std::vector<codegen::DependencyRecord> BoundDependencies(SymbolicProblem const& problem) {
  using R=analysis::CouplingRelation;
  auto theta=problem.model.MetricBindings();
  auto records=problem.data_edges.empty()
      ? RebuildBoundRuntimeDependencies(problem.projection)
      : std::vector<codegen::DependencyRecord>{};
  for(auto const& data:problem.data_edges) {
    codegen::DependencyRecord edge{unsigned(data.producer),unsigned(data.consumer),{}};
    auto counted=std::find_if(problem.projection.runtime_counted.begin(),
        problem.projection.runtime_counted.end(),[&](auto const& item) {
          return item.producer==data.producer && item.consumer==data.consumer;
        });
    if(counted!=problem.projection.runtime_counted.end())edge.counted=counted->contract;
    else edge.table=analysis::BuildDependencyTableLinear(data.relation.Reverse().BindParams(theta),
        problem.counts.at(data.producer),problem.counts.at(data.consumer));
    records.push_back(std::move(edge));
  }
  std::set<std::pair<unsigned,unsigned>> pairs;
  for(auto const& item:problem.projection.runtime_windows)pairs.emplace(item.producer,item.consumer);
  for(auto const& item:problem.projection.runtime_tables)pairs.emplace(item.producer,item.consumer);
  for(auto const& [p,c]:pairs)if(std::none_of(records.begin(),records.end(),[&](auto const& edge) {
    return edge.producer==p && edge.consumer==c;
  })) {
    codegen::DependencyRecord edge{p,c,{}};
    edge.table=analysis::BuildDependencyTableLinear(R::FromIslText("{ [c] -> [p] : false }"),
        problem.counts.at(p),problem.counts.at(c));records.push_back(std::move(edge));
  }
  for(auto& edge:records)if(!edge.counted) {
    unsigned p=edge.producer,c=edge.consumer;
    auto relation=edge.table->linear_relation;
    auto bounds="0<=c<"+std::to_string(problem.counts.at(c))+" and 0<=p<"+
        std::to_string(problem.counts.at(p));
    for(auto const& bound:BindRuntimeWindows(problem.projection,p,c,theta)) {
      if(bound.table && !problem.projection.options.force_all_dependencies) {
        relation=relation.Union(bound.table->linear_relation);continue;
      }
      std::string window=bounds;
      if(bound.window.narrowed && !problem.projection.options.force_all_dependencies) {
        auto const& w=bound.window;
        if(w.div<=0 || w.count<0)throw std::invalid_argument("invalid handoff runtime window");
        auto begin=std::to_string(w.scale)+"*floord(c,"+std::to_string(w.div)+")+("+
            std::to_string(bound.offset)+")";
        window+=" and "+begin+"<=p<"+begin+"+"+std::to_string(w.count);
      }
      relation=relation.Union(R::FromIslText("{ [c] -> [p] : "+window+" }"));
    }
    edge.table=analysis::BuildDependencyTableLinear(relation,problem.counts.at(p),problem.counts.at(c));
  }
  std::stable_sort(records.begin(),records.end(),[](auto const& a,auto const& b) {
    return std::tie(a.consumer,a.producer)<std::tie(b.consumer,b.producer);
  });
  return records;
}
}
unsigned ConfigureDmReductionFlow(PreparedFlow& prepared,SymbolicProblem const& problem,
    unsigned mask) {
  using namespace codegen;
  if(!problem.model.dm || mask>7 || prepared.flow.spaces.size()!=problem.counts.size() ||
      prepared.prices.size()!=problem.counts.size())
    throw std::invalid_argument("invalid DM handoff flow contract");
  std::vector<ProofStage> stages;std::vector<unsigned> counts;
  for(unsigned s=0;s<problem.counts.size();++s) {
    auto& space=prepared.flow.spaces[s];auto const& base=prepared.prices[s];
    if(space.pieces.size()!=base.pieces.size())
      throw std::invalid_argument("DM handoff base piece coverage changed");
    for(unsigned i=0;i<space.pieces.size();++i)space.pieces[i].parts=base.pieces[i].parts;
    space.handoff_reducer=false;
    space.rank_ns=space.count?base.total_isolated_ns/space.count:0;
    auto const& projected=problem.projection.stages.at(s);
    auto const& model=problem.model.stages.at(projected.logical_stage);
    ProofStage stage;stage.moe=model.moe;
    if(projected.combine)stage.kind=TaskKind::kGemmCombine;
    else switch(model.kind) {
      case StageKind::kGemm:stage.kind=TaskKind::kGemm;break;
      case StageKind::kDwPwFused:stage.kind=TaskKind::kDwPwFused;break;
      case StageKind::kDepthwiseConv:stage.kind=TaskKind::kDepthwiseConv;break;
      case StageKind::kGlobalPoolReduce:stage.kind=TaskKind::kGlobalPoolReduce;break;
      case StageKind::kMoETopK:stage.kind=TaskKind::kMoETopK;break;
      case StageKind::kMoECombine:stage.kind=TaskKind::kMoECombine;break;
      default:break;
    }
    stage.handoff_elided=space.fused_reducer;
    stages.push_back(stage);counts.push_back(problem.counts[s]);
  }
  if(!mask)return 0;
  std::vector<ProofDependency> dependencies;
  std::vector<RuntimeDependencyInterval> intervals;
  for(auto const& edge:BoundDependencies(problem)) {
    ProofDependency dep;dep.producer=edge.producer;dep.consumer=edge.consumer;
    auto table=edge.table;
    if(edge.counted) {
      dep.map=ProofDependency::Map::kCounted;
      table=analysis::BuildDependencyTableLinear(edge.counted->conservative_relation,
          edge.counted->producers,edge.counted->contributions.expected.size());
    }
    if(!table)throw std::invalid_argument("DM handoff proof requires an exact bound table");
    if(std::uint64_t(intervals.size())+table->intervals.size()>
        std::numeric_limits<unsigned>::max())
      throw std::overflow_error("DM handoff proof interval storage overflows");
    dep.table_rows=table->consumers;dep.table_stride=table->stride;
    dep.table_offset=intervals.size();
    for(auto const& interval:table->intervals)intervals.push_back({interval.first,interval.count});
    if(std::any_of(problem.data_edges.begin(),problem.data_edges.end(),[&](auto const& data) {
      return data.producer==int(dep.producer) && data.consumer==int(dep.consumer) && data.first_phase;
    }))dep.map=ProofDependency::Map::kPhase;
    dependencies.push_back(dep);
  }
  auto visit=[&](ProofDependency const& dep,unsigned c,unsigned p,auto const& callback) {
    return VisitDependencyTable({intervals.empty()?nullptr:intervals.data()+dep.table_offset,dep.table_rows,dep.table_stride},c,p,callback);
  };
  auto program=BuildDmReductionProof(stages,dependencies,counts,visit,mask&1,mask&6,(mask>>1)&3);
  for(unsigned p=0;p<stages.size();++p) {
    auto const& instruction=program.stages[p];
    if(instruction.target==kDmNoIndex)continue;
    prepared.flow.spaces[instruction.target].handoff_reducer=true;
    std::vector<unsigned> arrivals(counts[p]);
    if(instruction.counted_edge!=kDmNoIndex) {
      // Routing determines the actual targets. The proved I2 envelope bounds
      // target atomics; no router sample is treated as a legality proof.
      auto const& dep=dependencies[instruction.counted_edge];
      for(unsigned c=0;c<dep.table_rows;++c)
        if(!visit(dep,c,counts[p],[&](auto row) {
          for(int t=row.first;t<row.past;++t)++arrivals[t];
        }))throw std::invalid_argument("invalid counted handoff envelope");
    } else for(unsigned t=0;t<counts[p];++t) {
      unsigned index=instruction.task_offset+t;
      arrivals[t]=program.offsets[index+1]-program.offsets[index];
    }
    auto& space=prepared.flow.spaces[p];
    if(space.fused_reducer)space.handoff_reducer=true;
    std::vector<unsigned> maximum(space.pieces.size());
    for(unsigned t=0;t<counts[p];++t)
      maximum.at(space.piece_of_task.at(t))=std::max(maximum.at(space.piece_of_task.at(t)),arrivals[t]);
    // Publication latency is an inferred atomic/fence proxy, not a new fit.
    // Keep every reducer's body, memory and barriers in its own price.
    for(unsigned i=0;i<space.pieces.size();++i) {
      double arrival=maximum[i]*prepared.flow.publication_ns;
      space.pieces[i].parts.fixed_ns+=arrival;
      if(space.count)space.rank_ns+=space.pieces[i].count*arrival/space.count;
    }
  }
  return program.selected;
}
} // namespace tilemega::solver
