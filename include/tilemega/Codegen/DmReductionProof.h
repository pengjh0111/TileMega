// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmReduction.h>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/RuntimeDependencies.h>
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen {
struct DmReductionPlan {
  std::vector<DmReductionStage> stages;
  std::vector<std::uint32_t> offsets;
  std::vector<DmReductionArrival> arrivals;
  std::uint32_t tickets=0,selected=0;
};

// The inverse arrival program is derived from the very same proved RAW
// windows/tables that the ordinary executor waits on. A handoff may omit
// other incoming edges only when every owner task already guarantees their
// entire producer stage is complete; stage order alone is not that proof.
template<class Stage,class Dependency,class Visit>
DmReductionPlan BuildDmReductionProof(std::vector<Stage>& stages,
    std::vector<Dependency> const& dependencies,std::vector<unsigned> const& tasks,
    Visit const& visit,bool pooling,bool moe,unsigned moe_mask=3) {
  using Map=typename Dependency::Map;
  if(stages.size()!=tasks.size())throw std::invalid_argument("DM reduction task counts differ");
  using Row=std::vector<RuntimeWindowBounds>;
  auto merge=[](Row row) {
    std::sort(row.begin(),row.end(),[](auto a,auto b){return a.first<b.first;});
    Row result;
    for(auto range:row)if(range.first<range.past) {
      if(!result.empty() && result.back().past>=range.first)
        result.back().past=std::max(result.back().past,range.past);
      else result.push_back(range);
    }
    return result;
  };
  std::vector<std::vector<unsigned>> incoming(stages.size());
  for(unsigned e=0;e<dependencies.size();++e) {
    auto const& dep=dependencies[e];
    if(dep.consumer>=stages.size() || dep.producer>=dep.consumer)
      throw std::invalid_argument("DM reduction requires an acyclic stage order");
    incoming[dep.consumer].push_back(e);
  }
  auto read=[&](unsigned consumer,unsigned producer,unsigned task) {
    Row row;
    for(auto e:incoming[consumer]) {
      auto const& dep=dependencies[e];if(dep.producer!=producer)continue;
      if(dep.map==Map::kCounted || dep.map==Map::kPhase)
        throw std::invalid_argument("DM reduction requires static completion dependencies");
      if(!visit(dep,task,tasks[producer],
          [&](RuntimeWindowBounds r){row.push_back(r);}))
        throw std::invalid_argument("invalid DM reduction dependency row");
    }
    return merge(std::move(row));
  };
  auto full=[&](Row const& row,unsigned count) {
    return row.size()==1 && row[0].first==0 && std::uint64_t(row[0].past)==count;
  };
  std::vector<std::set<unsigned>> after_task(stages.size()),after_stage(stages.size());
  for(unsigned c=0;c<stages.size();++c) {
    after_stage[c].insert(c);std::set<unsigned> parents;
    for(auto e:incoming[c])parents.insert(dependencies[e].producer);
    for(auto p:parents) {
      bool dynamic=false;
      for(auto e:incoming[c])if(dependencies[e].producer==p &&
          (dependencies[e].map==Map::kCounted ||
           dependencies[e].map==Map::kPhase))dynamic=true;
      if(dynamic || !tasks[c] || !tasks[p])continue;
      bool every_nonempty=true,every_full=true;Row combined;
      for(unsigned t=0;t<tasks[c];++t) {
        auto row=read(c,p,t);every_nonempty&=!row.empty();every_full&=full(row,tasks[p]);
        combined.insert(combined.end(),row.begin(),row.end());
      }
      if(every_nonempty)after_task[c].insert(after_task[p].begin(),after_task[p].end());
      if(every_full)after_task[c].insert(after_stage[p].begin(),after_stage[p].end());
      if(full(merge(std::move(combined)),tasks[p]))
        after_stage[c].insert(after_stage[p].begin(),after_stage[p].end());
    }
    after_stage[c].insert(after_task[c].begin(),after_task[c].end());
  }
  DmReductionPlan result;result.stages.resize(stages.size());
  std::vector<std::vector<DmReductionArrival>> arrivals;
  for(unsigned s=0;s<stages.size();++s) {
    if(std::uint64_t(arrivals.size())+tasks[s]>=std::numeric_limits<unsigned>::max())
      throw std::overflow_error("DM reduction task program exceeds 32-bit storage");
    result.stages[s].task_offset=arrivals.size();arrivals.resize(arrivals.size()+tasks[s]);
  }
  for(unsigned c=0;c<stages.size();++c) {
    auto const& reducer=stages[c];
    bool pool=pooling && reducer.kind==TaskKind::kGlobalPoolReduce;
    bool select=moe && (moe_mask&1) && reducer.kind==TaskKind::kMoETopK;
    bool combine=moe && (moe_mask&2) && reducer.kind==TaskKind::kMoECombine;
    if((!pool && !select && !combine) || reducer.handoff_elided || !tasks[c])continue;
    for(auto edge:incoming[c]) {
      auto const& dep=dependencies[edge];unsigned p=dep.producer;
      auto const& producer=stages[p];
      if(result.stages[p].target!=kDmNoIndex)continue;
      bool candidate=pool?(IsGemmStage(producer.kind) || producer.kind==TaskKind::kGemmCombine ||
                           producer.kind==TaskKind::kDepthwiseConv):
          combine?(IsGemmStage(producer.kind) || producer.kind==TaskKind::kGemmCombine):
          (reducer.moe.step==DmMoeStep::kSelect || reducer.moe.step==DmMoeStep::kSelectAndDispatch)?
              (IsGemmStage(producer.kind) || producer.kind==TaskKind::kGemmCombine):
              producer.kind==TaskKind::kMoETopK;
      if(!candidate || !tasks[p])continue;
      bool ready=true;
      for(auto other:incoming[c])if(dependencies[other].producer!=p &&
          !after_task[p].count(dependencies[other].producer))ready=false;
      if(!ready)continue;
      bool counted=dep.map==Map::kCounted;
      if(counted && (!combine || std::count_if(incoming[c].begin(),incoming[c].end(),
          [&](unsigned e){return dependencies[e].producer==p;})!=1))continue;
      if(!counted) {
        bool dynamic=false;
        for(auto e:incoming[c])if(dependencies[e].producer==p &&
            (dependencies[e].map==Map::kCounted ||
             dependencies[e].map==Map::kPhase))dynamic=true;
        if(dynamic)continue;
      }
      std::vector<Row> rows;std::vector<unsigned> expected;
      if(!counted) {
        bool nonempty=true;
        for(unsigned t=0;t<tasks[c];++t) {
          auto row=read(c,p,t);unsigned count=0;
          for(auto r:row)count+=r.past-r.first;
          nonempty&=count>0;rows.push_back(std::move(row));expected.push_back(count);
        }
        if(!nonempty)continue;
      }else if(dep.table_rows!=tasks[c])
        throw std::invalid_argument("counted DM reduction target extent differs");
      if(std::uint64_t(result.tickets)+tasks[c]>std::numeric_limits<unsigned>::max())
        throw std::overflow_error("DM reduction ticket bank overflows");
      auto& program=result.stages[p];program.target=c;program.ticket_offset=result.tickets;
      program.counted_edge=counted?edge:kDmNoIndex;result.tickets+=tasks[c];++result.selected;
      if(!counted)for(unsigned t=0;t<rows.size();++t)
        for(auto r:rows[t])for(int source=r.first;source<r.past;++source)
          arrivals[program.task_offset+source].push_back({t,expected[t]});
      stages[c].handoff_elided=true;stages[p].dm_reduce_stage=c;break;
    }
  }
  result.offsets.push_back(0);
  for(auto const& row:arrivals) {
    if(std::uint64_t(result.arrivals.size())+row.size()>std::numeric_limits<unsigned>::max())
      throw std::overflow_error("DM reduction arrival program overflows");
    result.arrivals.insert(result.arrivals.end(),row.begin(),row.end());
    result.offsets.push_back(result.arrivals.size());
  }
  for(unsigned source=0;source<stages.size();++source) {
    unsigned depth=0,s=source;
    while(result.stages[s].target!=kDmNoIndex) {
      s=result.stages[s].target;
      if(++depth>6)throw std::invalid_argument("DM reduction chain exceeds device depth");
    }
  }
  return result;
}
} // namespace tilemega::codegen
