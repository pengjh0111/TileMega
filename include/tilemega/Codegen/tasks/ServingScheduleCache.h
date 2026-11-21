// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen::harness {

// Only host-owned indices and counts are serialized. DeviceModel's pointer
// bindings and counted event storage are re-created in the serving process.
inline constexpr std::uint32_t kServingScheduleMagic=0x43534d54;  // TMSC
inline constexpr std::uint32_t kServingScheduleVersion=1;

struct ServingScheduleScope {
  int batch,seq,past,capacity,grid;
  bool l2;
  std::vector<int> const& workers;
  std::vector<int> const& tasks;
  std::vector<std::uint32_t> const& dependencies;
};

template<class Model>
std::vector<std::uint32_t> FreezeServingSchedule(Model const& model,ServingScheduleScope const& scope) {
  auto count=[](std::size_t value) {
    if(value>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("serving schedule exceeds the index range");
    return std::uint32_t(value);
  };
  std::vector<std::uint32_t> words{kServingScheduleMagic,kServingScheduleVersion,
      unsigned(scope.batch),unsigned(scope.seq),unsigned(scope.past),unsigned(scope.capacity),
      unsigned(scope.grid),unsigned(scope.l2),count(model.stages.size()),count(model.schedule.size()),
      count(model.task_waits.size()),model.schedule_max_worker_span,unsigned(model.schedule_has_global_fanin),
      count(model.schedule_raw_polls),count(model.schedule_waiting_tasks),
      count(model.normalization_dummy_lower_bound)};
  for(auto const* table:{&model.stage_order,&model.stage_kappa,&model.event_offsets,
                         &model.event_flags,&model.schedule_offsets})
    words.insert(words.end(),table->begin(),table->end());
  for(int worker:scope.workers)words.push_back(unsigned(worker));
  for(int tasks:scope.tasks)words.push_back(unsigned(tasks));
  for(auto const& task:model.schedule)
    for(auto value:{task.stage,task.logical_task,task.dependency_begin,task.dependency_count,
                    task.wait_begin,task.wait_count,task.flags})
      words.push_back(value);
  for(auto const& wait:model.task_waits) {
    words.push_back(wait.producer);
    words.push_back(wait.group);
  }
  return words;
}

template<class Model>
void RestoreServingSchedule(Model& model,ServingScheduleScope const& scope,
                            std::vector<std::uint32_t> const& words) {
  std::size_t cursor=0;
  auto read=[&] {
    if(cursor>=words.size())throw std::invalid_argument("truncated serving schedule cache");
    return words[cursor++];
  };
  auto expect=[&](std::uint32_t value) {
    if(read()!=value)throw std::invalid_argument("serving schedule cache differs from the bound scope");
  };
  for(auto value:{kServingScheduleMagic,kServingScheduleVersion,unsigned(scope.batch),unsigned(scope.seq),
                  unsigned(scope.past),unsigned(scope.capacity),unsigned(scope.grid),unsigned(scope.l2),
                  unsigned(model.stages.size())})
    expect(value);
  auto tasks=read(),waits=read(),span=read(),fanin=read(),raw=read(),waiting=read(),normalization=read();
  for(auto const* table:{&model.stage_order,&model.stage_kappa,&model.event_offsets,&model.event_flags})
    for(auto value:*table)expect(value);

  auto& offsets=model.schedule_offsets;
  for(auto& value:offsets)value=read();
  if(offsets.front() || offsets.back()!=tasks || !std::is_sorted(offsets.begin(),offsets.end()) ||
     std::uint64_t(cursor)+scope.workers.size()+scope.tasks.size()+7ull*tasks+2ull*waits!=words.size() ||
     span>=offsets.size() || fanin>1 || waits>raw || waiting>tasks || normalization>waits)
    throw std::invalid_argument("invalid serving schedule dimensions");
  for(int worker:scope.workers)expect(unsigned(worker));
  for(int count:scope.tasks)expect(unsigned(count));

  std::vector<std::size_t> stage_first(scope.tasks.size()+1);
  for(std::size_t stage=0;stage<scope.tasks.size();++stage)
    stage_first[stage+1]=stage_first[stage]+
        (scope.l2 && !model.stages[stage].handoff_elided?std::size_t(scope.tasks[stage]):0);
  if(stage_first.back()!=tasks)throw std::invalid_argument("serving schedule task count changed");
  std::vector<bool> seen(tasks);
  std::uint32_t next_wait=0,waiting_count=0;
  for(std::uint32_t i=0;i<tasks;++i) {
    auto stage=read(),task=read(),dep_begin=read(),dep_count=read();
    auto wait_begin=read(),wait_count=read(),flags=read();
    if(stage>=scope.tasks.size() || task>=scope.tasks[stage] || model.stages[stage].handoff_elided ||
       dep_begin!=scope.dependencies[stage] ||
       dep_count!=scope.dependencies[stage+1]-dep_begin || wait_begin!=next_wait ||
       wait_count>waits-next_wait || flags)
      throw std::invalid_argument("invalid serving schedule task row");
    auto index=stage_first[stage]+task;
    if(seen[index])throw std::invalid_argument("duplicate task in serving schedule cache");
    seen[index]=true;
    model.schedule.push_back({stage,task,dep_begin,dep_count,wait_begin,wait_count,flags});
    next_wait+=wait_count;
    waiting_count+=(wait_count!=0);
  }
  if(next_wait!=waits || waiting_count!=waiting || normalization!=waits-waiting)
    throw std::invalid_argument("serving schedule wait counts changed");
  for(std::uint32_t i=0;i<waits;++i) {
    auto stage=read(),group=read();
    if(stage>=model.stages.size())throw std::invalid_argument("invalid cached event producer");
    auto flags=model.event_flags[stage];
    if(group==kWholeStageEventGroup) {
      if(!(flags&kNeedsAggregateEvent))throw std::invalid_argument("invalid cached aggregate event");
    }else {
      auto rows=model.event_offsets[stage+1]-model.event_offsets[stage];
      if(!(flags&kNeedsFineEvents) || group>=rows-unsigned(bool(flags&kNeedsAggregateEvent)))
        throw std::invalid_argument("invalid cached fine event");
    }
    model.task_waits.push_back({stage,group});
  }
  model.schedule_max_worker_span=span;
  model.schedule_has_global_fanin=bool(fanin);
  model.schedule_raw_polls=raw;
  model.schedule_waiting_tasks=waiting;
  model.normalization_dummy_lower_bound=normalization;
}

}  // namespace tilemega::codegen::harness
