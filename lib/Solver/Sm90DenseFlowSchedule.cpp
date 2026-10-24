// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseFlowSchedule.h>
#include <tilemega/Solver/Sm90DenseFlowEvents.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
Sm90DenseScheduleResult ScheduleSm90DenseFlowScope(Sm90DenseFlowEvents const& source,
    Sm90DenseScheduleMapping const& map,Sm90DenseSchedulePolicies const& policies) {
  using Phase=Sm90DenseSchedulePhase;using Id=std::size_t;
  auto count=map.operations.size();Sm90DenseScheduleResult out;
  out.starts.resize(count);out.completions.resize(count);out.rmw.resize(count);out.continuations.resize(count);out.owners.resize(map.groups.size());out.progress.resize(count);
  out.pending=map.unpriced;
  if(!count || map.finish>=count || !std::isfinite(map.origin))out.pending.push_back("unsupported/empty prototype scope");
  if(!out.pending.empty())return out;
  std::vector<Sm90DenseScheduleService> services(count);
  std::map<std::string,std::vector<double>> lanes;
  std::set<Id> mapped;
  for(Id i=0;i<count;++i) {
    auto const& op=map.operations[i];
    if(op.source_event>=source.events.size())throw std::invalid_argument("mapped source event out of bounds");
    mapped.insert(op.source_event);
    if(op.structural_join) {
      if(!op.service.empty() || !op.resource.empty())throw std::invalid_argument("structural join cannot charge service");
    }else {
      auto service=policies.services.find(op.service);auto resource=policies.resources.find(op.resource);
      if(service==policies.services.end() || !service->second) {out.pending.push_back(op.name+": service unpriced");continue;}
      if(resource==policies.resources.end() || !resource->second) {out.pending.push_back(op.name+": resource policy missing");continue;}
      auto const& s=*service->second;auto const& r=*resource->second;
      if(!std::isfinite(s.completion_delay) || s.completion_delay<0 || !std::isfinite(s.admission_occupancy) || s.admission_occupancy<0 ||
          (s.rmw_offset && (!std::isfinite(*s.rmw_offset) || *s.rmw_offset<0 || *s.rmw_offset>s.completion_delay)) || r.capacity<1)
        throw std::invalid_argument("invalid nonnegative service/capacity parameters");
      if(r.arbitration!="fifo_explicit_ties" && r.arbitration!="fifo_source_order_model" && r.arbitration!="fifo_reverse_source_order_model") {out.pending.push_back(op.name+": arbitration policy unimplemented");continue;}
      auto initial=map.initial_resource_available.find(op.resource);
      if(initial==map.initial_resource_available.end() || initial->second.size()!=std::size_t(r.capacity)) {out.pending.push_back(op.name+": initial resource state missing");continue;}
      for(auto time:initial->second)if(!std::isfinite(time))throw std::invalid_argument("invalid carried resource time");
      for(auto offset:s.progress_offsets)if(offset && (!std::isfinite(*offset) || *offset<0))throw std::invalid_argument("invalid progress offset");
      services[i]=s;lanes.try_emplace(op.resource,initial->second);
    }
  }
  for(auto event:map.source_scope)if(!mapped.count(event))out.pending.push_back("source scope has unmapped work");
  if(!out.pending.empty())return out;
  std::vector<int> group(count,-1),member(count,-1),merge_member(count,-1),merge_group(count,-1),returned_member(count,-1),returned_group(count,-1);
  for(Id g=0;g<map.groups.size();++g) {
    if(map.groups[g].empty())throw std::invalid_argument("empty RMW group");
    for(Id m=0;m<map.groups[g].size();++m) {
      auto const& x=map.groups[g][m];
      if(x.rmw>=count || x.publication_return>=count || x.merge>=count || group[x.rmw]>=0 || merge_group[x.merge]>=0 ||
          returned_group[x.publication_return]>=0 || !services[x.rmw].rmw_offset)
        throw std::invalid_argument("invalid RMW/member/return mapping");
      group[x.rmw]=g;member[x.rmw]=m;merge_group[x.merge]=g;merge_member[x.merge]=m;
      returned_group[x.publication_return]=g;returned_member[x.publication_return]=m;
    }
  }
  for(Id i=0;i<count;++i)if(services[i].rmw_offset && group[i]<0)throw std::invalid_argument("unbound RMW linearization");
  std::vector<std::array<std::vector<Id>,12>> users(count);
  std::vector<unsigned> remaining(count);
  for(Id i=0;i<count;++i)for(auto const& gate:map.operations[i].after) {
    if(gate.operation>=count || int(gate.phase)<0 || int(gate.phase)>11 ||
        ((gate.phase==Phase::kRmw || gate.phase==Phase::kContinue) && group[gate.operation]<0))throw std::invalid_argument("invalid service-start gate");
    if(int(gate.phase)>=4 && !services[gate.operation].progress_offsets[int(gate.phase)-4])out.pending.push_back("progress phase unpriced");
    users[gate.operation][int(gate.phase)].push_back(i);++remaining[i];
  }
  if(!out.pending.empty())return out;
  unsigned pending_progress=0;
  // Status 0 waits, 1 is ready, 2 runs, 3 completed, 4 is an inactive merge.
  std::vector<int> status(count);std::vector<double> ready(count,map.origin);
  std::vector<Id> newly_ready;Id completed=0;
  for(Id i=0;i<count;++i)if(!remaining[i])status[i]=1;
  std::vector<unsigned> arrivals(map.groups.size());std::vector<std::optional<bool>> last(count);
  struct Event {double time;int kind;Id op;Id resource=std::numeric_limits<Id>::max();};
  auto later=[](Event const& a,Event const& b) {return a.time>b.time;};
  std::priority_queue<Event,std::vector<Event>,decltype(later)> events(later);
  auto emit=[&](Id op,Phase phase,double now) {
    auto* value=int(phase)>=4?&out.progress[op][int(phase)-4]:phase==Phase::kStart?&out.starts[op]:
        phase==Phase::kDone?&out.completions[op]:phase==Phase::kRmw?&out.rmw[op]:&out.continuations[op];
    if(*value)throw std::invalid_argument("duplicate phase completion");
    *value=now;
    for(auto dependent:users[op][int(phase)])if(!--remaining[dependent] && status[dependent]!=4) {
      status[dependent]=1;ready[dependent]=std::max(now,map.operations[dependent].not_before.value_or(map.origin));
      if(ready[dependent]>now)events.push({ready[dependent],2,dependent});
      else newly_ready.push_back(dependent);
    }
  };
  auto continuation=[&](Id g,Id m,double now) {
    auto const& x=map.groups[g][m];
    if(last[x.rmw] && out.completions[x.publication_return] &&
        (!*last[x.rmw] || out.completions[x.merge]) && !out.continuations[x.rmw])emit(x.rmw,Phase::kContinue,now);
  };
  for(Id i=0;i<count;++i) {
    if(map.operations[i].not_before && !std::isfinite(*map.operations[i].not_before))throw std::invalid_argument("invalid source entry time");
    if(status[i]==1) {ready[i]=std::max(map.origin,map.operations[i].not_before.value_or(map.origin));
      if(ready[i]>map.origin)events.push({ready[i],2,i});else newly_ready.push_back(i);}
  }
  auto priority=[&](Id i)->std::optional<int> {
    if(map.operations[i].tie_priority)return map.operations[i].tie_priority;
    auto resource=policies.resources.find(map.operations[i].resource);
    if(resource!=policies.resources.end() && resource->second) {
      if(resource->second->arbitration=="fifo_source_order_model")return int(i);
      if(resource->second->arbitration=="fifo_reverse_source_order_model")return -int(i);
    }
    return {};
  };
  std::vector<std::optional<int>> priorities(count);
  for(Id i=0;i<count;++i)priorities[i]=priority(i);
  struct ReadyGroup {
    std::set<std::pair<int,Id>> order;
    std::map<int,unsigned> multiplicity;
    unsigned missing=0,duplicate_keys=0;
  };
  struct ReadyQueue {
    std::string name;
    std::map<double,ReadyGroup> groups;
    Id count=0,ambiguous_groups=0;
  };
  std::map<std::string,Id> resource_ids;
  for(auto const& op:map.operations)resource_ids.emplace(op.resource,0);
  std::vector<ReadyQueue> queues;
  for(auto& entry:resource_ids) {entry.second=queues.size();queues.push_back({entry.first});}
  std::vector<Id> resource_id(count);
  for(Id i=0;i<count;++i)resource_id[i]=resource_ids.at(map.operations[i].resource);
  std::vector<bool> queued(count);
  std::set<Id> active;
  auto enqueue=[&](Id i) {
    if(queued[i] || status[i]!=1)return;
    if(merge_group[i]>=0) {
      auto rmw=map.groups[merge_group[i]][merge_member[i]].rmw;
      if(!last[rmw] || !*last[rmw])return;
    }
    auto& queue=queues[resource_id[i]];auto& group=queue.groups[ready[i]];
    bool ambiguous=group.duplicate_keys;
    auto key=priorities[i].value_or(0);
    if(group.multiplicity[key]++==1)++group.duplicate_keys;
    if(!priorities[i])++group.missing;
    group.order.emplace(key,i);++queue.count;queued[i]=true;
    if(!ambiguous && group.duplicate_keys)++queue.ambiguous_groups;
    active.insert(resource_id[i]);
  };
  auto dequeue=[&](Id resource,Id i) {
    auto& queue=queues[resource];auto at=queue.groups.find(ready[i]);auto& group=at->second;
    bool ambiguous=group.duplicate_keys;auto key=priorities[i].value_or(0);
    auto multiplicity=group.multiplicity.find(key);
    if(multiplicity->second--==2)--group.duplicate_keys;
    if(!multiplicity->second)group.multiplicity.erase(multiplicity);
    if(!priorities[i])--group.missing;
    group.order.erase({key,i});--queue.count;queued[i]=false;
    if(ambiguous && !group.duplicate_keys)--queue.ambiguous_groups;
    if(group.order.empty())queue.groups.erase(at);
  };
  for(auto const& resource:lanes)for(double t:resource.second)if(t>map.origin)
    events.push({t,2,0,resource_ids.at(resource.first)});
  double now=map.origin;
  for(;;) {
    std::vector<Event> batch;
    while(!events.empty() && events.top().time==now) {batch.push_back(events.top());events.pop();}
    std::vector<Event> rmws;
    for(auto const& e:batch)if(e.kind==0)rmws.push_back(e);
    for(Id a=0;a<rmws.size();++a)for(Id b=a+1;b<rmws.size();++b)if(group[rmws[a].op]==group[rmws[b].op]) {
      auto pa=priority(rmws[a].op),pb=priority(rmws[b].op);
      if(!pa || !pb || *pa==*pb) {out.pending.push_back("RMW linearization tie unresolved");return out;}
    }
    std::sort(rmws.begin(),rmws.end(),[&](auto const& a,auto const& b) {
      return priority(a.op).value_or(0)<priority(b.op).value_or(0);
    });
    for(auto const& e:rmws) {
      Id g=group[e.op],m=member[e.op];last[e.op]=++arrivals[g]==map.groups[g].size();
      if(*last[e.op])out.owners[g]=e.op;
      else {auto merge=map.groups[g][m].merge;if(status[merge]!=3 && status[merge]!=4)++completed;status[merge]=4;}
      if(*last[e.op] && status[map.groups[g][m].merge]==1 && ready[map.groups[g][m].merge]<=now)
        newly_ready.push_back(map.groups[g][m].merge);
      emit(e.op,Phase::kRmw,now);continuation(g,m,now);
    }
    for(auto const& e:batch)if(e.kind>=3) {emit(e.op,static_cast<Phase>(e.kind+1),now);--pending_progress;}
    for(auto const& e:batch)if(e.kind==1) {
      if(status[e.op]!=2)throw std::invalid_argument("completion of nonrunning operation");
      status[e.op]=3;++completed;emit(e.op,Phase::kDone,now);
      if(returned_group[e.op]>=0)continuation(returned_group[e.op],returned_member[e.op],now);
      if(merge_group[e.op]>=0)continuation(merge_group[e.op],merge_member[e.op],now);
    }
    for(auto const& e:batch)if(e.kind==2) {
      if(e.resource!=std::numeric_limits<Id>::max())active.insert(e.resource);
      else if(status[e.op]==1 && ready[e.op]<=now)newly_ready.push_back(e.op);
    }
    // START may release work, but the old scan exposed it only next wave.
    for(auto i:newly_ready)enqueue(i);
    newly_ready.clear();auto wave=std::move(active);active.clear();
    bool launched=false;
    for(auto resource:wave) {
      auto& queue=queues[resource];if(!queue.count)continue;
      auto& available=lanes[queue.name];
      Id free=queue.name.empty()?queue.count:std::count_if(available.begin(),available.end(),[&](double t) {return t<=now;});
      if(!free)continue;
      // Preserve the old last-admitted ready-time test even if its whole group
      // fits: a later group outside capacity still triggers this strict check.
      if(free<queue.count) {
        Id position=free;auto group=queue.groups.begin();
        while(position>group->second.order.size()) {position-=group->second.order.size();++group;}
        if(group->second.order.size()>1 && (group->second.missing || group->second.duplicate_keys)) {
          out.pending.push_back("resource admission tie unresolved: "+queue.name);return out;
        }
      }
      std::vector<Id> admitted;
      if(queue.ambiguous_groups) {
        // Accepted equal-key ties had std::sort's unspecified order. Reproduce
        // those exact passes; no implicit source priority is added to the ABI.
        for(auto const& group:queue.groups)for(auto const& entry:group.second.order)admitted.push_back(entry.second);
        std::sort(admitted.begin(),admitted.end());
        std::sort(admitted.begin(),admitted.end(),[&](Id a,Id b) {return ready[a]<ready[b];});
        std::sort(admitted.begin(),admitted.end(),[&](Id a,Id b) {
          if(ready[a]!=ready[b])return ready[a]<ready[b];
          return priorities[a].value_or(0)<priorities[b].value_or(0);
        });
        admitted.resize(std::min(free,queue.count));
      }else {
        Id remaining=std::min(free,queue.count);
        for(auto const& group:queue.groups) {
          for(auto const& entry:group.second.order) {admitted.push_back(entry.second);if(!--remaining)break;}
          if(!remaining)break;
        }
      }
      for(auto i:admitted) {
        dequeue(resource,i);status[i]=2;emit(i,Phase::kStart,now);launched=true;
        auto const& s=services[i];
        if(!queue.name.empty()) {
          auto lane=std::find_if(available.begin(),available.end(),[&](double t) {return t<=now;});
          *lane=now+s.admission_occupancy;if(s.admission_occupancy)events.push({*lane,2,i,resource});
        }
        for(Id p=0;p<s.progress_offsets.size();++p)if(s.progress_offsets[p]) {events.push({now+*s.progress_offsets[p],int(3+p),i});++pending_progress;}
        if(s.rmw_offset)events.push({now+*s.rmw_offset,0,i});
        events.push({now+s.completion_delay,1,i});
      }
      if(queue.count && (queue.name.empty() || std::any_of(available.begin(),available.end(),[&](double t) {return t<=now;})))
        active.insert(resource);
    }
    if(completed==count && !pending_progress) {
      if(out.completions[map.finish])out.scope_time=*out.completions[map.finish]-map.origin;
      else out.pending.push_back("inactive finish boundary");
      out.resource_available=lanes;return out;
    }
    if(launched)continue;
    if(!events.empty()) {now=events.top().time;continue;}
    out.pending.push_back("source-start mapping deadlocked/incomplete");return out;
  }
}
}  // namespace tilemega::solver
