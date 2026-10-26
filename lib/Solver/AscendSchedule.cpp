// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/AscendSchedule.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
namespace {
std::uint32_t U32(std::size_t value) {
  if(value>std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("Ascend task/queue index overflow");
  return static_cast<std::uint32_t>(value);
}
std::vector<std::uint32_t> Participants(AscendScheduledTask const& task,
    ascend::WorkerMap const& map,int owner) {
  auto domain=map.Domain(task.role);
  if(std::find(domain.begin(),domain.end(),owner)==domain.end())
    throw std::invalid_argument("Task owner has an incompatible role");
  std::vector<std::uint32_t> result{unsigned(owner)};
  if(task.role==ascend::ImplementationRole::kMix1_2) {
    if(task.mix_aiv_lanes!=1 && task.mix_aiv_lanes!=2)
      throw std::invalid_argument("MIX must declare one or two AIV participants");
    for(auto const& w:map.workers)
      if(w.group==map.At(owner).group && w.role==ascend::TrackRole::kAiv &&
          unsigned(w.sub_block_id)<task.mix_aiv_lanes)result.push_back(w.global_worker_id);
    if(result.size()!=1+task.mix_aiv_lanes)throw std::invalid_argument("Incomplete MIX group");
  }
  return result;
}
struct Graph {
  std::vector<std::vector<std::uint32_t>> successors;
  std::vector<std::uint32_t> indegree,order;
};
Graph DataGraph(std::vector<AscendScheduledTask> const& tasks) {
  Graph g;g.successors.resize(tasks.size());g.indegree.resize(tasks.size());
  for(std::size_t t=0;t<tasks.size();++t) {
    std::set<std::uint32_t> unique;
    for(auto p:tasks[t].predecessors) {
      if(p>=tasks.size() || p==t || !unique.insert(p).second)
        throw std::invalid_argument("Invalid/duplicate semantic predecessor");
      g.successors[p].push_back(U32(t));++g.indegree[t];
    }
  }
  auto indegree=g.indegree;std::queue<std::uint32_t> ready;
  for(std::size_t t=0;t<tasks.size();++t)if(!indegree[t])ready.push(U32(t));
  while(!ready.empty()) {
    auto t=ready.front();ready.pop();g.order.push_back(t);
    for(auto s:g.successors[t])if(!--indegree[s])ready.push(s);
  }
  if(g.order.size()!=tasks.size())throw std::invalid_argument("Semantic dependencies contain a cycle");
  return g;
}
}
std::uint32_t AscendProgressValue(std::uint64_t epoch,std::uint32_t prefix,std::uint32_t stride) {
  if(!stride || prefix>=stride || epoch>(std::numeric_limits<std::uint32_t>::max()-prefix)/stride)
    throw std::overflow_error("Progress epoch/prefix encoding overflow");
  return static_cast<std::uint32_t>(epoch*stride+prefix);
}
AscendSchedule ScheduleAscendTasks(std::vector<AscendScheduledTask> const& tasks,
    ascend::WorkerMap const& map,AscendSchedulingCosts const& costs) {
  if(tasks.empty() || map.workers.empty() || tasks.size()>=std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Empty or unrepresentable native task graph");
  if(!std::isfinite(costs.gm_bytes_per_ns) || costs.gm_bytes_per_ns<=0 ||
      !std::isfinite(costs.cross_worker_ready_ns) || costs.cross_worker_ready_ns<0 || costs.provenance.empty())
    throw std::invalid_argument("Native scheduling requires explicit shared-GM and synchronization prices");
  auto graph=DataGraph(tasks);
  std::vector<double> rank(tasks.size(),0);
  for(auto i=graph.order.rbegin();i!=graph.order.rend();++i) {
    auto const& price=tasks[*i].price;
    if(!std::isfinite(price.service_ns) || price.service_ns<=0 || price.provenance.empty())
      throw std::invalid_argument("Unknown/zero/nonfinite task price cannot enter native ranking");
    double tail=0;for(auto next:graph.successors[*i])tail=std::max(tail,rank[next]);
    rank[*i]=std::max(price.service_ns,price.gm_bytes/costs.gm_bytes_per_ns)+tail;
  }
  auto compare=[&](std::uint32_t a,std::uint32_t b){return rank[a]!=rank[b]?rank[a]<rank[b]:a>b;};
  std::priority_queue<std::uint32_t,std::vector<std::uint32_t>,decltype(compare)> ready(compare);
  auto indegree=graph.indegree;
  for(std::size_t t=0;t<tasks.size();++t)if(!indegree[t])ready.push(U32(t));
  AscendSchedule out;
  out.owners.resize(tasks.size());out.ordinals.resize(tasks.size());out.waits.resize(tasks.size());
  out.participants.resize(tasks.size());out.start_ns.resize(tasks.size());out.finish_ns.resize(tasks.size());
  std::vector<double> available(map.workers.size(),0);
  std::vector<std::vector<std::uint32_t>> queues(map.workers.size());
  double gm_available=0;
  while(!ready.empty()) {
    auto id=ready.top();ready.pop();auto const& task=tasks[id];
    auto domain=task.legal_workers.empty()?map.Domain(task.role):task.legal_workers;
    if(domain.empty())throw std::invalid_argument("Native task has no legal worker");
    double best_end=std::numeric_limits<double>::infinity(),best_start=0;
    int best_owner=-1;std::vector<std::uint32_t> best_participants;
    const double gm_ns=task.price.gm_bytes/costs.gm_bytes_per_ns;
    for(int owner:domain) {
      auto participants=Participants(task,map,owner);
      double start=gm_ns?gm_available:0;
      for(auto w:participants)start=std::max(start,available.at(w));
      for(auto p:task.predecessors) {
        // One readiness latency covers concurrent incoming messages. A shared
        // link/GM service constraint is accounted once by the global calendar.
        const double ready_ns=out.finish_ns[p]+(out.owners[p]==unsigned(owner)?0:costs.cross_worker_ready_ns);
        start=std::max(start,ready_ns);
      }
      double end=start+std::max(task.price.service_ns,gm_ns);
      if(end<best_end || (end==best_end && owner<best_owner)) {
        best_end=end;best_start=start;best_owner=owner;best_participants=std::move(participants);
      }
    }
    out.owners[id]=best_owner;out.participants[id]=best_participants;
    out.start_ns[id]=best_start;out.finish_ns[id]=best_end;
    out.makespan_ns=std::max(out.makespan_ns,best_end);out.gm_floor_ns+=gm_ns;
    if(gm_ns)gm_available=best_start+gm_ns;
    for(auto w:best_participants){available[w]=best_end;queues[w].push_back(id);}
    for(auto s:graph.successors[id])if(!--indegree[s])ready.push(s);
  }
  out.offsets.push_back(0);std::uint32_t largest=0;
  for(unsigned w=0;w<queues.size();++w) {
    auto const& queue=queues[w];largest=std::max(largest,U32(queue.size()));
    for(unsigned ordinal=0;ordinal<queue.size();++ordinal)
      if(out.owners[queue[ordinal]]==w)out.ordinals[queue[ordinal]]=ordinal+1;
    out.queue.insert(out.queue.end(),queue.begin(),queue.end());out.offsets.push_back(U32(out.queue.size()));
    if(!queue.empty())out.fifo_edges+=queue.size()-1;
  }
  if(largest==std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("Progress stride overflow");
  out.epoch_stride=largest+1;
  for(std::size_t t=0;t<tasks.size();++t) {
    std::map<unsigned,unsigned> needed;
    for(auto p:tasks[t].predecessors) {
      ++out.semantic_edges;
      for(auto consumer:out.participants[t])if(out.owners[p]!=consumer)
        needed[out.owners[p]]=std::max(needed[out.owners[p]],out.ordinals[p]);
    }
    for(auto const& [worker,prefix]:needed)out.waits[t].push_back({worker,prefix});
    out.lowered_waits+=out.waits[t].size();
  }
  AuditAscendSchedule(tasks,map,out);return out;
}
AscendSchedule PriceAscendSchedule(std::vector<AscendScheduledTask> const& tasks,
    ascend::WorkerMap const& map,AscendSchedulingCosts const& costs,AscendSchedule out) {
  AuditAscendSchedule(tasks,map,out);
  if(!std::isfinite(costs.gm_bytes_per_ns) || costs.gm_bytes_per_ns<=0 ||
      !std::isfinite(costs.cross_worker_ready_ns) || costs.cross_worker_ready_ns<0 || costs.provenance.empty())
    throw std::invalid_argument("Fixed native plan requires sourced shared-resource prices");
  auto graph=DataGraph(tasks);
  for(unsigned w=0;w<map.workers.size();++w)
    for(unsigned slot=out.offsets[w]+1;slot<out.offsets[w+1];++slot) {
      graph.successors[out.queue[slot-1]].push_back(out.queue[slot]);++graph.indegree[out.queue[slot]];
    }
  out.start_ns.assign(tasks.size(),0);out.finish_ns.assign(tasks.size(),0);
  out.makespan_ns=out.gm_floor_ns=0;
  std::vector<unsigned> ready;std::vector<double> available(map.workers.size(),0);
  for(unsigned t=0;t<tasks.size();++t) {
    auto const& price=tasks[t].price;
    if(!std::isfinite(price.service_ns) || price.service_ns<=0 || price.provenance.empty())
      throw std::invalid_argument("Unknown/zero task price in fixed native plan");
    if(!graph.indegree[t])ready.push_back(t);
  }
  double gm_available=0;unsigned visited=0;
  while(!ready.empty()) {
    auto begin=[&](unsigned id) {
      double time=tasks[id].price.gm_bytes?gm_available:0;
      for(auto w:out.participants[id])time=std::max(time,available[w]);
      for(auto p:tasks[id].predecessors)time=std::max(time,out.finish_ns[p]+
        (out.owners[p]==out.owners[id]?0:costs.cross_worker_ready_ns));
      return time;
    };
    auto chosen=std::min_element(ready.begin(),ready.end(),[&](unsigned a,unsigned b) {
      auto x=begin(a),y=begin(b);return x==y?a<b:x<y;
    });
    unsigned id=*chosen;double start=begin(id);ready.erase(chosen);++visited;
    double gm_ns=tasks[id].price.gm_bytes/costs.gm_bytes_per_ns;
    double finish=start+std::max(tasks[id].price.service_ns,gm_ns);
    out.start_ns[id]=start;out.finish_ns[id]=finish;out.makespan_ns=std::max(out.makespan_ns,finish);
    out.gm_floor_ns+=gm_ns;if(gm_ns)gm_available=start+gm_ns;
    for(auto w:out.participants[id])available[w]=finish;
    for(auto next:graph.successors[id])if(!--graph.indegree[next])ready.push_back(next);
  }
  if(visited!=tasks.size())throw std::invalid_argument("Fixed native schedule has a waiting cycle");
  return out;
}
void AuditAscendSchedule(std::vector<AscendScheduledTask> const& tasks,
    ascend::WorkerMap const& map,AscendSchedule const& out) {
  if(out.owners.size()!=tasks.size() || out.ordinals.size()!=tasks.size() || out.waits.size()!=tasks.size() ||
      out.participants.size()!=tasks.size() || out.offsets.size()!=map.workers.size()+1 ||
      out.offsets.front()!=0 || out.offsets.back()!=out.queue.size())
    throw std::invalid_argument("Malformed native schedule tables");
  auto graph=DataGraph(tasks);std::vector<std::vector<unsigned>> seen(tasks.size());
  for(unsigned w=0;w<map.workers.size();++w) {
    if(out.offsets[w]>out.offsets[w+1])throw std::invalid_argument("Invalid queue offset order");
    for(unsigned q=out.offsets[w];q<out.offsets[w+1];++q) {
      unsigned t=out.queue.at(q);
      if(t>=tasks.size())throw std::invalid_argument("Queue references an invalid task");
      seen[t].push_back(w);
      if(out.owners[t]==w && out.ordinals[t]!=q-out.offsets[w]+1)
        throw std::invalid_argument("Progress ordinal disagrees with publisher FIFO");
      if(q>out.offsets[w]) {
        unsigned p=out.queue[q-1];graph.successors[p].push_back(t);++graph.indegree[t];
      }
    }
  }
  for(unsigned t=0;t<tasks.size();++t) {
    auto expected=Participants(tasks[t],map,out.owners[t]);std::sort(expected.begin(),expected.end());
    if(!tasks[t].legal_workers.empty() && std::find(tasks[t].legal_workers.begin(),tasks[t].legal_workers.end(),out.owners[t])==tasks[t].legal_workers.end())
      throw std::invalid_argument("Owner violates placement/affinity constraints");
    if(seen[t]!=expected || !out.ordinals[t] || out.ordinals[t]>=out.epoch_stride)
      throw std::invalid_argument("Missing/duplicate participant or invalid progress ordinal");
    std::map<unsigned,unsigned> required,actual;
    for(auto p:tasks[t].predecessors)
      for(auto w:expected)if(out.owners[p]!=w)required[out.owners[p]]=std::max(required[out.owners[p]],out.ordinals[p]);
    for(auto wait:out.waits[t])
      if(!actual.emplace(wait.producer_worker,wait.required_prefix).second)
        throw std::invalid_argument("Duplicate producer frontier wait");
    if(!out.fifo_wait_elision && actual!=required)
      throw std::invalid_argument("Frontier waits do not cover semantic predecessors");
    if(out.fifo_wait_elision)for(auto const& [producer,prefix]:actual)
      if(!required.count(producer) || required.at(producer)!=prefix)
        throw std::invalid_argument("Elided plan changes a required frontier value");
  }
  std::queue<unsigned> ready;for(unsigned t=0;t<tasks.size();++t)if(!graph.indegree[t])ready.push(t);
  std::vector<unsigned> order;
  while(!ready.empty()) {auto t=ready.front();ready.pop();order.push_back(t);for(auto s:graph.successors[t])if(!--graph.indegree[s])ready.push(s);}
  if(order.size()!=tasks.size())throw std::invalid_argument("Dependencies plus worker FIFO contain a deadlock");
  if(out.fifo_wait_elision) {
    std::vector<std::vector<unsigned>> observed(map.workers.size(),std::vector<unsigned>(map.workers.size()));
    std::uint64_t waits=0;
    for(auto t:order) {
      for(auto wait:out.waits[t]) {
        ++waits;
        for(auto worker:out.participants[t])observed[worker][wait.producer_worker]=
            std::max(observed[worker][wait.producer_worker],wait.required_prefix);
      }
      for(auto p:tasks[t].predecessors)for(auto worker:out.participants[t])
        if(worker!=out.owners[p] && observed[worker][out.owners[p]]<out.ordinals[p])
          throw std::invalid_argument("FIFO frontier elision lacks a prior observation on a participant");
    }
    if(waits!=out.lowered_waits || waits>out.waits_before_fifo_elision)
      throw std::invalid_argument("Invalid FIFO frontier wait accounting");
  }
}
AscendSchedule ElideAscendFifoWaits(std::vector<AscendScheduledTask> const& tasks,
    ascend::WorkerMap const& map,AscendSchedule out) {
  AuditAscendSchedule(tasks,map,out);
  if(out.fifo_wait_elision)return out;
  auto graph=DataGraph(tasks);
  for(unsigned w=0;w<map.workers.size();++w)
    for(unsigned slot=out.offsets[w]+1;slot<out.offsets[w+1];++slot) {
      graph.successors[out.queue[slot-1]].push_back(out.queue[slot]);++graph.indegree[out.queue[slot]];
    }
  std::queue<unsigned> ready;
  for(unsigned t=0;t<tasks.size();++t)if(!graph.indegree[t])ready.push(t);
  std::vector<std::vector<unsigned>> observed(map.workers.size(),std::vector<unsigned>(map.workers.size()));
  out.waits_before_fifo_elision=out.lowered_waits;out.lowered_waits=0;out.fifo_wait_elision=true;
  while(!ready.empty()) {
    auto t=ready.front();ready.pop();std::vector<AscendFrontierWait> kept;
    for(auto wait:out.waits[t]) {
      const bool redundant=std::all_of(out.participants[t].begin(),out.participants[t].end(),[&](auto worker) {
        return worker==wait.producer_worker || observed[worker][wait.producer_worker]>=wait.required_prefix;
      });
      if(!redundant) {
        kept.push_back(wait);++out.lowered_waits;
        for(auto worker:out.participants[t])observed[worker][wait.producer_worker]=
            std::max(observed[worker][wait.producer_worker],wait.required_prefix);
      }
    }
    out.waits[t]=std::move(kept);
    for(auto s:graph.successors[t])if(!--graph.indegree[s])ready.push(s);
  }
  AuditAscendSchedule(tasks,map,out);return out;
}
AscendSchedule DiagnosticAscendSchedule(std::vector<AscendScheduledTask> const& tasks,
    ascend::WorkerMap const& map) {
  auto graph=DataGraph(tasks);
  if(tasks.empty())throw std::invalid_argument("Empty diagnostic task graph");
  AscendSchedule out;out.owners.resize(tasks.size());out.ordinals.resize(tasks.size());
  out.participants.resize(tasks.size());out.waits.resize(tasks.size());
  std::vector<std::vector<unsigned>> queues(map.workers.size());
  for(auto id:graph.order) {
    auto const& t=tasks[id];auto domain=t.legal_workers.empty()?map.Domain(t.role):t.legal_workers;
    if(domain.empty())throw std::invalid_argument("Diagnostic task has no legal worker");
    auto owner=*std::min_element(domain.begin(),domain.end(),[&](int a,int b) {
      auto load=[&](int w){std::size_t n=0;for(auto p:Participants(t,map,w))n=std::max(n,queues.at(p).size());return n;};
      auto x=load(a),y=load(b);return x==y?a<b:x<y;
    });
    out.owners[id]=owner;out.participants[id]=Participants(t,map,owner);
    for(auto w:out.participants[id])queues.at(w).push_back(id);
  }
  out.offsets.push_back(0);unsigned maximum=0;
  for(unsigned w=0;w<queues.size();++w) {
    for(unsigned i=0;i<queues[w].size();++i)
      if(out.owners[queues[w][i]]==w)out.ordinals[queues[w][i]]=i+1;
    maximum=std::max(maximum,U32(queues[w].size()));
    out.queue.insert(out.queue.end(),queues[w].begin(),queues[w].end());
    out.offsets.push_back(U32(out.queue.size()));
    if(!queues[w].empty())out.fifo_edges+=queues[w].size()-1;
  }
  if(maximum==UINT32_MAX)throw std::overflow_error("Diagnostic epoch stride overflow");
  out.epoch_stride=maximum+1;
  for(unsigned id=0;id<tasks.size();++id) {
    std::map<unsigned,unsigned> waits;
    for(auto p:tasks[id].predecessors) {
      ++out.semantic_edges;
      for(auto w:out.participants[id])if(out.owners[p]!=w)
        waits[out.owners[p]]=std::max(waits[out.owners[p]],out.ordinals[p]);
    }
    for(auto const& [w,n]:waits)out.waits[id].push_back({w,n});
    out.lowered_waits+=out.waits[id].size();
  }
  AuditAscendSchedule(tasks,map,out);return out;
}
std::vector<AscendScheduledTask> AscendTasksFromSymbolic(SymbolicProblem const& problem,
    PlanSkeleton const& skeleton,std::vector<AscendServicePrice> const& prices) {
  if(!skeleton.ascend_workers || problem.offsets.empty() || problem.offsets.back()<1 ||
      prices.size()!=std::size_t(problem.offsets.back()))throw std::invalid_argument("Unbound/unpriced native CTG");
  std::vector<AscendScheduledTask> tasks(prices.size());
  for(unsigned s=0;s<skeleton.spaces.size();++s) {
    auto const& space=skeleton.spaces[s];
    if(!space.ascend_role || space.colocated_producer>=0 || space.same_group_affinity_producer>=0)
      throw std::invalid_argument("Native list scheduling requires explicit affinity lowering");
    for(int tile=0;tile<space.count;++tile) {
      auto id=space.offset+tile;auto& task=tasks.at(id);
      task.stage=s;task.logical_task=tile;task.role=*space.ascend_role;
      task.price=prices.at(id);task.legal_workers=space.legal_workers;
      std::set<unsigned> predecessors;
      for(auto e:skeleton.incoming.at(s)) {
        auto const& edge=skeleton.edges.at(e);auto const& producer=skeleton.spaces.at(edge.producer);
        edge.oracle->reverse.Query({tile},skeleton.theta).ForEach([&](auto const& p) {
          if(p.size()!=1 || p[0]<0 || p[0]>=producer.count)
            throw std::invalid_argument("Native predecessor lies outside its CTG task domain");
          predecessors.insert(producer.offset+p[0]);
        });
      }
      task.predecessors.assign(predecessors.begin(),predecessors.end());
    }
  }
  return tasks;
}
std::string AscendSchedule::ToJson() const {
  using namespace json;Array owner,ordinal,offset,refs,start,finish,all_waits;
  for(auto v:owners)owner.emplace_back(int(v));for(auto v:ordinals)ordinal.emplace_back(int(v));
  for(auto v:offsets)offset.emplace_back(double(v));for(auto v:queue)refs.emplace_back(double(v));
  for(auto v:start_ns)start.emplace_back(v);for(auto v:finish_ns)finish.emplace_back(v);
  for(auto const& task:waits){Array ws;for(auto w:task)ws.emplace_back(Object{{"producer_worker",int(w.producer_worker)},{"required_prefix",double(w.required_prefix)}});all_waits.emplace_back(std::move(ws));}
  return Value(Object{{"schema","tilemega.ascend.schedule.v2"},{"algorithm","heterogeneous_rank_eft_shared_gm"},
    {"owners",std::move(owner)},{"ordinals",std::move(ordinal)},{"schedule_offsets",std::move(offset)},
    {"schedule",std::move(refs)},{"start_ns",std::move(start)},{"finish_ns",std::move(finish)},
    {"frontier_waits",std::move(all_waits)},{"epoch_stride",double(epoch_stride)},
    {"semantic_edges",double(semantic_edges)},{"fifo_edges",double(fifo_edges)},
    {"wait_lowering",fifo_wait_elision?"experimental_fifo_observed_frontier":"per_task_frontier"},
    {"waits_before_fifo_elision",double(fifo_wait_elision?waits_before_fifo_elision:lowered_waits)},
    {"lowered_waits",double(lowered_waits)},{"makespan_ns",makespan_ns},{"shared_gm_floor_ns",gm_floor_ns},
    {"wait_graph_acyclic",true},{"device_execution_verified",false}}).Dump();
}
} // namespace tilemega::solver
