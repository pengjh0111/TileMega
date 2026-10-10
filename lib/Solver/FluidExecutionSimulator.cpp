// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/DramFluid.h>
#include <algorithm>
#include <cmath>
#include <queue>
#include <deque>
#include <optional>
#include <stdexcept>
namespace tilemega::solver {
bool SimulateFluidExecution(SimulatorInput const& input,MaterializedPlan const& plan,
    SimulatorOptions const& options,HopCurve const& hop,SimulatorResult* out,std::string* error) try {
  if(!input.graph || !out || options.window!=1 || !(options.dram_gbps>0) || !input.prefetch_ns.empty())throw std::invalid_argument("fluid simulation requires FIFO W=1, positive DRAM rate, and no prefetch credit");
  bool const paged=options.page_bytes>0 || options.pages_per_worker>0;
  if(paged && (options.page_bytes<=0 || options.pages_per_worker<=0))
    throw std::invalid_argument("paged fluid simulation needs a positive page ring");
  int nodes=input.graph->successors.size(),workers=plan.queue.size();
  if(input.task_price_parts.size()!=std::size_t(nodes))throw std::invalid_argument("missing per-task fluid prices");
  if(!input.inline_reducer.empty() && input.inline_reducer.size()!=std::size_t(nodes))
    throw std::invalid_argument("invalid inline reducer mask");
  if(!input.inline_body_reserved.empty() && input.inline_body_reserved.size()!=std::size_t(nodes))
    throw std::invalid_argument("invalid reserved inline body mask");
  auto reserved_inline=[&](int n){return !input.inline_body_reserved.empty() && input.inline_body_reserved[n];};
  bool reserved_mode=std::any_of(input.inline_body_reserved.begin(),input.inline_body_reserved.end(),
      [](unsigned char value){return value!=0;});
  auto inline_node=[&](int n){return !input.inline_reducer.empty() && input.inline_reducer[n];};
  for(int n=0;n<nodes;++n)if(reserved_inline(n) && !inline_node(n))
    throw std::invalid_argument("reserved inline body is not a reducer");
  for(auto const* mask:{&input.publication_required,&input.consumer_wait_required})if(!mask->empty() && mask->size()!=std::size_t(nodes))throw std::invalid_argument("invalid fluid synchronization mask");
  PreparedExecutionGraph local_graph;auto graph=input.prepared_graph;
  if(!graph){if(!PrepareExecutionGraph(*input.graph,&local_graph,error))return false;graph=&local_graph;}
  PreparedExecutionPlan local_plan;auto prepared=input.prepared_plan;
  if(!prepared){if(!PrepareExecutionPlan(*graph,plan,&local_plan,error))return false;prepared=&local_plan;}
  std::vector<std::vector<int>> forced(nodes);
  for(auto [p,s]:input.fluid_forced_local_hops){if(p<0 || p>=nodes || s<0 || s>=nodes || prepared->owner[p]!=prepared->owner[s])throw std::invalid_argument("invalid forced local hop");forced[p].push_back(s);}
  auto pending=prepared->unmet;std::vector<int> remaining=graph->producer_count,head(workers),fluid_owner;
  std::vector<int> prefetch_head(workers);
  std::vector<double> ready(nodes),available(workers),work(workers);
  std::vector<int> inline_worker(nodes,-1);
  std::vector<std::deque<int>> inline_ready(workers);
  std::vector<unsigned char> busy(workers);
  std::vector<double> page_hold(workers),reserved(nodes),prefetch_bytes(nodes);
  std::vector<unsigned char> prefetch_ready(nodes,!paged);
  std::vector<unsigned char> state(nodes),compute(nodes),bytes(nodes),closing(nodes);
  struct Group {double end=0,best=0,second=0;int owner=-1;
    void Add(int w,double finish,double arrival){end=std::max(end,finish);if(w==owner){best=std::max(best,arrival);return;}if(arrival>=best){second=best;best=arrival;owner=w;}else second=std::max(second,arrival);}
    double Ready(int w)const{return std::max(end,w==owner?second:best);}
  };
  std::vector<Group> arrivals(remaining.size());
  enum Kind{Start,Main,Computed,End};struct Event{double at;Kind kind;int node;};
  struct Later{bool operator()(Event const& a,Event const& b)const{return std::tie(a.at,a.kind,a.node)>std::tie(b.at,b.kind,b.node);}};
  std::priority_queue<Event,std::vector<Event>,Later> events;DramFluidServer fluid(options.dram_gbps);
  std::optional<InflightDramServer> inflight;
  if(options.inflight_dram)inflight.emplace(options.dram_gbps,options.inflight_curve_bytes,
      options.inflight_curve_gbps,options.cta_stream_curve_bytes,options.cta_stream_curve_gbps);
  auto fluid_next=[&](){return inflight?inflight->Next():fluid.Next();};
  auto fluid_advance=[&](double dt){return inflight?inflight->Advance(dt):fluid.Advance(dt);};
  double now=0;int completed=0,running=0;
  double const page_capacity=double(options.page_bytes)*options.pages_per_worker;
  *out={};out->tasks.resize(nodes);out->cross_worker_edges=prepared->cross_edges;out->same_worker_edges=prepared->same_edges;
  auto enqueue=[&](int w){
    if(busy[w])return;
    if(!inline_ready[w].empty()) {
      int n=inline_ready[w].front();inline_ready[w].pop_front();busy[w]=1;
      events.push({std::max({now,available[w],ready[n]}),Start,n});return;
    }
    auto const& queue=prepared->queue[w];
    while(head[w]<int(queue.size()) && inline_node(queue[head[w]]))++head[w];
    if(head[w]>=int(queue.size()))return;int n=queue[head[w]];
    if(pending[n] || state[n] || !prefetch_ready[n])return;
    state[n]=1;busy[w]=1;events.push({std::max({available[w],ready[n],paged?now:0.}),Start,n});};
  auto launch_prefetch=[&](int w){
    if(!paged)return;
    auto const& queue=prepared->queue[w];
    while(prefetch_head[w]<int(queue.size())) {
      int n=queue[prefetch_head[w]];
      if(inline_node(n)){prefetch_ready[n]=1;++prefetch_head[w];continue;}
      auto const& parts=input.task_price_parts[n];
      double bytes=options.no_external_dram?0:std::min(parts.no_producer_dram_bytes,page_capacity);
      if(bytes<0)throw std::invalid_argument("negative prefetch bytes");
      double hold=bytes>0?std::ceil(bytes/options.page_bytes)*options.page_bytes:0;
      if(page_hold[w]+hold>page_capacity+1e-6)break;
      page_hold[w]+=hold;reserved[n]=hold;prefetch_bytes[n]=bytes;++prefetch_head[w];
      if(bytes==0){prefetch_ready[n]=1;enqueue(w);continue;}
      int id=inflight?inflight->Add(bytes,parts.dram_rate_cap,
          std::max(16.,std::min(parts.inflight_bytes,page_capacity)))
          :fluid.Add(bytes,parts.dram_rate_cap);
      if(id!=int(fluid_owner.size()))throw std::runtime_error("prefetch fluid id discontinuity");
      fluid_owner.push_back(-n-1);
    }
  };
  auto finish=[&](int n){if(compute[n] && bytes[n] && !closing[n]){
    closing[n]=1;
    if(paged){int w=prepared->owner[n];page_hold[w]-=reserved[n];reserved[n]=0;launch_prefetch(w);}
    bool publish=input.publication_required.empty()?prepared->cross_fanout[n]>0:input.publication_required[n]!=0;
    events.push({now+(publish?options.publication_ns:0),End,n});
  }};
  for(int w=0;w<workers;++w){launch_prefetch(w);enqueue(w);}
  for(int n=0;n<nodes;++n)if(inline_node(n) && pending[n]==0) {
    state[n]=1;inline_worker[n]=prepared->owner[n];
    if(reserved_inline(n)){inline_ready[inline_worker[n]].push_back(n);enqueue(inline_worker[n]);}
    else events.push({0,Start,n});
  }
  while(completed<nodes) {
    double next=std::min(events.empty()?std::numeric_limits<double>::infinity():events.top().at,now+fluid_next());
    if(!std::isfinite(next))throw std::runtime_error("fluid FIFO deadlock");
    auto delivered=fluid_advance(next-now);now=next;for(int id:delivered){int owner=fluid_owner[id];
      if(owner<0){int n=-owner-1;prefetch_ready[n]=1;enqueue(prepared->owner[n]);}
      else {bytes[owner]=1;finish(owner);}}
    while(!events.empty() && events.top().at<=now){auto e=events.top();events.pop();int n=e.node,w=inline_node(n)?inline_worker[n]:prepared->owner[n];auto const& parts=input.task_price_parts[n];auto& task=out->tasks[n];
      if(e.kind==Start){if(!inline_node(n) || reserved_inline(n))++running;
        task.worker=w;task.start_ns=now;
        task.block_ns=inline_node(n)?0:std::max(0.,now-available[w]);out->total_block_ns+=task.block_ns;
        bool wait=!inline_node(n) && (input.consumer_wait_required.empty()?prepared->cross_input[n]!=0:input.consumer_wait_required[n]!=0);
        events.push({now+(wait?options.consumer_wait_ns:0)+
            (inline_node(n) && !reserved_inline(n)?0:parts.fixed_ns),Main,n});
      }else if(e.kind==Main){
        double demand=parts.dram_bytes-(options.no_external_dram?parts.no_producer_dram_bytes:0)
            -(paged?prefetch_bytes[n]:0);
        if(demand<0)throw std::invalid_argument("negative fluid demand");
        if(demand>0){int id=inflight?inflight->Add(demand,parts.dram_rate_cap,
            parts.inflight_bytes):fluid.Add(demand,parts.dram_rate_cap);
          if(id!=int(fluid_owner.size()))throw std::runtime_error("fluid id discontinuity");fluid_owner.push_back(n);}else bytes[n]=1;
        events.push({now+parts.compute_ns,Computed,n});
      }else if(e.kind==Computed){compute[n]=1;finish(n);}
      else{if(!inline_node(n) || reserved_inline(n))--running;++completed;task.end_ns=now;
        out->total_work_ns+=now-task.start_ns;
        if(!inline_node(n) || reserved_inline(n)) {
          work[w]+=now-task.start_ns;available[w]=now;busy[w]=0;
          if(!inline_node(n))++head[w];
        }
        state[n]=2;
        int g=graph->group_of_node[n];double edge=options.flat_hop?hop.c0:hop.Ns(prepared->cross_fanout[n],std::max(1,running));
        arrivals[g].Add(w,now,now+edge);
        for(int s:forced[n])ready[s]=std::max(ready[s],now+edge);
        std::vector<int> wake;
        if(--remaining[g]==0)graph->successors[g].Visit([&](int s){
          ready[s]=std::max(ready[s],inline_node(s)?now:arrivals[g].Ready(prepared->owner[s]));
          if(--pending[s]==0) {
            if(inline_node(s)) {
              inline_worker[s]=w;state[s]=1;
              if(reserved_inline(s))inline_ready[w].push_back(s);
              else events.push({ready[s],Start,s});
            }
            else if(reserved_mode)wake.push_back(prepared->owner[s]);
            else enqueue(prepared->owner[s]);
          }
        });
        enqueue(w);for(int worker:wake)enqueue(worker);
      }
    }
  }
  out->makespan_ns=now;out->critical_path_ns=now;out->solo_work_ns=out->total_work_ns;
  for(int w=0;w<workers;++w)if(work[w]>out->busiest_worker_ns){out->busiest_worker_ns=work[w];out->busiest_worker=w;}
  if(options.all_external_miss && !options.no_external_dram && now+1e-6<options.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in fluid simulator");
  return true;
 }catch(std::exception const& e){if(error)*error=e.what();return false;}
} // namespace tilemega::solver
