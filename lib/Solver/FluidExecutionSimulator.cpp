// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/DramFluid.h>
#include <algorithm>
#include <cmath>
#include <queue>
#include <optional>
#include <stdexcept>
namespace tilemega::solver {
bool SimulateFluidExecution(SimulatorInput const& input,MaterializedPlan const& plan,
    SimulatorOptions const& options,HopCurve const& hop,SimulatorResult* out,std::string* error) try {
  if(!input.graph || !out || options.window!=1 || !(options.dram_gbps>0) || !input.prefetch_ns.empty())throw std::invalid_argument("fluid simulation requires FIFO W=1, positive DRAM rate, and no prefetch credit");
  int nodes=input.graph->successors.size(),workers=plan.queue.size();
  if(input.task_price_parts.size()!=std::size_t(nodes))throw std::invalid_argument("missing per-task fluid prices");
  for(auto const* mask:{&input.publication_required,&input.consumer_wait_required})if(!mask->empty() && mask->size()!=std::size_t(nodes))throw std::invalid_argument("invalid fluid synchronization mask");
  PreparedExecutionGraph local_graph;auto graph=input.prepared_graph;
  if(!graph){if(!PrepareExecutionGraph(*input.graph,&local_graph,error))return false;graph=&local_graph;}
  PreparedExecutionPlan local_plan;auto prepared=input.prepared_plan;
  if(!prepared){if(!PrepareExecutionPlan(*graph,plan,&local_plan,error))return false;prepared=&local_plan;}
  std::vector<std::vector<int>> forced(nodes);
  for(auto [p,s]:input.fluid_forced_local_hops){if(p<0 || p>=nodes || s<0 || s>=nodes || prepared->owner[p]!=prepared->owner[s])throw std::invalid_argument("invalid forced local hop");forced[p].push_back(s);}
  auto pending=prepared->unmet;std::vector<int> remaining=graph->producer_count,head(workers),fluid_owner;
  std::vector<double> ready(nodes),available(workers),work(workers);
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
  *out={};out->tasks.resize(nodes);out->cross_worker_edges=prepared->cross_edges;out->same_worker_edges=prepared->same_edges;
  auto enqueue=[&](int w){auto const& queue=prepared->queue[w];if(head[w]>=int(queue.size()))return;int n=queue[head[w]];if(pending[n] || state[n])return;state[n]=1;events.push({std::max(available[w],ready[n]),Start,n});};
  auto finish=[&](int n){if(compute[n] && bytes[n] && !closing[n]){closing[n]=1;bool publish=input.publication_required.empty()?prepared->cross_fanout[n]>0:input.publication_required[n]!=0;events.push({now+(publish?options.publication_ns:0),End,n});}};
  for(int w=0;w<workers;++w)enqueue(w);
  while(completed<nodes) {
    double next=std::min(events.empty()?std::numeric_limits<double>::infinity():events.top().at,now+fluid_next());
    if(!std::isfinite(next))throw std::runtime_error("fluid FIFO deadlock");
    auto delivered=fluid_advance(next-now);now=next;for(int id:delivered){int n=fluid_owner[id];bytes[n]=1;finish(n);}
    while(!events.empty() && events.top().at<=now){auto e=events.top();events.pop();int n=e.node,w=prepared->owner[n];auto const& parts=input.task_price_parts[n];auto& task=out->tasks[n];
      if(e.kind==Start){++running;task.worker=w;task.start_ns=now;task.block_ns=std::max(0.,now-available[w]);out->total_block_ns+=task.block_ns;
        bool wait=input.consumer_wait_required.empty()?prepared->cross_input[n]!=0:input.consumer_wait_required[n]!=0;
        events.push({now+(wait?options.consumer_wait_ns:0)+parts.fixed_ns,Main,n});
      }else if(e.kind==Main){double demand=parts.dram_bytes-(options.no_external_dram?parts.no_producer_dram_bytes:0);
        if(demand<0)throw std::invalid_argument("negative fluid demand");
        if(demand>0){int id=inflight?inflight->Add(demand,parts.dram_rate_cap,
            parts.inflight_bytes):fluid.Add(demand,parts.dram_rate_cap);
          if(id!=int(fluid_owner.size()))throw std::runtime_error("fluid id discontinuity");fluid_owner.push_back(n);}else bytes[n]=1;
        events.push({now+parts.compute_ns,Computed,n});
      }else if(e.kind==Computed){compute[n]=1;finish(n);}
      else{--running;++completed;task.end_ns=now;work[w]+=now-task.start_ns;out->total_work_ns+=now-task.start_ns;available[w]=now;++head[w];
        int g=graph->group_of_node[n];double edge=options.flat_hop?hop.c0:hop.Ns(prepared->cross_fanout[n],std::max(1,running));
        arrivals[g].Add(w,now,now+edge);
        for(int s:forced[n])ready[s]=std::max(ready[s],now+edge);
        if(--remaining[g]==0)graph->successors[g].Visit([&](int s){ready[s]=std::max(ready[s],arrivals[g].Ready(prepared->owner[s]));if(--pending[s]==0)enqueue(prepared->owner[s]);});
        enqueue(w);
      }
    }
  }
  out->makespan_ns=now;out->critical_path_ns=now;out->solo_work_ns=out->total_work_ns;
  for(int w=0;w<workers;++w)if(work[w]>out->busiest_worker_ns){out->busiest_worker_ns=work[w];out->busiest_worker=w;}
  if(options.all_external_miss && !options.no_external_dram && now+1e-6<options.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in fluid simulator");
  return true;
 }catch(std::exception const& e){if(error)*error=e.what();return false;}
} // namespace tilemega::solver
