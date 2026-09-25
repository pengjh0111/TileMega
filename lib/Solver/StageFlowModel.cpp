// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Solver/DramFluid.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
namespace tilemega::solver {
int CoarsenRelease(int maximum,int n,int kappa){if(maximum<0 || maximum>=n || kappa<1)throw std::invalid_argument("invalid release domain");return std::min(n-1,kappa*(maximum/kappa)+kappa-1);}
void SetFlowCalibration(FlowProblem& p,TargetSpec const& target,ScalarType dtype,HopCurve const& hop) {
  auto name=dtype==ScalarType::kBF16?"bf16":"f32";auto const& event=target.EventCalibrationFor(name);
  if(!event.task_publication.ns || !event.task_wait.ns)throw std::invalid_argument("flow synchronization calibration missing");
  p.dram_gbps=target.CalibrationFor(name).dram_gbps;p.publication_ns=*event.task_publication.ns;p.consumer_wait_ns=*event.task_wait.ns;p.hop_ns=hop.c0;
}
FlowResult EvaluateFlow(FlowProblem const& p,FlowOptions const& options) {
  if(p.workers<=0 || !(p.dram_gbps>0))throw std::invalid_argument("invalid flow resources");
  int n=p.spaces.size();long total=0;for(auto const& s:p.spaces){if(s.count<0 || s.piece_of_task.size()!=std::size_t(s.count))throw std::invalid_argument("flow piece coverage");total+=s.count;}
  long workers=options.infinite_workers?total:p.workers,free=workers,finished=0;
  std::vector<std::vector<int>> incoming(n),outgoing(n),pending(n),last_edge(n);
  std::vector<std::vector<unsigned char>> completed(n);
  std::vector<int> prefix(n,-1),launched(n);std::vector<std::size_t> edge_cursor(p.edges.size());
  std::vector<bool> waits(n),all_waits(n),publishes(n);
  for(std::size_t e=0;e<p.edges.size();++e){auto const& r=p.edges[e];if(!r.sorted || r.producer<0 || r.producer>=n || r.consumer<0 || r.consumer>=n)throw std::invalid_argument("invalid flow edge");
    outgoing[r.producer].push_back(e);incoming[r.consumer].push_back(e);
    bool sync=!(r.colocated && r.kappa==1);publishes[r.producer]=publishes[r.producer] || sync;
    waits[r.consumer]=waits[r.consumer] || (sync && !r.all_producer);all_waits[r.consumer]=all_waits[r.consumer] || (sync && r.all_producer);
  }
  struct Ready {int task;double time;};std::vector<std::deque<Ready>> ready(n);
  for(int s=0;s<n;++s){pending[s].assign(p.spaces[s].count,0);last_edge[s].assign(p.spaces[s].count,-1);completed[s].resize(p.spaces[s].count);}
  for(auto const& edge:p.edges)for(auto const& [h,j]:*edge.sorted) {
    if(h<0 || h>=p.spaces[edge.producer].count || j<0 || j>=p.spaces[edge.consumer].count)throw std::invalid_argument("release coordinate outside task domain");
    ++pending[edge.consumer][j];
  }
  for(int s=0;s<n;++s)for(int j=0;j<p.spaces[s].count;++j)if(pending[s][j]==0)ready[s].push_back({j,0});
  enum Kind {Arrival,MainStart,ComputeEnd,PublishEnd};
  struct Event {double time;Kind kind;int id;std::size_t begin=0,end=0;std::uint64_t serial=0;};
  struct Later {bool operator()(Event const& a,Event const& b) const{return std::tie(a.time,a.serial)>std::tie(b.time,b.serial);}};
  std::priority_queue<Event,std::vector<Event>,Later> events;std::uint64_t serial=0;
  auto push=[&](double time,Kind kind,int id,std::size_t begin=0,std::size_t end=0){events.push({time,kind,id,begin,end,serial++});};
  struct Cohort {int space,piece;std::vector<int> tasks;double start=0,main=0,wait=0,fixed=0;bool compute=false,bytes=false,closing=false;};
  std::vector<Cohort> cohorts;std::vector<int> fluid_owner;DramFluidServer fluid(p.dram_gbps);double now=0;
  FlowResult result;result.spaces.resize(n);
  auto close=[&](int id){auto& c=cohorts[id];if(c.compute && c.bytes && !c.closing){c.closing=true;auto& s=result.spaces[c.space];s.mainloop_ns+=(now-c.main)*c.tasks.size();push(now+(!options.no_sync && publishes[c.space]?p.publication_ns:0),PublishEnd,id);}};
  while(finished<total) {
    while(!events.empty() && events.top().time<=now) {
      auto event=events.top();events.pop();
      if(event.kind==Arrival) {
        auto const& edge=p.edges[event.id];for(auto i=event.begin;i<event.end;++i){int j=edge.sorted->at(i).second;if(j<0 || j>=p.spaces[edge.consumer].count)throw std::runtime_error("release consumer out of bounds");
          last_edge[edge.consumer][j]=event.id;if(--pending[edge.consumer][j]==0)ready[edge.consumer].push_back({j,now});}
      } else {
        auto& c=cohorts[event.id];auto const& parts=p.spaces[c.space].pieces[c.piece].parts;
        if(event.kind==MainStart) {
          c.main=now;double bytes=parts.dram_bytes-(options.no_external?parts.no_producer_dram_bytes:0);
          if(bytes<0)throw std::runtime_error("negative counterfactual traffic");
          if(bytes>0){int id=fluid.Add(bytes,parts.dram_rate_cap,c.tasks.size());if(id!=int(fluid_owner.size()))throw std::runtime_error("fluid id discontinuity");fluid_owner.push_back(event.id);}else c.bytes=true;
          push(now+parts.compute_ns,ComputeEnd,event.id);
        }else if(event.kind==ComputeEnd){c.compute=true;close(event.id);}
        else {
          int s=c.space;free+=c.tasks.size();finished+=c.tasks.size();for(int j:c.tasks)completed[s][j]=1;
          auto& stats=result.spaces[s];stats.last_end=now;stats.last_edge=last_edge[s][c.tasks.back()];stats.publication_ns+=(!options.no_sync && publishes[s]?p.publication_ns:0)*c.tasks.size();
          int before=prefix[s];while(prefix[s]+1<p.spaces[s].count && completed[s][prefix[s]+1])++prefix[s];
          if(prefix[s]!=before)for(int e:outgoing[s]){auto const& edge=p.edges[e];auto begin=edge_cursor[e];while(edge_cursor[e]<edge.sorted->size() && edge.sorted->at(edge_cursor[e]).first<=prefix[s])++edge_cursor[e];
            if(begin!=edge_cursor[e])push(now+(options.no_sync || (edge.colocated && edge.kappa==1)?0:p.hop_ns),Arrival,e,begin,edge_cursor[e]);}
        }
      }
    }
    while(free>0) {
      int selected=-1;for(int s=0;s<n;++s)if(!ready[s].empty()) {
        if(selected<0 || std::make_tuple(ready[s].front().time,-p.spaces[s].rank_ns,p.spaces[s].order)<std::make_tuple(ready[selected].front().time,-p.spaces[selected].rank_ns,p.spaces[selected].order))selected=s;
      }
      if(selected<0)break;int s=selected,pi=p.spaces[s].piece_of_task.at(ready[s].front().task);auto const& parts=p.spaces[s].pieces.at(pi).parts;
      bool wait=!options.no_sync && (waits[s] || (all_waits[s] && launched[s]<std::min<long>(p.spaces[s].count,workers)));
      Cohort cohort;cohort.space=s;cohort.piece=pi;cohort.start=now;cohort.wait=wait?p.consumer_wait_ns:0;cohort.fixed=options.no_fixed?0:parts.fixed_ns;
      while(free>0 && !ready[s].empty() && p.spaces[s].piece_of_task[ready[s].front().task]==pi) {
        bool next_wait=!options.no_sync && (waits[s] || (all_waits[s] && launched[s]<std::min<long>(p.spaces[s].count,workers)));
        if(next_wait!=wait)break;cohort.tasks.push_back(ready[s].front().task);ready[s].pop_front();--free;++launched[s];
      }
      auto& stats=result.spaces[s];if(stats.first_start<0)stats.first_start=now;stats.wait_ns+=cohort.wait*cohort.tasks.size();stats.fixed_ns+=cohort.fixed*cohort.tasks.size();
      int id=cohorts.size();cohorts.push_back(std::move(cohort));push(now+cohorts[id].wait+cohorts[id].fixed,MainStart,id);
    }
    if(!events.empty() && events.top().time<=now)continue;
    if(finished==total)break;
    double next=std::min(events.empty()?std::numeric_limits<double>::infinity():events.top().time,now+fluid.Next());
    if(!std::isfinite(next))throw std::runtime_error("flow deadlock: incomplete release coverage or cycle");
    auto done=fluid.Advance(next-now);now=next;for(int id:done){auto c=fluid_owner[id];cohorts[c].bytes=true;close(c);}
  }
  result.makespan_ns=now;result.delivered_bytes=fluid.Delivered();
  if(p.all_external_miss && !options.no_external && now+1e-6<p.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in StageFlowModel");
  int last=-1;for(int s=0;s<n;++s)if(last<0 || result.spaces[s].last_end>result.spaces[last].last_end)last=s;
  std::vector<bool> seen(n);while(last>=0){if(seen[last])throw std::runtime_error("cyclic flow critical chain");seen[last]=true;result.critical_chain.push_back(last);int e=result.spaces[last].last_edge;last=e<0?-1:p.edges[e].producer;}
  std::reverse(result.critical_chain.begin(),result.critical_chain.end());return result;
}
FlowDecomposition DecomposeFlow(FlowProblem const& p) {
  FlowDecomposition d;d.original=EvaluateFlow(p);FlowOptions o;o.no_sync=true;d.no_sync=EvaluateFlow(p,o).makespan_ns;
  o.no_fixed=true;d.no_fixed=EvaluateFlow(p,o).makespan_ns;o.infinite_workers=true;d.infinite=EvaluateFlow(p,o).makespan_ns;
  o={};o.no_external=true;d.no_external=EvaluateFlow(p,o).makespan_ns;
  d.synchronization=d.original.makespan_ns-d.no_sync;d.fixed=d.no_sync-d.no_fixed;
  d.contention=d.no_fixed-std::max(p.floor_ns,d.infinite);d.chain=std::max(0.,d.infinite-p.floor_ns);
  d.pg_upper_bound=d.original.makespan_ns-std::max(p.floor_ns,d.no_external);return d;
}
} // namespace tilemega::solver
