// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Solver/DramFluid.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
namespace tilemega::solver {
int CoarsenRelease(int maximum,int n,int kappa){if(maximum<0 || maximum>=n || kappa<1)throw std::invalid_argument("invalid release domain");return std::min(n-1,kappa*(maximum/kappa)+kappa-1);}
void SetFlowCalibration(FlowProblem& p,TargetSpec const& target,ScalarType dtype,HopCurve const& hop) {
  auto name=dtype==ScalarType::kBF16?"bf16":"f32";auto const& event=target.EventCalibrationFor(name);
  if(!event.task_publication.ns || !event.task_wait.ns)throw std::invalid_argument("flow synchronization calibration missing");
  p.dram_gbps=target.CalibrationFor(name).dram_gbps;p.publication_ns=*event.task_publication.ns;p.consumer_wait_ns=*event.task_wait.ns;p.hop_ns=hop.c0;
  auto const& cal=target.CalibrationFor(name);
  if(p.inflight_dram) {
    p.inflight_curve_bytes=cal.inflight_curve_bytes;
    p.inflight_curve_gbps=cal.inflight_curve_gbps;
    p.cta_stream_curve_bytes=cal.cta_stream_curve_bytes;
    p.cta_stream_curve_gbps=cal.cta_stream_curve_gbps;
  }
}
FlowResult EvaluateFlow(FlowProblem const& p,FlowOptions const& options) {
  if(p.workers<=0 || !(p.dram_gbps>0))throw std::invalid_argument("invalid flow resources");
  bool const paged=p.page_bytes>0 || p.pages_per_worker>0;
  if(paged && (p.page_bytes<=0 || p.pages_per_worker<=0))
    throw std::invalid_argument("paged flow needs a positive page size and ring capacity");
  int n=p.spaces.size();long total=0;for(auto const& s:p.spaces){if(s.count<0 || s.piece_of_task.size()!=std::size_t(s.count))throw std::invalid_argument("flow piece coverage");total+=s.count;}
  long workers=options.infinite_workers?total:p.workers,free=workers,finished=0;
  std::vector<std::vector<int>> incoming(n),outgoing(n);
  struct Scratch {std::vector<std::vector<int>> pending,gate_pending,active_cohort,last_edge,last_cause;
    std::vector<std::vector<double>> gate_tail;std::vector<std::vector<unsigned char>> completed;};
  static thread_local Scratch scratch;
  auto& pending=scratch.pending;auto& gate_pending=scratch.gate_pending;
  auto& active_cohort=scratch.active_cohort;auto& gate_tail=scratch.gate_tail;
  auto& last_edge=scratch.last_edge;auto& last_cause=scratch.last_cause;auto& completed=scratch.completed;
  pending.resize(n);gate_pending.resize(n);active_cohort.resize(n);gate_tail.resize(n);
  last_edge.resize(n);last_cause.resize(n);completed.resize(n);
  std::vector<int> prefix(n,-1),launched(n);
  std::vector<std::size_t> edge_cursor(p.edges.size()),first_cursor(p.edges.size());
  std::vector<bool> waits(n),all_waits(n),publishes(n);
  for(std::size_t e=0;e<p.edges.size();++e){auto const& r=p.edges[e];if(!r.sorted || r.producer<0 || r.producer>=n || r.consumer<0 || r.consumer>=n ||
      (r.phase && (!r.first || r.phase_iterations<1)))throw std::invalid_argument("invalid flow edge");
    outgoing[r.producer].push_back(e);incoming[r.consumer].push_back(e);
    bool sync=!(r.colocated && r.kappa==1) && !p.spaces[r.consumer].fused_reducer;
    publishes[r.producer]=publishes[r.producer] || sync;
    waits[r.consumer]=waits[r.consumer] || (sync && !r.all_producer);all_waits[r.consumer]=all_waits[r.consumer] || (sync && r.all_producer);
  }
  struct Ready {int task;double time;};std::vector<std::deque<Ready>> ready(n);
  std::vector<std::vector<unsigned char>> prefetched(n);
  std::vector<std::vector<double>> dependency_ready(n),page_hold(n),prefetch_bytes(n);
  for(int s=0;s<n;++s){pending[s].assign(p.spaces[s].count,0);gate_pending[s].assign(p.spaces[s].count,0);
    active_cohort[s].assign(p.spaces[s].count,-1);gate_tail[s].assign(p.spaces[s].count,0);
    last_edge[s].assign(p.spaces[s].count,-1);last_cause[s].assign(p.spaces[s].count,-1);completed[s].assign(p.spaces[s].count,0);}
  for(auto const& edge:p.edges)for(auto const& [h,j]:*edge.sorted) {
    if(h<0 || h>=p.spaces[edge.producer].count || j<0 || j>=p.spaces[edge.consumer].count)throw std::invalid_argument("release coordinate outside task domain");
    if(edge.phase)++gate_pending[edge.consumer][j];
    else ++pending[edge.consumer][j];
  }
  for(auto const& edge:p.edges)if(edge.phase)for(auto const& [h,j]:*edge.first) {
    if(h<0 || h>=p.spaces[edge.producer].count || j<0 || j>=p.spaces[edge.consumer].count)
      throw std::invalid_argument("first phase release outside task domain");
    ++pending[edge.consumer][j];
  }
  for(int s=0;s<n;++s) {
    prefetched[s].assign(p.spaces[s].count,!paged);
    dependency_ready[s].assign(p.spaces[s].count,0);
    page_hold[s].assign(p.spaces[s].count,0);
    prefetch_bytes[s].assign(p.spaces[s].count,0);
    for(int j=0;j<p.spaces[s].count;++j)
      if(pending[s][j]==0 && !paged)ready[s].push_back({j,0});
  }
  using SpaceKey=std::tuple<double,double,int,int>;
  std::priority_queue<SpaceKey,std::vector<SpaceKey>,std::greater<SpaceKey>> ready_spaces;
  std::priority_queue<SpaceKey,std::vector<SpaceKey>,std::greater<SpaceKey>> fused_spaces;
  auto queue_space=[&](int s){
    auto& queue=p.spaces[s].fused_reducer?fused_spaces:ready_spaces;
    queue.emplace(ready[s].front().time,-p.spaces[s].rank_ns,p.spaces[s].order,s);
  };
  for(int s=0;s<n;++s)if(!ready[s].empty())queue_space(s);
  enum Kind {Arrival,FirstArrival,GateArrival,MainStart,ComputeEnd,PublishEnd};
  struct Event {double time;Kind kind;int id;std::size_t begin=0,end=0;std::uint64_t serial=0;int cause=-1;};
  struct Later {bool operator()(Event const& a,Event const& b) const{return std::tie(a.time,a.serial)>std::tie(b.time,b.serial);}};
  std::priority_queue<Event,std::vector<Event>,Later> events;std::uint64_t serial=0;
  auto push=[&](double time,Kind kind,int id,std::size_t begin=0,std::size_t end=0,int cause=-1){events.push({time,kind,id,begin,end,serial++,cause});};
  struct Cohort {int space,piece,cause=-1,edge=-1;std::size_t begin=0,count=0;double start=0,main=0,wait=0,fixed=0,end=0,publication=0;bool compute=false,bytes=false,closing=false,fused=false;};
  static thread_local std::vector<Cohort> cohorts;cohorts.clear();
  static thread_local std::vector<int> cohort_tasks;cohort_tasks.clear();cohort_tasks.reserve(total);
  std::vector<int> fluid_owner;DramFluidServer fluid(p.dram_gbps);
  struct PrefetchCohort {int space,first,count;double bytes,reserved;};
  std::vector<PrefetchCohort> prefetch_cohorts;
  std::vector<int> prefetch_spaces(n);std::iota(prefetch_spaces.begin(),prefetch_spaces.end(),0);
  std::stable_sort(prefetch_spaces.begin(),prefetch_spaces.end(),[&](int a,int b){
    return p.spaces[a].order<p.spaces[b].order;});
  std::size_t prefetch_space=0;int prefetch_task=0;
  // A page is owned by one CTA.  A device-wide sum of free pages lets an
  // unrelated CTA prefetch into a full ring and made PG-1 look much faster
  // than the executor.  The Level-1 home is the L1 grid-stride owner; Level-2
  // materialization is checked separately by the FIFO fluid simulator.
  std::vector<double> held_pages(workers,0);
  double const per_worker_capacity=double(p.page_bytes)*p.pages_per_worker+
      p.lookahead_bytes;
  std::optional<InflightDramServer> inflight;
  if(p.inflight_dram)inflight.emplace(p.dram_gbps,p.inflight_curve_bytes,
      p.inflight_curve_gbps,p.cta_stream_curve_bytes,p.cta_stream_curve_gbps);
  auto fluid_next=[&](){return inflight?inflight->Next():fluid.Next();};
  auto fluid_advance=[&](double dt){return inflight?inflight->Advance(dt):fluid.Advance(dt);};
  double now=0;
  FlowResult result;result.spaces.resize(n);int last_completed=-1;
  auto mark_prefetched=[&](int s,int j,double at){
    prefetched[s][j]=1;
    if(pending[s][j]==0) {
      bool empty=ready[s].empty();
      ready[s].push_back({j,std::max(at,dependency_ready[s][j])});
      if(empty)queue_space(s);
    }
  };
  auto launch_prefetch=[&](){
    if(!paged)return;
    while(prefetch_space<prefetch_spaces.size()) {
      int s=prefetch_spaces[prefetch_space];
      if(prefetch_task>=p.spaces[s].count){++prefetch_space;prefetch_task=0;continue;}
      int j=prefetch_task,pi=p.spaces[s].piece_of_task[j];
      auto const& parts=p.spaces[s].pieces[pi].parts;
      double bytes=options.no_external?0:std::min(parts.no_producer_dram_bytes,per_worker_capacity);
      if(bytes<0)throw std::runtime_error("negative no-producer prefetch bytes");
      double reserve=bytes>0?std::ceil(bytes/p.page_bytes)*p.page_bytes:0;
      if(reserve>per_worker_capacity)throw std::runtime_error("prefetch exceeds a worker page ring");
      int count=0;
      if(reserve>0) {
        while(j+count<p.spaces[s].count &&
              p.spaces[s].piece_of_task[j+count]==pi &&
              held_pages[(j+count)%workers]+
                  (count/workers+1)*reserve<=per_worker_capacity+1e-6)
          ++count;
        if(count==0)break;
      }else {
        count=1;
        while(j+count<p.spaces[s].count && p.spaces[s].piece_of_task[j+count]==pi)++count;
      }
      for(int at=j;at<j+count;++at){
        page_hold[s][at]=reserve;prefetch_bytes[s][at]=bytes;
        held_pages[at%workers]+=reserve;
      }
      prefetch_task+=count;
      if(bytes==0)for(int at=j;at<j+count;++at)mark_prefetched(s,at,now);
      else {
        int id=inflight?inflight->Add(bytes,parts.dram_rate_cap,
            std::max(16.,std::min(parts.inflight_bytes,per_worker_capacity)),count)
            :fluid.Add(bytes,parts.dram_rate_cap,count);
        if(id!=int(fluid_owner.size()))throw std::runtime_error("prefetch fluid id discontinuity");
        int index=prefetch_cohorts.size();prefetch_cohorts.push_back({s,j,count,bytes,reserve});
        fluid_owner.push_back(-index-1);
      }
    }
  };
  auto close=[&](int id){auto& c=cohorts[id];if(c.compute && c.bytes && !c.closing){
    double gated=now;
    for(std::size_t i=c.begin;i<c.begin+c.count;++i) {
      int task=cohort_tasks[i];
      if(gate_pending[c.space][task])return;
      gated=std::max(gated,gate_tail[c.space][task]);
    }
    c.closing=true;
    // A prefetched page is held until its consumer finishes its mainloop.
    // Releasing the entire task at MainStart allowed arbitrarily many future
    // tasks to refill a ring whose current task had not read a single page.
    if(paged)for(std::size_t i=c.begin;i<c.begin+c.count;++i){
      int task=cohort_tasks[i];held_pages[task%workers]-=page_hold[c.space][task];
      page_hold[c.space][task]=0;
    }
    auto& s=result.spaces[c.space];s.mainloop_ns+=(now-c.main)*c.count;
    push(gated+(!options.no_sync && publishes[c.space]?p.publication_ns:0),PublishEnd,id);
  }};
  while(finished<total) {
    while(!events.empty() && events.top().time<=now) {
      auto event=events.top();events.pop();
      if(event.kind==Arrival || event.kind==FirstArrival || event.kind==GateArrival) {
        auto const& edge=p.edges[event.id];for(auto i=event.begin;i<event.end;++i){
          int j=(event.kind==FirstArrival?edge.first:edge.sorted)->at(i).second;
          if(j<0 || j>=p.spaces[edge.consumer].count)throw std::runtime_error("release consumer out of bounds");
          if(event.kind==GateArrival) {
            gate_tail[edge.consumer][j]=std::max(gate_tail[edge.consumer][j],
                now+p.spaces[edge.consumer].pieces[p.spaces[edge.consumer].piece_of_task[j]].parts.compute_ns/
                    edge.phase_iterations);
            if(--gate_pending[edge.consumer][j]<0)throw std::runtime_error("phase gate over-release");
            int id=active_cohort[edge.consumer][j];
            if(id>=0)close(id);
            continue;
          }
          last_edge[edge.consumer][j]=event.id;last_cause[edge.consumer][j]=event.cause;if(--pending[edge.consumer][j]==0){
            dependency_ready[edge.consumer][j]=now;
            if(prefetched[edge.consumer][j]){bool empty=ready[edge.consumer].empty();ready[edge.consumer].push_back({j,now});if(empty)queue_space(edge.consumer);}}}
      } else {
        auto& c=cohorts[event.id];auto const& parts=p.spaces[c.space].pieces[c.piece].parts;
        if(event.kind==MainStart) {
          c.main=now;double prefetched_total=0;
          if(paged)for(std::size_t i=c.begin;i<c.begin+c.count;++i){int task=cohort_tasks[i];
            prefetched_total+=prefetch_bytes[c.space][task];}
          double bytes=parts.dram_bytes-(options.no_external?parts.no_producer_dram_bytes:0)
              -(paged?prefetched_total/c.count:0);
          if(bytes<0)throw std::runtime_error("negative counterfactual traffic");
          if(bytes>0){int id=inflight?inflight->Add(bytes,parts.dram_rate_cap,
              parts.inflight_bytes,c.count):fluid.Add(bytes,parts.dram_rate_cap,c.count);
            if(id!=int(fluid_owner.size()))throw std::runtime_error("fluid id discontinuity");fluid_owner.push_back(event.id);}else c.bytes=true;
          push(now+parts.compute_ns,ComputeEnd,event.id);
        }else if(event.kind==ComputeEnd){c.compute=true;close(event.id);}
        else {
          int s=c.space;c.end=now;c.publication=!options.no_sync && publishes[s]?p.publication_ns:0;last_completed=event.id;if(!c.fused)free+=c.count;finished+=c.count;for(std::size_t i=c.begin;i<c.begin+c.count;++i)completed[s][cohort_tasks[i]]=1;
          auto& stats=result.spaces[s];stats.last_end=now;stats.last_edge=last_edge[s][cohort_tasks[c.begin+c.count-1]];stats.publication_ns+=(!options.no_sync && publishes[s]?p.publication_ns:0)*c.count;
          int before=prefix[s];while(prefix[s]+1<p.spaces[s].count && completed[s][prefix[s]+1])++prefix[s];
          if(prefix[s]!=before)for(int e:outgoing[s]){auto const& edge=p.edges[e];auto begin=edge_cursor[e];while(edge_cursor[e]<edge.sorted->size() && edge.sorted->at(edge_cursor[e]).first<=prefix[s])++edge_cursor[e];
            double hop=options.no_sync || p.spaces[edge.consumer].fused_reducer || (edge.colocated && edge.kappa==1)?0:p.hop_ns;
            if(begin!=edge_cursor[e])push(now+hop,edge.phase?GateArrival:Arrival,e,begin,edge_cursor[e],event.id);
            if(edge.phase) {
              begin=first_cursor[e];
              while(first_cursor[e]<edge.first->size() && edge.first->at(first_cursor[e]).first<=prefix[s])++first_cursor[e];
              if(begin!=first_cursor[e])push(now+hop,FirstArrival,e,begin,first_cursor[e],event.id);
            }}
        }
      }
    }
    launch_prefetch();
    while(free>0 || !fused_spaces.empty()) {
      if(fused_spaces.empty() && ready_spaces.empty())break;
      int s;
      if(!fused_spaces.empty()) {s=std::get<3>(fused_spaces.top());fused_spaces.pop();}
      else {s=std::get<3>(ready_spaces.top());ready_spaces.pop();}
      int pi=p.spaces[s].piece_of_task.at(ready[s].front().task);auto const& parts=p.spaces[s].pieces.at(pi).parts;
      bool fused=p.spaces[s].fused_reducer;
      bool wait=!fused && !options.no_sync && (waits[s] || (all_waits[s] && launched[s]<std::min<long>(p.spaces[s].count,workers)));
      Cohort cohort;cohort.begin=cohort_tasks.size();cohort.space=s;cohort.piece=pi;cohort.start=now;cohort.wait=wait?p.consumer_wait_ns:0;cohort.fixed=fused || options.no_fixed?0:parts.fixed_ns;cohort.fused=fused;
      while((fused || free>0) && !ready[s].empty() && p.spaces[s].piece_of_task[ready[s].front().task]==pi) {
        bool next_wait=!fused && !options.no_sync &&
            (waits[s] || (all_waits[s] && launched[s]<std::min<long>(p.spaces[s].count,workers)));
        if(next_wait!=wait)break;cohort.cause=last_cause[s][ready[s].front().task];cohort.edge=last_edge[s][ready[s].front().task];cohort_tasks.push_back(ready[s].front().task);++cohort.count;ready[s].pop_front();if(!fused)--free;++launched[s];
      }
      if(!ready[s].empty())queue_space(s);
      auto& stats=result.spaces[s];if(stats.first_start<0)stats.first_start=now;stats.wait_ns+=cohort.wait*cohort.count;stats.fixed_ns+=cohort.fixed*cohort.count;
      int id=cohorts.size();cohorts.push_back(std::move(cohort));
      for(std::size_t i=cohorts[id].begin;i<cohorts[id].begin+cohorts[id].count;++i)
        active_cohort[s][cohort_tasks[i]]=id;
      push(now+cohorts[id].wait+cohorts[id].fixed,MainStart,id);
    }
    if(!events.empty() && events.top().time<=now)continue;
    if(finished==total)break;
    double next=std::min(events.empty()?std::numeric_limits<double>::infinity():events.top().time,now+fluid_next());
    if(!std::isfinite(next))throw std::runtime_error("flow deadlock: incomplete release coverage or cycle");
    auto done=fluid_advance(next-now);now=next;for(int id:done){auto owner=fluid_owner[id];
      if(owner<0){auto const& prefetch=prefetch_cohorts[-owner-1];
        for(int j=prefetch.first;j<prefetch.first+prefetch.count;++j)
          mark_prefetched(prefetch.space,j,now);
      }else {cohorts[owner].bytes=true;close(owner);}}
  }
  result.makespan_ns=now;result.delivered_bytes=inflight?inflight->Delivered():fluid.Delivered();
  if(p.all_external_miss && !options.no_external && now+1e-6<p.dram_floor_ns)throw std::runtime_error("T >= T_dram assertion failed in StageFlowModel");
  std::vector<bool> seen(cohorts.size());
  while(last_completed>=0) {
    if(seen[last_completed])throw std::runtime_error("cyclic flow critical chain");seen[last_completed]=true;
    auto const& c=cohorts[last_completed];double hop=0;
    if(c.edge>=0){auto const& e=p.edges[c.edge];if(!options.no_sync && !(e.colocated && e.kappa==1))hop=p.hop_ns;}
    result.critical_chain.push_back(c.space);
    result.critical_links.push_back({c.space,cohort_tasks[c.begin+c.count-1],c.edge,c.start,c.end,c.wait,c.fixed,std::max(0.,c.end-c.publication-c.main),c.publication,hop});
    last_completed=c.cause;
  }
  std::reverse(result.critical_chain.begin(),result.critical_chain.end());
  std::reverse(result.critical_links.begin(),result.critical_links.end());return result;
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
