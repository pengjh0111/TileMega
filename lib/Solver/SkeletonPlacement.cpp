// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonPlacement.h>
#include <tilemega/Solver/CoResidency.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <limits>
#include <tuple>

namespace tilemega::solver {
namespace {
struct Arrival {
  double best=0,second=0,latest_finish=0;int owner=-1,second_owner=-1;
  struct Critical {double time=-1;int node=-1,worker=-1;};
  std::array<Critical,2> critical;
  void ByOwner(double time,int w) {
    if(time>best){if(owner!=w){second=best;second_owner=owner;owner=w;}best=time;}
    else if(w!=owner && time>second){second=time;second_owner=w;}
  }
  void Top(Critical value) {
    if(value.node<0)return;
    for(auto const& x:critical)if(x.node==value.node)return;
    auto before=[](Critical const& a,Critical const& b){return a.time>b.time || (a.time==b.time && a.node<b.node);};
    if(before(value,critical[0])){critical[1]=critical[0];critical[0]=value;}
    else if(before(value,critical[1]))critical[1]=value;
  }
  void Add(double finish,double hop,int w,int node){ByOwner(finish+hop,w);Top({finish+hop,node,w});latest_finish=std::max(latest_finish,finish);}
  void Merge(Arrival const& a) {
    ByOwner(a.best,a.owner);
    if(a.second_owner>=0)ByOwner(a.second,a.second_owner);
    for(auto const& x:a.critical)Top(x);
    latest_finish=std::max(latest_finish,a.latest_finish);
  }
};
struct Ready {
  double est,negative_rank;int stage_order,tile,stage,node;
  auto key() const{return std::tie(est,negative_rank,stage_order,tile);}
  bool operator<(Ready const& other) const{return key()>other.key();}
};
}
bool ScheduleBySkeleton(SkeletonRequest const& request,EftSchedule* out,
    SkeletonPlacementStats* statistics,std::string* error) {
  auto fail=[&](std::string const& why){if(error)*error=why;return false;};
  if(!out || !request.skeleton)return fail("missing Skeleton schedule request");
  auto const& skeleton=*request.skeleton;int grid=skeleton.grid;
  int nodes=int(skeleton.task_ns.size()),spaces=int(skeleton.spaces.size());
  if(grid<1 || spaces<1)return fail("empty resident grid or task spaces");
  for(double time:skeleton.task_ns)if(!std::isfinite(time)||time<0)return fail("invalid task duration");
  if(!request.task_lanes.empty() && request.task_lanes.size()!=std::size_t(nodes))return fail("lane count mismatch");
  SkeletonPlacementStats stats;
  out->worker.assign(nodes,-1);out->slot.assign(nodes,-1);out->start_ns.assign(nodes,0);out->end_ns.assign(nodes,0);out->makespan_ns=0;
  std::vector<int> sm(grid),stage_of(nodes),next_slot(grid,0),last(grid,-1),completed(spaces,0);
  int sms=request.sms>0?request.sms:grid;
  for(int w=0;w<int(sm.size());++w)sm[w]=request.worker_sm.empty()?w%sms:request.worker_sm.at(w);
  std::vector<std::vector<int>> siblings(sms);
  for(int w=0;w<int(sm.size());++w){if(sm[w]<0 || sm[w]>=sms)return fail("invalid SM map");siblings[sm[w]].push_back(w);}
  if(request.ctas_per_sm>0)for(auto const& row:siblings)if(int(row.size())>request.ctas_per_sm)return fail("grid exceeds residency");
  // Only this task-space DAG supplies rank; no tile adjacency is built.
  std::vector<int> indegree(spaces,0),topo;
  for(auto const& edge:skeleton.edges)++indegree[edge.consumer];
  std::queue<int> space_ready;for(int s=0;s<spaces;++s)if(!indegree[s])space_ready.push(s);
  while(!space_ready.empty()){int s=space_ready.front();space_ready.pop();topo.push_back(s);
    for(int e:skeleton.outgoing[s])if(--indegree[skeleton.edges[e].consumer]==0)space_ready.push(skeleton.edges[e].consumer);}
  if(int(topo.size())!=spaces)return fail("task-space dependency cycle");
  std::vector<double> rank(spaces),hop(nodes,0),avail(grid,0);
  for(auto it=topo.rbegin();it!=topo.rend();++it){int s=*it;double tail=0;
    for(int e:skeleton.outgoing[s])tail=std::max(tail,request.hop.Ns(1,1)+rank[skeleton.edges[e].consumer]);
    rank[s]=skeleton.spaces[s].task_ns+tail;}
  std::vector<long> pending(nodes,-1);std::vector<Arrival> arrivals(nodes);
  std::vector<bool> arrival_ready(nodes,false),placed(nodes,false);
  // Per-space owner maxima make an all-to-all edge O(Np + Nc), not O(Np*Nc).
  std::vector<Arrival> space_arrivals(spaces);
  std::priority_queue<Ready> ready;
  auto incoming_count=[&](int s,int t){long n=0;for(int index:skeleton.incoming[s]){auto const& edge=skeleton.edges[index];
    n+=edge.all_producer?skeleton.spaces[edge.producer].count:edge.oracle->reverse.Query({t},skeleton.theta).Count();}return n;};
  for(int s=0;s<spaces;++s){auto const& space=skeleton.spaces[s];for(int t=0;t<space.count;++t){int node=space.offset+t;
    stage_of[node]=s;pending[node]=incoming_count(s,t);
    if(pending[node]==0)ready.push({0,-rank[s],space.order,t,s,node});}}
  auto arrival=[&](int s,int t)->Arrival const& {
    int node=skeleton.spaces[s].offset+t;if(arrival_ready[node])return arrivals[node];auto& result=arrivals[node];
    for(int index:skeleton.incoming[s]){auto const& edge=skeleton.edges[index];int p=edge.producer;
      if(edge.all_producer){
        result.Merge(space_arrivals[p]);
      }else edge.oracle->reverse.Query({t},skeleton.theta).ForEach([&](auto const& coordinate){
        int pred=skeleton.spaces[p].offset+int(coordinate.at(0));
        if(pred<0 || pred>=nodes || !placed[pred])throw std::logic_error("ready task has an unplaced predecessor");
        result.Add(out->end_ns[pred],hop[pred],out->worker[pred],pred);
      });
    }
    arrival_ready[node]=true;return result;
  };
  try {
    std::vector<int> candidates,members;
    while(!ready.empty()) {
      Ready task=ready.top();ready.pop();auto const& a=arrival(task.stage,task.tile);
      skeleton.Spread(task.stage,task.tile,candidates);
      // Spread is injective: (k-1)*floor(W/k) < W. Only the two
      // affinity workers can duplicate it; worker ties are explicit below.
      for(auto const& p:a.critical)if(p.node>=0 &&
          std::find(candidates.begin(),candidates.end(),p.worker)==candidates.end())candidates.push_back(p.worker);
      double est=std::numeric_limits<double>::infinity(),chosen_start=0,chosen_end=est;int chosen=-1;
      for(int w:candidates) {
        double start=std::max(avail[w],w==a.owner?a.second:a.best);est=std::min(est,start);
        double stretch=1;
        if(siblings[sm[w]].size()>1) {
          members.clear();members.push_back(task.node);
          for(int sibling:siblings[sm[w]])if(sibling!=w && last[sibling]>=0 && out->end_ns[last[sibling]]>start)members.push_back(last[sibling]);
          if(members.size()>1)stretch=std::max(1.0,request.task_lanes.empty()?double(members.size()):LaneStretch(request.task_lanes,members));
        }
        double finish=start+skeleton.task_ns[task.node]*stretch;
        if(chosen<0 || finish<chosen_end || (finish==chosen_end && w<chosen)){chosen=w;chosen_start=start;chosen_end=finish;}
      }
      task.est=est;
      if(!ready.empty() && task.key()>ready.top().key()){ready.push(task);++stats.lazy_requeues;continue;}
      int node=task.node;
      out->worker[node]=chosen;out->slot[node]=next_slot[chosen]++;out->start_ns[node]=chosen_start;out->end_ns[node]=chosen_end;
      if(last[chosen]>=0){++stats.adjacent_slots;if(stage_of[last[chosen]]!=task.stage)++stats.transitions;}
      last[chosen]=node;avail[chosen]=chosen_end;placed[node]=true;++stats.placed;
      stats.candidate_sum+=candidates.size();
      int home=(skeleton.spaces[task.stage].base+task.tile)%grid;
      if(std::any_of(a.critical.begin(),a.critical.end(),[&](auto const& p){return p.node>=0 && p.worker==chosen;}))++stats.affinity;
      else if(chosen==home)++stats.home;else ++stats.spread_other;
      out->makespan_ns=std::max(out->makespan_ns,chosen_end);
      long fanout=0;for(int index:skeleton.outgoing[task.stage]){auto const& edge=skeleton.edges[index];
        fanout+=edge.all_producer?skeleton.spaces[edge.consumer].count:edge.oracle->forward.Query({task.tile},skeleton.theta).Count();}
      hop[node]=fanout?request.hop.Ns(int(fanout),1):0;
      space_arrivals[task.stage].Add(chosen_end,hop[node],chosen,node);
      ++completed[task.stage];
      for(int index:skeleton.outgoing[task.stage]) {auto const& edge=skeleton.edges[index];
        if(edge.all_producer && completed[task.stage]!=skeleton.spaces[task.stage].count)continue;
        long amount=edge.all_producer?skeleton.spaces[task.stage].count:1;
        edge.oracle->forward.Query({task.tile},skeleton.theta).ForEach([&](auto const& coordinate){
          int s=edge.consumer,t=int(coordinate.at(0)),n=skeleton.spaces[s].offset+t;
          if(t<0 || t>=skeleton.spaces[s].count)throw std::runtime_error("Oracle successor outside task domain");
          pending[n]-=amount;if(pending[n]<0)throw std::runtime_error("duplicate dependency completion");
          if(pending[n]==0){auto const& a=arrival(s,t);ready.push({a.latest_finish,-rank[s],skeleton.spaces[s].order,t,s,n});}
        });
      }
    }
  }catch(std::exception const& e){return fail(e.what());}
  if(stats.placed!=std::uint64_t(nodes))return fail("unplaced tasks remain after readiness exhausted");
  stats.interleaving=stats.adjacent_slots?double(stats.transitions)/stats.adjacent_slots:0;
  if(statistics)*statistics=stats;return true;
}
}
