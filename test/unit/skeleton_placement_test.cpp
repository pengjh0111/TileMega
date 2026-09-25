// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/SkeletonPlacement.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <iostream>
#include <cstring>
#include <numeric>
#include <stdexcept>
using namespace tilemega;
solver::MaterializedPlan materialize(solver::PlanSkeleton const& sk,solver::EftSchedule const& schedule,
    codegen::RuntimeTaskGraph const& graph) {
  solver::PlanRequest r;r.mode=dialect::PlacementMode::kEft;r.grid=sk.grid;r.graph=&graph;
  r.eft_worker=schedule.worker;r.eft_slot=schedule.slot;r.physical_worker.resize(sk.grid);std::iota(r.physical_worker.begin(),r.physical_worker.end(),0);
  for(auto const& s:sk.spaces)r.counts.push_back(s.count);
  for(int s:sk.stage_order)r.stage_order.push_back(s);
  solver::MaterializedPlan p;std::string error;
  if(!solver::MaterializePlanPlacement(r,&p,&error) || !solver::CheckPlanLegality(graph,p,&error))throw std::runtime_error(error);
  return p;
}
int main() {
 try {
  analysis::IslContext isl;analysis::CouplingCache cache;
  for(int scenario=0;scenario<3;++scenario) {
    solver::PlanSkeleton sk;sk.grid=4;sk.residency=1;sk.stage_order={0,1,2};sk.incoming.resize(3);sk.outgoing.resize(3);
    for(int s=0;s<3;++s)sk.spaces.push_back({8,s*8,0,2,s,s==0?25.5:1,s==0?204.0:8.0});
    sk.task_ns.resize(24,1);for(int t=0;t<8;++t)sk.task_ns[t]=t%2?1:50;
    for(int s=0;s<2;++s) {
      std::string rel=scenario==0?"{ [p] -> [q] : 0<=p<8 and q=p }":
        scenario==1?"{ [p] -> [q] : 0<=p<8 and 0<=q<8 }":
                    "{ [p] -> [q] : 0<=p<8 and 0<=q<8 and p%2=q%2 }";
      sk.edges.push_back({s,s+1,cache.OracleFor(rel),scenario==1});sk.incoming[s+1].push_back(s);sk.outgoing[s].push_back(s);
    }
    auto graph=codegen::MaterializeRuntimeTaskGraph({8,8,8},{},4);
    for(auto const& edge:sk.edges)for(int t=0;t<8;++t)
      edge.oracle->forward.Query({t}).ForEach([&](auto const& q){graph.successors[sk.spaces[edge.producer].offset+t].push_back(sk.spaces[edge.consumer].offset+q[0]);});
    solver::SkeletonRequest request;request.skeleton=&sk;request.hop.c0=2;
    solver::EftSchedule schedule;solver::SkeletonPlacementStats stats;std::string error;
    if(!solver::ScheduleBySkeleton(request,&schedule,&stats,&error))throw std::runtime_error(error);
    auto p=materialize(sk,schedule,graph);
    for(std::size_t from=0;from<graph.successors.size();++from)for(int to:graph.successors[from])
      if(schedule.start_ns[to]<schedule.end_ns[from]+(schedule.worker[from]==schedule.worker[to]?0:2))throw std::runtime_error("dependency started early");
    if(scenario==0) {
      auto first_consumer=*std::min_element(schedule.start_ns.begin()+8,schedule.start_ns.end());
      auto last_producer=*std::max_element(schedule.start_ns.begin(),schedule.start_ns.begin()+8);
      if(!(first_consumer<last_producer))throw std::runtime_error("ready consumers wait for the upstream stage");
    }
    if(stats.affinity+stats.home+stats.spread_other!=24 || stats.candidate_sum>24*4)throw std::runtime_error("candidate accounting mismatch");
    std::uint64_t moved=0;
    for(int s=0;s<3;++s)for(int t=0;t<8;++t)moved+=schedule.worker[s*8+t]!=sk.Home(s,t);
    if(moved!=stats.moved_from_home)throw std::runtime_error("home displacement mismatch");
    std::uint64_t digest=1469598103934665603ull;
    for(std::size_t n=0;n<schedule.worker.size();++n) {
      for(auto value:{double(schedule.worker[n]),double(schedule.slot[n]),schedule.start_ns[n],schedule.end_ns[n]}) {
        std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));digest=(digest^bits)*1099511628211ull;
      }
    }
    std::cout<<"SCHEDULE_DIGEST case="<<scenario<<" value="<<digest<<'\n';
    std::cout<<"READY_PLACEMENT case="<<scenario<<" makespan="<<schedule.makespan_ns<<" interleaving="<<stats.interleaving<<" requeues="<<stats.lazy_requeues<<" PASS\n";
    request.pure_template=true;
    if(!solver::ScheduleBySkeleton(request,&schedule,&stats,&error) || stats.moved_from_home!=0)
      throw std::runtime_error("pure template moved a task from home");
    std::cout<<"TEMPLATE_DISPLACEMENT case="<<scenario<<" moved="<<stats.moved_from_home<<" affinity="<<stats.affinity<<" PASS\n";
  }
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
