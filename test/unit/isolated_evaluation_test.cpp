// SPDX-License-Identifier: BSD-3-Clause
#include "../../lib/Solver/IsolatedEvaluation.h"
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/SkeletonPlacement.h>
#include <iomanip>
using namespace tilemega;
int main() {
 try {
  analysis::IslContext isl;analysis::CouplingCache cache;
  solver::PlanSkeleton sk;sk.grid=4;sk.residency=1;sk.stage_order={0,1};
  sk.incoming.resize(2);sk.outgoing.resize(2);
  for(int s=0;s<2;++s)sk.spaces.push_back({8,s*8,0,2,s,3.0,24.0});
  sk.task_ns.assign(16,3);
  auto relation=cache.OracleFor("{ [p] -> [q] : 0<=p<8 and 0<=q<8 and p%2=q%2 }");
  sk.edges.push_back({0,1,relation,false});sk.incoming[1]={0};sk.outgoing[0]={0};
  // Warm the inherited Oracle before exercising mutable child query state.
  relation->forward.Query({0});
  auto run=[&](double hop) {
    solver::SkeletonRequest request;request.skeleton=&sk;request.hop.c0=hop;
    solver::EftSchedule schedule;solver::SkeletonPlacementStats stats;std::string error;
    if(!solver::ScheduleBySkeleton(request,&schedule,&stats,&error))throw std::runtime_error(error);
    std::ostringstream out;out<<std::setprecision(17);
    for(std::size_t t=0;t<schedule.worker.size();++t)
      out<<schedule.worker[t]<<' '<<schedule.slot[t]<<' '<<schedule.start_ns[t]<<' '<<schedule.end_ns[t]<<'\n';
    return out.str();
  };
  auto prefix=std::filesystem::temp_directory_path()/ ("tilemega-isolation-"+std::to_string(getpid()));
  std::vector<std::function<std::string()>> jobs;std::vector<std::string> paths,expected;
  for(int i=0;i<4;++i){expected.push_back(run(i));jobs.push_back([&,i]{return run(i);});paths.push_back((prefix/std::to_string(i)).string());}
  auto queries=relation->forward.queries();auto results=solver::EvaluateIsolated(jobs,paths);
  for(std::size_t i=0;i<results.size();++i)
    if(results[i].status || results[i].payload!=expected[i])throw std::runtime_error("child schedule differs from serial schedule");
  if(relation->forward.queries()!=queries)throw std::runtime_error("child mutated parent Oracle state");
  auto errors=solver::EvaluateIsolated({[]{throw std::runtime_error("intentional test failure");return std::string();}}, {(prefix/"error").string()});
  if(errors.front().status==0)throw std::runtime_error("worker exception was lost");
  std::filesystem::remove_all(prefix);
  std::cout<<"ISOLATED_EVALUATION exact_schedules=4 parent_state_unchanged=1 exception_propagated=1 PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
