// SPDX-License-Identifier: BSD-3-Clause
// Independently compare Coarsen's scalar release endpoint with the events
// requested by the unchanged runtime projection before local poll omission.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <iostream>
#include <set>
using namespace tilemega;
int main() try {
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SOLVER_R9B/fit/target.json");
  long checks=0,mismatches=0;
  for(auto model:{"gqa2","mha4"})for(int seq:{4,128})for(int kappa:{1,2,4}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SEQSCAN/raw/export/"+model+".json",context);
    auto problem=solver::PrepareSymbolicProblem(*module,target,{seq,3,seq+3},128,1,kappa,nullptr,false);
    auto theta=problem.model.MetricBindings();
    auto dependencies=problem.projection.dependencies.BindParams(theta);
    auto requested=problem.projection.requested_events.BindParams(theta);
    for(int stage=0;stage<int(problem.counts.size());++stage) {
      int n=problem.counts[stage];if(!n)continue;
      for(int task:std::set<int>{0,n/2,n-1}) {
        auto fiber="{ [cs,c] : cs="+std::to_string(stage)+" and c="+std::to_string(task)+" }";
        std::map<int,int> maximum,events;
        for(auto const& [consumer,producer]:dependencies.IntersectDomain(fiber).Points())
          maximum[producer[0]]=std::max(maximum.count(producer[0])?maximum[producer[0]]:-1,int(producer[1]));
        for(auto const& [consumer,event]:requested.IntersectDomain(fiber).Points()) {
          int producer=event[1],kind=event[2],group=event[3];
          int end=kind==0?problem.counts[producer]-1:kind==2?group:
              std::min(problem.counts[producer]-1,kappa*(group+1)-1);
          events[producer]=std::max(events.count(producer)?events[producer]:-1,end);
        }
        for(auto const& [producer,last]:maximum) {
          int flow=solver::CoarsenRelease(last,problem.counts[producer],kappa);++checks;
          if(!events.count(producer) || flow!=events[producer]) {
            ++mismatches;std::cout<<"RELEASE_DIFFERENCE model="<<model<<" seq="<<seq<<" kappa="<<kappa<<" consumer="<<stage<<','<<task<<" producer="<<producer<<" flow="<<flow<<" runtime="<<(events.count(producer)?events[producer]:-1)<<'\n';
          }
        }
      }
    }
  }
  std::cout<<"RUNTIME_RELEASE checks="<<checks<<" mismatches="<<mismatches<<'\n';
  return mismatches?1:0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
