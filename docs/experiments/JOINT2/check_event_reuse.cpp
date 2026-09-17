// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/JointPlacement.h>
#include <iostream>
#include <stdexcept>
using namespace tilemega;
int main()try {
  int cells=0;
  for(int grid:{1,2,4})for(int width:{4,7}) {
    auto graph=codegen::MaterializeRuntimeTaskGraph({width,width,width},{},grid);
    for(int n=0;n<2*width;++n)graph.successors[n].push_back(n+width);
    solver::SimulatorInput input;input.graph=&graph;
    for(int n=0;n<3*width;++n)input.task_ns.push_back(1+(n*7)%13);
    solver::PlanRequest request;request.grid=grid;request.graph=&graph;request.counts={width,width,width};request.stage_order={0,1,2};
    request.physical_worker.resize(grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
    solver::SimulatorOptions options;options.observed_task_times=true;options.flat_hop=true;options.publication_ns=2;options.consumer_wait_ns=3;
    solver::HopCurve hop;hop.c0=1;
    auto callback=[&](bool reference) {
      return [&,reference](solver::MaterializedPlan const& plan,codegen::RuntimeTaskGraph& changed,
          solver::SimulatorInput& priced,std::vector<int>& rows) {
        if(reference)changed=graph;
        priced.publication_required.assign(3*width,0);priced.consumer_wait_required.assign(3*width,0);
        for(int n=0;n<width;++n) {
          int consumer=width+(n+1)%width;
          if(plan.owner[0][n]==plan.owner[1][consumer-width])continue;
          changed.successors[n].push_back(consumer);rows.push_back(n);
          priced.publication_required[n]=1;priced.consumer_wait_required[consumer]=1;
        }
        if(reference) {rows.resize(3*width);std::iota(rows.begin(),rows.end(),0);}
        for(int n:rows) {
          auto& row=changed.successors[n];std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
        }
      };
    };
    auto a=solver::SolvePlacementCatalog(input,request,options,hop,callback(false));
    auto b=solver::SolvePlacementCatalog(input,request,options,hop,callback(true));
    if(a.size()!=6 || b.size()!=6)throw std::runtime_error("catalog incomplete");
    for(std::size_t i=0;i<a.size();++i) {
      if(a[i].name!=b[i].name || a[i].error!=b[i].error || a[i].predicted_ns!=b[i].predicted_ns ||
          a[i].bounds.lower_bound_ns!=b[i].bounds.lower_bound_ns || a[i].plan.owner!=b[i].plan.owner || a[i].plan.slot!=b[i].plan.slot)
        throw std::runtime_error("changed-row replay differs from independent full graph rebuild");
    }
    ++cells;
  }
  std::cout<<"EVENT_REUSE PASS cells="<<cells<<" candidates="<<6*cells<<" independent_full_rebuild=identical\n";
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
