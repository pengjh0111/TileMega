// SPDX-License-Identifier: BSD-3-Clause
#include "../cell_inputs.h"
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
using namespace tilemega;
using namespace tilemega::solver;
using namespace tilemega::experiments;
using Clock=std::chrono::steady_clock;
static double micros(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
int main(int argc,char** argv) try {
  if (argc!=3) throw std::runtime_error("usage: evaluate REPO OUT");
  std::string repo=argv[1],out=argv[2],error;
  std::ifstream manifest(repo+"/docs/experiments/SIMULATOR/raw/manifest.tsv");
  std::ifstream price(repo+"/docs/experiments/SIMULATOR/raw/stage_price.tsv");
  std::map<std::pair<std::string,int>,std::map<int,double>> weights;
  std::string line;std::getline(price,line);
  while (std::getline(price,line)) {
    auto f=Split(line,'\t');weights[{f[0],std::stoi(f[1])}][std::stoi(f[2])]=std::stod(f[6]);
  }
  std::ofstream result(out+"/evaluations.tsv"),ranks(out+"/ranks.tsv");
  result<<std::setprecision(12)<<"model\tseq\tcandidate\tprepare_us\tcoarse_us\tfull_us\tcoarse_ns\tfull_ns\n";
  ranks<<std::setprecision(12)<<"model\tseq\tk\tindex\tcandidate\tsimulated\tpredicted_ns\tbatch_us\n";
  HopCurve hop;if (!HopCurve::FromTsv(repo+"/docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&error)) throw std::runtime_error(error);
  std::getline(manifest,line);
  while (std::getline(manifest,line)) {
    auto f=Split(line,'\t');int seq=std::stoi(f[1]);auto cell=ReadCellDump(f[4]+"_p5_s"+f[1]+".out");
    auto graph=codegen::MaterializeRuntimeTaskGraph(cell.counts,ReadDependencies(f[3],0),cell.grid);
    SimulatorInput input;input.graph=&graph;input.task_ns.resize(graph.stage_offsets.back());
    for (int s=0;s<int(cell.counts.size());++s)
      for (int n=graph.stage_offsets[s];n<graph.stage_offsets[s+1];++n) input.task_ns[n]=weights.at({f[0],seq}).at(s);
    if (f[5]!="-") input.worker_sm=ReadWorkerSm(f[5]+"/slots.tsv",cell.grid);
    auto start=Clock::now();PreparedPlanBounds prepared;
    if (!PreparePlanBounds(input,&prepared,&error)) throw std::runtime_error(error);
    double preparation=micros(start);
    SimulatorOptions options;options.sms=cell.num_sms;options.ctas_per_sm=cell.ctas_per_sm;options.proportional_sharing=true;
    PlanRequest request;request.grid=cell.grid;request.counts=cell.counts;request.stage_order=cell.stage_order;
    request.physical_worker.resize(cell.grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);request.graph=&graph;
    std::vector<MaterializedPlan> plans;std::vector<std::string> names;
    for (auto const& c:Candidates()) {
      request.mode=c.mode;request.params=c.params;MaterializedPlan plan;
      if (!MaterializePlanPlacement(request,&plan,&error)) continue;
      if (!CheckPlanLegality(graph,plan,&error)) throw std::runtime_error(error);
      names.push_back(c.name);plans.push_back(std::move(plan));
    }
    std::vector<MaterializedPlan const*> pointers;
    for (std::size_t i=0;i<plans.size();++i) {
      pointers.push_back(&plans[i]);PlanBounds b;start=Clock::now();
      if (!EvaluatePlanBounds(prepared,plans[i],&b,&error)) throw std::runtime_error(error);
      double coarse=micros(start);SimulatorResult full;start=Clock::now();
      if (!SimulateExecution(input,plans[i],options,hop,&full,&error)) throw std::runtime_error(error);
      double full_us=micros(start);
      result<<f[0]<<'\t'<<seq<<'\t'<<names[i]<<'\t'<<preparation<<'\t'<<coarse<<'\t'<<full_us<<'\t'<<b.lower_bound_ns<<'\t'<<full.makespan_ns<<'\n';
    }
    for (std::size_t k=1;k<=plans.size();++k) {
      std::vector<RankedPlan> ranked;start=Clock::now();
      if (!RankPlans(input,prepared,pointers,options,hop,k,&ranked,&error)) throw std::runtime_error(error);
      double batch=micros(start);
      for (std::size_t i=0;i<ranked.size();++i) {
        auto const& r=ranked[i];ranks<<f[0]<<'\t'<<seq<<'\t'<<k<<'\t'<<i<<'\t'<<names[r.index]<<'\t'<<r.simulated<<'\t'<<r.makespan_ns<<'\t'<<batch<<'\n';
      }
    }
    result.flush();ranks.flush();std::cout<<"BOUNDS "<<f[0]<<" seq="<<seq<<" plans="<<plans.size()<<std::endl;
  }
} catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
