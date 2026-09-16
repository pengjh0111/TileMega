// SPDX-License-Identifier: BSD-3-Clause
#include "../SIMULATOR/cell_inputs.h"
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
using namespace tilemega;
using namespace tilemega::solver;
using namespace tilemega::experiments;
using Clock=std::chrono::steady_clock;
static double micros(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
int main(int argc,char** argv) try {
  if (argc!=4) throw std::runtime_error("usage: evaluate REPO OUT TARGET");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto target=TargetSpec::FromJson(argv[3]);
  std::string repo=argv[1],out=argv[2],error;
  std::ifstream manifest(repo+"/docs/experiments/SIMULATOR/raw/manifest.tsv");
  std::string line;
  std::ofstream result(out+"/evaluations.tsv"),ranks(out+"/ranks.tsv");
  result<<std::setprecision(12)<<"model\tseq\tcandidate\tprepare_us\tcoarse_us\tfull_us\tcoarse_ns\tfull_ns\tbinding_cp_ns\tsemantic_cp_ns\tqueue_lb_ns\n";
  ranks<<std::setprecision(12)<<"model\tseq\tk\tindex\tcandidate\tsimulated\tpredicted_ns\tbatch_us\n";
  HopCurve hop;if (!HopCurve::FromTsv(repo+"/docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&error)) throw std::runtime_error(error);
  std::getline(manifest,line);
  while (std::getline(manifest,line)) {
    auto f=Split(line,'\t');int seq=std::stoi(f[1]);auto cell=ReadCellDump(f[4]+"_p5_s"+f[1]+".out");
    auto graph=codegen::MaterializeRuntimeTaskGraph(cell.counts,ReadDependencies(f[3],0),cell.grid);
    SimulatorInput input;input.graph=&graph;input.task_ns.resize(graph.stage_offsets.back());
    auto path=f[0]=="real" ? f.at(6) : repo+"/docs/experiments/SEQSCAN/raw/export/"+f[0]+".json";
    auto module=frontend::TorchExportImporter{}.Import(path,context);
    auto model=ModelDescription::FromCouplingGraph(*module,{seq,3,seq+3},"calibration-replay");
    std::vector<GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
    auto semantic=InstantiateModelTasks(model,configs);CostModel cost(target,model.dtype);
    if (cell.counts.size()!=model.stages.size()) throw std::runtime_error("historical stage shape changed");
    for (int s=0;s<int(cell.counts.size());++s) {
      auto const& stage=model.stages[s];auto const& g=configs[stage.IsCollective()?stage.gemm:0];
      auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x){return x.stage==s;});
      if (found==model.task_semantics.end()) throw std::runtime_error("missing semantic stage");
      auto task=DeriveModelTaskInput(model,*found,semantic,stage.IsCollective()? &g:nullptr);
      auto traits=ModelTaskTraits(model,s,g);analysis::ParamBinding point;
      for (auto const& coordinate:task.cost_coordinates) point.Bind(coordinate,0);
      if(task.scalar_access) point.Bind("q",0);
      double ns=cost.TaskInstanceNs(task,traits,{cell.ctas_per_sm},model,1,point,1.0);
      for (int n=graph.stage_offsets[s];n<graph.stage_offsets[s+1];++n) input.task_ns[n]=ns;
    }
    if (f[5]!="-") input.worker_sm=ReadWorkerSm(f[5]+"/slots.tsv",cell.grid);
    auto start=Clock::now();PreparedPlanBounds prepared;
    if (!PreparePlanBounds(input,&prepared,&error)) throw std::runtime_error(error);
    double preparation=micros(start);
    input.prepared_graph=&prepared.graph;
    SimulatorOptions options;options.sms=cell.num_sms;options.ctas_per_sm=cell.ctas_per_sm;options.proportional_sharing=true;
    options.observed_task_times=true;options.flat_hop=true;
    auto const& rates=target.EventCalibrationFor("bf16");
    options.publication_ns=rates.task_publication.ns.value_or(0);
    options.consumer_wait_ns=rates.task_wait.ns.value_or(0);
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
      result<<f[0]<<'\t'<<seq<<'\t'<<names[i]<<'\t'<<preparation<<'\t'<<coarse<<'\t'<<full_us<<'\t'<<b.lower_bound_ns<<'\t'<<full.makespan_ns<<'\t'<<b.binding_path_ns<<'\t'<<b.critical_path_ns<<'\t'<<b.queue_lb_ns<<'\n';
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
