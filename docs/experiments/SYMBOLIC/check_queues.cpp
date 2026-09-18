// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/ParametricPlacement.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
using namespace tilemega;
using namespace tilemega::solver;
static std::string serialize(MaterializedPlan const& plan) {
  std::ostringstream out;
  out << "workers=" << plan.queue.size() << "\nworker\tslot\tstage\tlogical_task\n";
  for (std::size_t worker=0;worker<plan.queue.size();++worker)
    for (std::size_t slot=0;slot<plan.queue[worker].size();++slot) {
      auto const& task=plan.queue[worker][slot];
      out << worker << '\t' << slot << '\t' << task.stage << '\t' << task.logical << '\n';
    }
  return out.str();
}
int main(int argc,char** argv) try {
  if(argc!=4)throw std::invalid_argument("check_queues CG MANIFEST OUT");
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  if(!module)throw std::invalid_argument("invalid CG");
  auto runtime=codegen::ReadRuntimePlan(*module);
  auto kappa=module->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_kappa");
  std::filesystem::path out=argv[3];std::filesystem::create_directories(out);
  std::ifstream input(argv[2]);std::string line;std::getline(input,line);
  std::ofstream records(out/"comparisons.tsv");
  records << "family\tgrid\tseq\tnodes\tidentical\ttemplate\tnative\n";
  while(std::getline(input,line)) {
    std::istringstream row(line);std::string family,map_file;int grid,seq;
    if(!(row>>family>>grid>>seq>>map_file))throw std::invalid_argument("invalid manifest row");
    // The native host binds theta before projection; the opposite side still
    // evaluates the independently proved symbolic pi/sigma map at that point.
    auto model=ModelDescription::FromCouplingGraph(*module,{seq,3,seq+3},"queue-roundtrip");
    RuntimeProjectionOptions options{grid,128,kappa?int(kappa.getInt()):1};options.count_wait_entries=false;
    auto projection=ProjectRuntimeQueues(model,runtime,options);
    analysis::ParamBinding theta;theta.Bind("S",seq).Bind("G",grid);
    std::vector<int> counts;
    for(auto const& stage:projection.stages)counts.push_back(stage.task_count.Eval(theta));
    auto graph=codegen::MaterializeRuntimeTaskGraph(counts,{},grid);
    analysis::VisitFiniteRelation(isl,projection.dependencies.BindParams(theta).ToString(),4,[&](long const* edge) {
      graph.successors.at(graph.stage_offsets.at(edge[2])+edge[3]).push_back(graph.stage_offsets.at(edge[0])+edge[1]);
    });
    for(auto& successors:graph.successors) {
      std::sort(successors.begin(),successors.end());
      successors.erase(std::unique(successors.begin(),successors.end()),successors.end());
    }
    auto edges=runtime.dependencies;
    std::stable_sort(edges.begin(),edges.end(),[](auto const& a,auto const& b){return a.consumer<b.consumer;});
    PlanRequest request;request.grid=grid;request.counts=counts;request.graph=&graph;
    for(auto const& entry:BuildVariantStageSchedule(edges,model.stages.size()).schedule)
      for(std::size_t s=0;s<projection.stages.size();++s)
        if(projection.stages[s].logical_stage==int(entry.stage))request.stage_order.push_back(s);
    request.physical_worker.resize(grid);
    std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
    if(family=="rotate")request.mode=dialect::PlacementMode::kRotate;
    else if(family=="band" || family=="wavefront") {
      request.mode=dialect::PlacementMode::kTemplate;request.params={family=="band"?0:1};
    } else if(family!="legacy_grid_stride")throw std::invalid_argument("unknown family");
    std::ifstream map_input(map_file);if(!map_input)throw std::invalid_argument("missing map");
    std::ostringstream text;text<<map_input.rdbuf();ParametricPlacement placement;
    placement.pi_sigma=analysis::CouplingRelation::FromIslText(text.str());
    auto symbolic=EvaluateParametricPlacement(placement,theta,counts,grid);
    MaterializedPlan native;std::string error;
    if(!MaterializePlanPlacement(request,&native,&error) || !CheckPlanLegality(graph,native,&error))
      throw std::invalid_argument(error);
    auto a=serialize(symbolic),b=serialize(native);
    auto prefix=family+"_g"+std::to_string(grid)+"_s"+std::to_string(seq);
    std::ofstream(out/(prefix+".template.tsv"))<<a;
    std::ofstream(out/(prefix+".native.tsv"))<<b;
    records<<family<<'\t'<<grid<<'\t'<<seq<<'\t'<<graph.successors.size()<<'\t'<<(a==b)
           <<'\t'<<prefix<<".template.tsv\t"<<prefix<<".native.tsv\n";records.flush();
    if(a!=b)throw std::runtime_error("queue mismatch "+prefix);
    std::cout<<"QUEUE_EQUAL "<<prefix<<" nodes="<<graph.successors.size()<<'\n'<<std::flush;
  }
} catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
