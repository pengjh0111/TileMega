// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <mlir/Parser/Parser.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <tilemega/Codegen/tasks/TaskResources.h>
using namespace tilemega;
namespace {
// Diagnostic only: sum semantic task traffic, not unique device-DRAM bytes.
// Boundary pieces are exact price classes; coordinate-varying spaces retain
// one piece per coordinate. No task placement or executor state is changed.
void AuditTraffic(solver::SymbolicProblem const& problem,solver::PreparedFlow const& flow,
    analysis::DramFloor const& floor,std::ostream& out) {
  auto model=problem.model;model.metric_bindings.values.erase("Tm");model.metric_bindings.values.erase("Tn");
  auto theta=model.MetricBindings();auto graph=solver::InstantiateModelTasks(model,problem.geometry);
  out<<std::setprecision(17)<<"space\ttasks\tnominal_read_bytes\tnominal_write_bytes\tphysical_read_bytes\tphysical_write_bytes\tno_producer_read_bytes\tproduced_read_bytes\texternal_write_bytes\n";
  for(std::size_t s=0;s<problem.counts.size();++s) {
    auto const& projected=problem.projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto semantic=*std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& sem){return sem.stage==projected.logical_stage && (!stage.IsCollective() || sem.op.kind==analysis::OperatorKind::kMatmul);});
    auto g=stage.IsCollective()?problem.geometry.at(stage.gemm):solver::GemmConfig{};
    solver::DerivedTaskInput input;
    if(projected.combine) {
      input=solver::DeriveCombineTaskInput(model,projected.logical_stage,g,graph,problem.threads,problem.runtime.ownership_flags & codegen::kCombinerTileOwnership,true);
      semantic.op.name=input.task.name;semantic.op.kind=analysis::OperatorKind::kReduction;semantic.op.element_reads.clear();
    } else input=solver::DeriveModelTaskInput(model,semantic,graph,stage.IsCollective()?&g:nullptr);
    solver::BindTaskDramProvenance(input,semantic,floor,theta);
    bool partial=!projected.combine && graph.Find(semantic.op.reduction.combiner);
    double sums[7]={};long count=0;
    std::vector<analysis::ParamBinding> points;
    for(auto const& piece:flow.prices[s].pieces)points.push_back(piece.representative);
    // Scalar task traffic already used the physical q-domain before R9b.
    auto prior_domain=stage.IsCollective() && !projected.combine ? analysis::AccessDomain::kNominalTile : analysis::AccessDomain::kPhysicalTensor;
    auto nominal_values=solver::DeriveTaskMemoryTrafficBatch(input,theta,points,2,partial?4:2,prior_domain);
    auto physical_values=solver::DeriveTaskMemoryTrafficBatch(input,theta,points,2,partial?4:2,analysis::AccessDomain::kPhysicalTensor);
    for(std::size_t i=0;i<points.size();++i) {
      auto n=flow.prices[s].pieces[i].count.Eval(theta);count+=n;
      auto const& nominal=nominal_values[i];auto const& physical=physical_values[i];
      double values[]={nominal.global_read_bytes,nominal.global_write_bytes,physical.global_read_bytes,physical.global_write_bytes,physical.no_producer_read_bytes,physical.produced_read_bytes,physical.external_write_bytes};
      for(int j=0;j<7;++j)sums[j]+=n*values[j];
    }
    if(count!=problem.counts[s] || sums[4]+sums[5]!=sums[2])throw std::runtime_error("traffic piece/provenance mismatch");
    out<<input.task.name<<'\t'<<count;for(double value:sums)out<<'\t'<<value;out<<'\n';out.flush();
  }
}
}
int main(int argc,char** argv) try {
  if(argc!=7 && argc!=8 && !(argc==9 && std::string(argv[7])=="--traffic"))throw std::invalid_argument("usage: tilemega-flow-audit CG target seq fixture residency kappa [materialization_prefix | --traffic output.tsv]");
  analysis::IslContext isl;analysis::ScopedExactAnalysisMemo memo;mlir::MLIRContext ctx;ctx.getOrLoadDialect<dialect::CGDialect>();ctx.getOrLoadDialect<dialect::ExecDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&ctx);if(!module)throw std::runtime_error("cannot parse CG");
  auto target=TargetSpec::FromJson(argv[2]);int seq=std::stoi(argv[3]),residency=std::stoi(argv[5]),kappa=std::stoi(argv[6]);
  auto begin=std::chrono::steady_clock::now();
  auto problem=solver::PrepareSymbolicProblem(*module,target,{seq,3,seq+3},target.res.num_sms*residency,residency,kappa,nullptr,false);
  std::cout<<"RELATIONS ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()<<std::endl;
  auto floor=solver::DeriveModelDramFloor(*module,problem.model,target,argv[4]);
  std::cout<<"FLOOR ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()<<std::endl;
  analysis::CouplingCache coupling;solver::FlowPreparationCache cache;solver::HopCurve hop;std::string error;if(!solver::HopCurve::FromTsv(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&error))throw std::runtime_error(error);
  auto prepared=solver::PrepareFlow(problem,floor,target,residency,hop,coupling,cache,false);
  std::cout<<std::setprecision(17)<<"PREPARE ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()<<" spaces="<<prepared.flow.spaces.size()<<" edges="<<prepared.flow.edges.size()<<" price_hits="<<cache.prices.hits<<" price_misses="<<cache.prices.misses<<" varying="<<prepared.varying_spaces.size()<<std::endl;
  if(argc==9){std::ofstream traffic(argv[8]);if(!traffic)throw std::runtime_error("cannot open traffic output");AuditTraffic(problem,prepared,floor,traffic);return 0;}
  auto structure_start=std::chrono::steady_clock::now();
  auto quick=solver::PrepareFlowStructure(problem,problem.geometry,target.res.num_sms*residency,kappa,coupling,&cache);
  if(quick.counts!=problem.counts)throw std::runtime_error("semantic stage counts differ from runtime projection");
  auto quick_flow=solver::PrepareFlow(quick,floor,target,residency,hop,coupling,cache,false);
  std::map<std::pair<int,int>,std::vector<std::pair<int,int>>> laws;
  for(auto const& e:prepared.flow.edges)laws[{e.producer,e.consumer}]=*e.sorted;
  for(auto const& e:quick_flow.flow.edges){auto it=laws.find({e.producer,e.consumer});if(it==laws.end() || it->second!=*e.sorted)throw std::runtime_error("semantic release law differs from runtime projection");laws.erase(it);}
  if(!laws.empty())throw std::runtime_error("semantic flow omitted a runtime edge");
  if(solver::EvaluateFlow(quick_flow.flow).makespan_ns!=solver::EvaluateFlow(prepared.flow).makespan_ns)throw std::runtime_error("semantic flow differs from projected flow");
  std::cout<<"STRUCTURE ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-structure_start).count()<<" counts_exact=1 releases_exact=1 makespan_exact=1"<<std::endl;
  auto warm_begin=std::chrono::steady_clock::now();
  quick=solver::PrepareFlowStructure(problem,problem.geometry,target.res.num_sms*residency,kappa,coupling,&cache);
  auto warm=solver::PrepareFlow(quick,floor,target,residency,hop,coupling,cache,false);
  std::cout<<"PREPARE_WARM ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-warm_begin).count()<<" space_hits="<<cache.space_hits<<" space_misses="<<cache.space_misses<<std::endl;
  auto first=solver::EvaluateFlow(prepared.flow);double sum=0,max=0;
  for(int i=0;i<10;++i){auto start=std::chrono::steady_clock::now();auto value=solver::EvaluateFlow(prepared.flow);double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();sum+=elapsed;max=std::max(max,elapsed);if(value.makespan_ns!=first.makespan_ns)throw std::runtime_error("nondeterministic flow");}
  std::cout<<"FLOW ns="<<first.makespan_ns<<" floor_ns="<<prepared.flow.floor_ns<<" delivered="<<first.delivered_bytes<<" mean_ms="<<sum/10<<" max_ms="<<max<<std::endl;
  auto parts=solver::DecomposeFlow(prepared.flow);
  std::cout<<"DECOMPOSE sync="<<parts.synchronization<<" fixed="<<parts.fixed<<" contention="<<parts.contention<<" chain="<<parts.chain<<" pg="<<parts.pg_upper_bound<<std::endl;
  if(argc==8) {
    solver::ApplyFlowPrices(problem,prepared,target,residency);
    auto skeleton=solver::BuildPlanSkeleton(problem,target.res.num_sms*residency,residency,8,false,coupling);
    for(auto& s:skeleton.spaces){s.colocation.reset();s.colocated_producer=-1;}
    solver::SkeletonRequest request;request.skeleton=&skeleton;request.pure_template=true;request.hop=hop;request.sms=skeleton.grid;
    solver::EftSchedule schedule;solver::SkeletonPlacementStats stats;
    if(!solver::ScheduleBySkeleton(request,&schedule,&stats,&error))throw std::runtime_error(error);
    solver::SkeletonSolvedPoint point;point.module=std::move(module);point.problem=std::move(problem);point.skeleton=std::move(skeleton);point.schedule=std::move(schedule);point.flow=prepared;
    point.candidate.config=point.problem.geometry;point.candidate.key="flow-audit";point.candidate.residency=residency;point.candidate.actual_limit=residency;point.candidate.placement=stats;
    solver::SkeletonSearchOptions opts;opts.kappa=kappa;opts.common.placement.target=target;opts.common.placement.hop=hop;opts.common.placement.dims={seq,3,seq+3};
    auto start=std::chrono::steady_clock::now();auto entry=solver::FinalizeSkeletonPoint(std::move(point),opts,argv[7]);
    std::cout<<"FLUID ns="<<entry.evaluation.makespan_ns<<" total_prepare_and_sim_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<std::endl;
  }
  for(auto const& s:prepared.varying_spaces)std::cout<<"VARYING "<<s<<'\n';
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
