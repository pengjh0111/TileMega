// SPDX-License-Identifier: BSD-3-Clause
// Fixed R9b points for placement and pipeline-stage ablations; no outer search.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <iostream>
#include <regex>
using namespace tilemega;
int main(int argc,char** argv) try {
  if(argc!=14)throw std::invalid_argument("tilemega-flow-point export target seq past kappa residency kbase|W geometries prefix hop fixture pure actual_limit");
  analysis::IslContext isl;analysis::ScopedExactAnalysisMemo memo;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto bridge=frontend::ReadExportBridge(argv[1]);auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  frontend::TorchExportImporter importer;auto imported=importer.ImportSemantics(argv[1],plan,context);
  auto classes=solver::BuildOperatorClasses(imported);std::vector<solver::GemmConfig> configs;
  std::string key=argv[8];std::regex geometry("([0-9]+)x([0-9]+)x([0-9]+)s([0-9]+)k([0-9]+)");
  for(std::sregex_iterator i(key.begin(),key.end(),geometry),end;i!=end;++i)configs.push_back({std::stoi((*i)[1]),std::stoi((*i)[2]),std::stoi((*i)[3]),std::stoi((*i)[4]),std::stoi((*i)[5])});
  solver::SkeletonSearchOptions opts;opts.common.placement.target=TargetSpec::FromJson(argv[2]);
  int seq=std::stoi(argv[3]),past=std::stoi(argv[4]),residency=std::stoi(argv[6]),grid=opts.common.placement.target.res.num_sms*residency;
  opts.common.placement.dims={seq,past,seq+past};opts.kappa=std::stoi(argv[5]);opts.all_workers=std::string(argv[7])=="W";opts.k_base=opts.all_workers?grid:std::stoi(argv[7]);
  std::string error;if(!solver::HopCurve::FromTsv(argv[10],&opts.common.placement.hop,&error))throw std::runtime_error(error);
  analysis::CouplingCache cache;solver::SkeletonSolvedPoint point;
  point.module=importer.InstantiateForGranularity(imported,context,solver::ClassGranularity(imported,classes,configs),&cache);
  point.problem=solver::PrepareSymbolicProblem(*point.module,opts.common.placement.target,opts.common.placement.dims,grid,residency,opts.kappa,nullptr,false);
  auto floor=solver::DeriveModelDramFloor(*point.module,point.problem.model,opts.common.placement.target,argv[11]);
  solver::FlowPreparationCache prices;
  point.flow=solver::PrepareFlow(point.problem,floor,opts.common.placement.target,residency,opts.common.placement.hop,cache,prices);
  point.candidate.score=solver::EvaluateFlow(point.flow->flow).makespan_ns;
  solver::ApplyFlowPrices(point.problem,*point.flow,opts.common.placement.target,residency);
  point.skeleton=solver::BuildPlanSkeleton(point.problem,grid,residency,opts.k_base,opts.all_workers,cache);
  solver::SkeletonRequest request;request.skeleton=&point.skeleton;request.hop=opts.common.placement.hop;request.sms=grid;request.pure_template=std::stoi(argv[12])!=0;
  if(!solver::ScheduleBySkeleton(request,&point.schedule,&point.candidate.placement,&error))throw std::runtime_error(error);
  point.candidate.config=configs;point.candidate.key=key;point.candidate.residency=residency;point.candidate.actual_limit=std::stoi(argv[13]);point.candidate.estimated_limit=residency;point.candidate.kappa=opts.kappa;
  std::cout<<"FLOW_POINT flow_ns="<<point.candidate.score<<" grid="<<grid<<'\n';
  solver::SolverTiming timing;opts.common.timing=&timing;
  auto result=solver::FinalizeSkeletonPoint(std::move(point),opts,argv[9]);std::string prefix=argv[9];
  std::ofstream source(prefix+".cu");source<<codegen::CouplingGraphToCUDA{}.LowerVariants({{*result.module,1u,static_cast<unsigned>(seq)}});
  std::error_code ec;llvm::raw_fd_ostream cg(prefix+".mlir",ec);result.module->print(cg);
  std::ofstream times(prefix+".timing.tsv");timing.Write(times,"flow-point",argv[1],seq);
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
