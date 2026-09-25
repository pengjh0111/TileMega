// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/SkeletonPlacement.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  analysis::IslContext isl;mlir::MLIRContext context;
  std::string root=TILEMEGA_SOURCE_DIR,path=root+"/docs/experiments/E2E_GEN/raw/export_bridge.json";
  auto bridge=frontend::ReadExportBridge(path);auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  for(auto const& buffer:plan.buffers)if(buffer.name.find("inv_freq")!=std::string::npos && buffer.constant!=64)
    throw std::runtime_error("F32 frequency storage was counted as BF16 words");
  frontend::TorchExportImporter importer;auto imported=importer.ImportSemantics(path,plan,context);
  frontend::ImportOptions options;options.gemms.resize(plan.gemms.size(),{32,16,32,2,1});
  options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
  analysis::CouplingCache cache;auto module=importer.InstantiateForGranularity(imported,context,options,&cache);
  auto target=TargetSpec::FromJson(root+"/docs/experiments/COSTMODEL/event_fit/target.json");
  auto problem=solver::PrepareSymbolicProblem(*module,target,{4,3,7},8,1,1);
  solver::SymbolicPriceCache prices;
  auto cached=solver::PrepareSymbolicProblem(*module,target,{4,3,7},8,1,1,&prices);
  auto warm=solver::PrepareSymbolicProblem(*module,target,{4,3,7},8,1,1,&prices);
  if(cached.task_ns!=problem.task_ns || warm.task_ns!=problem.task_ns || warm.prefetch_ns!=problem.prefetch_ns)
    throw std::runtime_error("semantic price cache changed prices");
  auto uncached_r2=solver::PrepareSymbolicProblem(*module,target,{4,3,7},16,2,1);
  auto cached_r2=solver::PrepareSymbolicProblem(*module,target,{4,3,7},16,2,1,&prices);
  if(uncached_r2.task_ns!=cached_r2.task_ns)throw std::runtime_error("work cache changed residency pricing");
  auto same_relation=[](auto const& a,auto const& b){return a.IsSubset(b) && b.IsSubset(a);};
  if(uncached_r2.projection.dependencies.ToString()!=cached_r2.projection.dependencies.ToString() ||
      !same_relation(uncached_r2.projection.requested_events,cached_r2.projection.requested_events) ||
      !same_relation(uncached_r2.projection.waits,cached_r2.projection.waits) ||
      uncached_r2.execution_dependencies.ToString()!=cached_r2.execution_dependencies.ToString())
    throw std::runtime_error("prepared cache changed resident dependency or event relations");
  if(uncached_r2.projection.max_worker_task_refs.Eval(uncached_r2.model.MetricBindings())!=
      cached_r2.projection.max_worker_task_refs.Eval(cached_r2.model.MetricBindings()))
    throw std::runtime_error("regridding changed the concrete legacy-queue bound");
  auto uncached_s8=solver::PrepareSymbolicProblem(*module,target,{8,3,11},16,2,1);
  auto cached_s8=solver::PrepareSymbolicProblem(*module,target,{8,3,11},16,2,1,&prices);
  if(uncached_s8.task_ns!=cached_s8.task_ns || uncached_s8.counts!=cached_s8.counts)
    throw std::runtime_error("prepared cache reused the wrong theta");
  std::cout<<"PRICE_CACHE exact=1 entries="<<prices.prices.size()<<'\n';
  solver::RuntimeProjectionOptions po{8,problem.threads,1};po.count_wait_entries=false;
  auto windows=solver::ProjectRuntimeQueues(problem.model,problem.runtime,po);
  auto exact=problem.projection.dependencies.BindParams(problem.model.MetricBindings());
  if(!exact.IsSubset(windows.dependencies))throw std::runtime_error("exact CG dependencies escape runtime waits");
  auto extra=windows.dependencies.Subtract(exact);
  std::cout<<"DEPENDENCIES exact="<<exact.Points().size()<<" runtime_extra="<<extra.Points().size()<<'\n';
  for(int k:{1,2,4}) {
    auto execution_problem=solver::PrepareSymbolicProblem(*module,target,{8,3,11},8,1,k);
    auto theta=execution_problem.model.MetricBindings();
    using Point=std::pair<std::vector<long>,std::vector<long>>;
    std::set<Point> expected,actual;
    for(auto const& [consumer,event]:execution_problem.projection.requested_events.BindParams(theta).Points()) {
      int stage=event[1],kind=event[2],group=event[3];
      int begin=kind==0?0:group*k,end=kind==0?execution_problem.counts[stage]:std::min(execution_problem.counts[stage],begin+k);
      if(kind==2){begin=group;end=group+1;}
      for(int p=begin;p<end;++p)expected.insert({consumer,{stage,p}});
    }
    for(auto const& edge:execution_problem.execution_dependencies.BindParams(theta).Points())actual.insert(edge);
    if(actual!=expected)throw std::runtime_error("symbolic execution ordering differs from exact event expansion");
    auto executable=solver::BuildPlanSkeleton(execution_problem,8,1,4,false,cache);
    solver::SkeletonRequest request;request.skeleton=&executable;
    solver::EftSchedule schedule;std::string error;
    if(!solver::ScheduleBySkeleton(request,&schedule,nullptr,&error))throw std::runtime_error(error);
    for(auto const& [c,p]:expected) {
      int cn=execution_problem.offsets[c[0]]+c[1],pn=execution_problem.offsets[p[0]]+p[1];
      if(schedule.worker[cn]==schedule.worker[pn] && schedule.slot[pn]>=schedule.slot[cn])
        throw std::runtime_error("executor producer follows consumer on the same worker");
      if(schedule.end_ns[pn]>schedule.start_ns[cn])throw std::runtime_error("executor predecessor not ready");
    }
    std::cout<<"EXECUTION_ORDER kappa="<<k<<" pairs="<<actual.size()<<" exact=1 legal=1 PASS\n";
  }
  auto skeleton=solver::BuildPlanSkeleton(problem,8,1,4,false,cache);
  auto res2=solver::BuildPlanSkeleton(cached_r2,16,2,4,false,cache);
  auto reuse2=solver::BuildPlanSkeleton(cached_r2,16,2,4,false,cache,nullptr,&skeleton);
  solver::SkeletonRequest full_request;full_request.skeleton=&res2;
  solver::SkeletonRequest reuse_request;reuse_request.skeleton=&reuse2;
  solver::EftSchedule full_schedule,reuse_schedule;std::string schedule_error;
  if(!solver::ScheduleBySkeleton(full_request,&full_schedule,nullptr,&schedule_error) ||
      !solver::ScheduleBySkeleton(reuse_request,&reuse_schedule,nullptr,&schedule_error))throw std::runtime_error(schedule_error);
  if(full_schedule.worker!=reuse_schedule.worker || full_schedule.slot!=reuse_schedule.slot ||
      full_schedule.start_ns!=reuse_schedule.start_ns || full_schedule.end_ns!=reuse_schedule.end_ns)
    throw std::runtime_error("resident skeleton reuse changed the placement");
  std::cout<<"RESIDENT_REUSE grid=8->16 exact_schedule=1 PASS\n";
  int prefix=0;for(int s:skeleton.stage_order){auto const& space=skeleton.spaces[s];
    if(space.base!=prefix%8)throw std::runtime_error("incorrect continuous rotate base");prefix+=space.count;
    auto spread=skeleton.Spread(s,0);if(spread.front()!=skeleton.Home(s,0) || spread.size()!=std::size_t(space.width))throw std::runtime_error("spread definition mismatch");}
  int colocated=0;
  solver::SkeletonRequest pure;pure.skeleton=&skeleton;pure.pure_template=true;
  solver::EftSchedule template_schedule;
  if(!solver::ScheduleBySkeleton(pure,&template_schedule,nullptr,&schedule_error))throw std::runtime_error(schedule_error);
  for(std::size_t s=0;s<skeleton.spaces.size();++s) {
    auto const& space=skeleton.spaces[s];
    if(space.width!=4)throw std::runtime_error("candidate width is not constant");
    if(space.colocation){++colocated;if(space.colocation->reverse.UniqueMapText().empty())throw std::runtime_error("missing Unique map text");}
    for(int t=0;t<space.count;++t) {
      int home=skeleton.Home(s,t);
      if(template_schedule.worker[space.offset+t]!=home)throw std::runtime_error("pure template moved a task from home");
      if(space.colocation)space.colocation->reverse.Query({t},skeleton.theta).ForEach([&](auto const& p){
        if(home!=skeleton.Home(space.colocated_producer,p[0]))throw std::runtime_error("Unique colocation mismatch");});
    }
  }
  if(!colocated)throw std::runtime_error("fixture lacks a tested colocation edge");
  std::cout<<"TEMPLATE constant_width=4 colocated="<<colocated<<" pure_home=1 PASS\n";
  solver::WritePlanSkeleton(*module,skeleton);
  if(mlir::failed(mlir::verify(*module)))throw std::runtime_error("Skeleton IR invalid");
  if(argc>1){std::string text;llvm::raw_string_ostream out(text);module->print(out);std::ofstream(argv[1])<<text;}
  std::map<std::string,int> kinds,structures;for(auto const& e:skeleton.edges){++structures[analysis::ToString(e.oracle->structure)];++kinds[analysis::ToString(e.oracle->reverse.kind())];++kinds[analysis::ToString(e.oracle->forward.kind())];}
  for(auto [k,n]:structures)std::cout<<"STRUCTURE "<<k<<" edges="<<n<<'\n';
  for(auto [k,n]:kinds)std::cout<<"ORACLE_KIND "<<k<<" directions="<<n<<'\n';
  std::cout<<"SKELETON spaces="<<skeleton.spaces.size()<<" edges="<<skeleton.edges.size()<<" workers=8 PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
