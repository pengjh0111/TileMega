// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Builders.h>
#include <isl/map.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace tilemega::solver {
SymbolicProblem PrepareSymbolicProblem(mlir::ModuleOp module,TargetSpec const& target,
    ModelDims dims,int grid,int residency,int kappa) {
  if(grid<=0 || residency<=0 || dims.seq<=0 || kappa<=0)throw std::invalid_argument("invalid symbolic problem dimensions");
  auto runtime=codegen::ReadRuntimePlan(module);
  auto model=ModelDescription::FromCouplingGraph(module,dims,"skeleton");
  std::vector<GemmConfig> geometry;
  for(auto const& g:runtime.gemms)geometry.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  int threads=0;
  for(std::size_t s=0;s<model.stages.size();++s) {
    auto const& stage=model.stages[s];auto const& g=geometry.at(stage.IsCollective()?stage.gemm:0);
    threads=std::max(threads,ModelTaskTraits(model,int(s),g).threads);
  }
  RuntimeProjectionOptions po{grid,threads,kappa};po.count_wait_entries=false;
  auto projection=ProjectRuntimeQueues(model,runtime,po);
  std::vector<int> counts,offsets{0};
  for(auto const& stage:projection.stages){counts.push_back(int(stage.task_count.Eval({})));offsets.push_back(offsets.back()+counts.back());}
  std::optional<analysis::OperatorGraph> semantic_graph;
  semantic_graph=InstantiateModelTasks(model,geometry);
  struct Prices {std::vector<double> task_ns,prefetch_ns;} input;input.task_ns.resize(offsets.back());
  input.prefetch_ns.resize(offsets.back());
  CostModel cost(target,model.dtype);
  auto theta=model.MetricBindings();
  for (std::size_t s=0;s<counts.size();++s) {
    auto const& projected=projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto const& g=geometry.at(stage.IsCollective() ? stage.gemm : 0);
    if (projected.combine) {
      auto task=DeriveCombineTaskInput(model,projected.logical_stage,g,*semantic_graph,threads,
          runtime.ownership_flags & codegen::kCombinerTileOwnership,cost.options().fp32_partials);
      auto resources=codegen::ReadSimtTaskResources(codegen::TaskKind::kGemmCombine,threads);
      BackendTraits traits;traits.threads=resources.threads;traits.smem_bytes=resources.shared_bytes;traits.shape_legal=true;
      std::vector<analysis::ParamBinding> coordinates(counts[s]);
      for (int t=0;t<counts[s];++t) coordinates[t].Bind("q",t);
      if(task.work.task_count.Eval(theta)!=counts[s])
        throw std::invalid_argument("combine access ownership disagrees with projected count");
      std::vector<double> prefetch;PrefetchPricing pricing{0,&prefetch};
      auto prices=PriceTaskInstances(cost,task,traits,{residency},model,1,coordinates,residency,&pricing);
      std::copy(prices.begin(),prices.end(),input.task_ns.begin()+offsets[s]);
      std::copy(prefetch.begin(),prefetch.end(),input.prefetch_ns.begin()+offsets[s]);
      continue;
    }
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x) {
      return x.stage==projected.logical_stage && (!stage.IsCollective() || x.op.kind==analysis::OperatorKind::kMatmul);
    });
    if (found==model.task_semantics.end()) throw std::invalid_argument("stage lacks derived semantic task costs");
    auto task=DeriveModelTaskInput(model,*found,*semantic_graph,stage.IsCollective() ? &g : nullptr);
    auto traits=ModelTaskTraits(model,projected.logical_stage,g);
    int chunks=stage.IsCollective() ? cost.Chunks(model.gemms.at(stage.gemm),g) : 1;
    std::vector<analysis::ParamBinding> coordinates(counts[s]);
    if (task.scalar_access) {
      for (int t=0;t<counts[s];++t) coordinates[t].Bind("q",t);
    } else {
      auto ownership=ProjectTaskOwnership(*found,task.task,stage,threads).BindParams(theta);
      auto names=ownership.RangeDimNames();
      for (auto const& [physical,logical]:ownership.Points()) {
        if (physical.size()!=1 || physical[0]<0 || physical[0]>=counts[s])
          throw std::invalid_argument("task cost ownership disagrees with projected count");
        for (std::size_t i=0;i<names.size();++i) coordinates[physical[0]].Bind(names[i],logical[i]);
      }
    }
    std::vector<double> prefetch;PrefetchPricing pricing{0,&prefetch};
    auto prices=PriceTaskInstances(cost,task,traits,{residency},model,chunks,coordinates,residency,&pricing);
    std::copy(prices.begin(),prices.end(),input.task_ns.begin()+offsets[s]);
    std::copy(prefetch.begin(),prefetch.end(),input.prefetch_ns.begin()+offsets[s]);
  }
  return {std::move(runtime),std::move(model),std::move(geometry),std::move(projection),
    std::move(counts),std::move(offsets),std::move(input.task_ns),std::move(input.prefetch_ns),threads};
}
std::vector<int> PlanSkeleton::Spread(int stage,int tile) const {
  auto const& space=spaces.at(stage);int home=(space.base+tile)%grid;
  std::vector<int> result;result.reserve(space.width);
  for(int i=0;i<space.width;++i)result.push_back((home+i*(grid/space.width))%grid);
  return result;
}
PlanSkeleton BuildPlanSkeleton(SymbolicProblem const& problem,int grid,int residency,
    int k_base,bool all_workers,analysis::CouplingCache& cache,SolverTiming* timing) {
  SolverPhase phase(timing,"skeleton");
  if(grid<=0 || residency<=0 || k_base<=0)throw std::invalid_argument("Skeleton needs known resident worker count");
  PlanSkeleton result;result.grid=grid;result.residency=residency;result.task_ns=problem.task_ns;
  result.theta=problem.model.MetricBindings();
  std::vector<double> loads;
  for(std::size_t s=0;s<problem.counts.size();++s) {
    int n=problem.counts[s],offset=problem.offsets[s];
    double load=std::accumulate(problem.task_ns.begin()+offset,problem.task_ns.begin()+offset+n,0.0);
    result.spaces.push_back({n,offset,0,0,0,n ? load/n:0,load});loads.push_back(load);
  }
  auto sorted=loads;std::sort(sorted.begin(),sorted.end());
  double median=sorted.empty()?1:sorted[sorted.size()/2];
  if(!sorted.empty() && sorted.size()%2==0)median=(median+sorted[sorted.size()/2-1])/2;
  if(median<=0)median=1;
  auto deps=problem.runtime.dependencies;
  std::stable_sort(deps.begin(),deps.end(),[](auto const& a,auto const& b){return a.consumer<b.consumer;});
  int prefix=0;
  for(auto const& logical:BuildVariantStageSchedule(deps,problem.model.stages.size()).schedule)
    for(std::size_t s=0;s<problem.counts.size();++s)if(problem.projection.stages[s].logical_stage==int(logical.stage)) {
      auto& space=result.spaces[s];space.base=prefix%grid;prefix+=space.count;
      space.width=all_workers ? grid:std::min(grid,std::max(2,int(std::ceil(k_base*loads[s]/median))));
      space.order=result.stage_order.size();result.stage_order.push_back(int(s));
    }
  result.incoming.resize(result.spaces.size());result.outgoing.resize(result.spaces.size());
  auto const& relation=problem.projection.dependencies;
  auto* raw=isl_map_read_from_str(analysis::SharedIslContext().raw(),relation.ToString().c_str());
  auto* stage_map=isl_map_project_out(isl_map_copy(raw),isl_dim_in,1,1);
  stage_map=isl_map_project_out(stage_map,isl_dim_out,1,1);
  char* text=isl_map_to_str(stage_map);if(!text){isl_map_free(raw);isl_map_free(stage_map);throw std::runtime_error("stage graph projection failed");}
  auto stage_relation=analysis::CouplingRelation::FromIslText(text);free(text);isl_map_free(stage_map);
  // Enumerates only task-space pairs, never tile dependencies.
  for(auto const& [consumer,producer]:stage_relation.BindParams(result.theta).Points()) {
    int c=int(consumer.at(0)),p=int(producer.at(0));
    auto* edge=isl_map_fix_si(isl_map_copy(raw),isl_dim_in,0,c);
    edge=isl_map_fix_si(edge,isl_dim_out,0,p);
    edge=isl_map_project_out(edge,isl_dim_in,0,1);edge=isl_map_project_out(edge,isl_dim_out,0,1);
    edge=isl_map_reverse(edge);text=isl_map_to_str(edge);
    if(!text){isl_map_free(edge);isl_map_free(raw);throw std::runtime_error("stage-pair Oracle failed");}
    auto oracle=cache.OracleFor(text);free(text);isl_map_free(edge);
    bool all=oracle->reverse.IsAllBox({{0,result.spaces[p].count-1}}) &&
             oracle->forward.IsAllBox({{0,result.spaces[c].count-1}});
    int index=result.edges.size();result.edges.push_back({p,c,std::move(oracle),all});
    result.incoming[c].push_back(index);result.outgoing[p].push_back(index);
  }
  isl_map_free(raw);return result;
}
void WritePlanSkeleton(mlir::ModuleOp module,PlanSkeleton const& skeleton) {
  mlir::OpBuilder b(module.getContext());
  for(auto op:llvm::make_early_inc_range(module.getOps<dialect::SkeletonOp>()))op.erase();
  b.setInsertionPointToEnd(module.getBody());
  llvm::SmallVector<mlir::Attribute> spaces,edges;
  for(std::size_t s=0;s<skeleton.spaces.size();++s){auto const& t=skeleton.spaces[s];
    spaces.push_back(b.getDictionaryAttr({b.getNamedAttr("stage",b.getI64IntegerAttr(s)),
      b.getNamedAttr("count",b.getI64IntegerAttr(t.count)),b.getNamedAttr("base",b.getI64IntegerAttr(t.base)),
      b.getNamedAttr("width",b.getI64IntegerAttr(t.width)),b.getNamedAttr("load_ns",b.getF64FloatAttr(t.load_ns))}));}
  for(auto const& e:skeleton.edges)edges.push_back(b.getDictionaryAttr({
    b.getNamedAttr("producer",b.getI64IntegerAttr(e.producer)),b.getNamedAttr("consumer",b.getI64IntegerAttr(e.consumer)),
    b.getNamedAttr("structure",b.getStringAttr(analysis::ToString(e.oracle->structure))),
    b.getNamedAttr("predecessor",b.getStringAttr(analysis::ToString(e.oracle->reverse.kind()))),
    b.getNamedAttr("successor",b.getStringAttr(analysis::ToString(e.oracle->forward.kind()))),
    b.getNamedAttr("all_producer",b.getBoolAttr(e.all_producer))}));
  mlir::OperationState state(b.getUnknownLoc(),"tmexec.skeleton");
  state.addAttribute("workers",b.getI64IntegerAttr(skeleton.grid));state.addAttribute("residency",b.getI64IntegerAttr(skeleton.residency));
  state.addAttribute("spaces",b.getArrayAttr(spaces));state.addAttribute("edges",b.getArrayAttr(edges));
  state.addAttribute("affinity_limit",b.getI64IntegerAttr(2));b.create(state);
}
}
