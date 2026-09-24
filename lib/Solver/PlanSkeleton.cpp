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
#include <sstream>
#include <iomanip>

namespace tilemega::solver {
namespace {
analysis::CouplingRelation ExactRuntimeDependencies(ModelDescription const& model,
    analysis::OperatorGraph const& graph,RuntimeProjection const& projection,int threads) {
  struct Owner {int stage;analysis::CouplingRelation map;};
  std::map<std::string,Owner> owners;
  std::vector<int> entry(model.stages.size(),-1),done(entry);
  for(std::size_t s=0;s<projection.stages.size();++s){int logical=projection.stages[s].logical_stage;
    if(entry[logical]<0)entry[logical]=s;done[logical]=s;}
  for(auto const& node:graph.nodes) {
    auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& s){
      return s.op.name==node.name || s.op.reduction.combiner==node.name;});
    if(semantic==model.task_semantics.end())throw std::invalid_argument("CG task lacks physical ownership: "+node.name);
    bool combine=semantic->op.reduction.combiner==node.name;
    bool gemm=semantic->op.kind==analysis::OperatorKind::kMatmul && !combine;
    int logical=semantic->stage,physical=gemm?entry[logical]:done[logical];
    auto declaration=*semantic;
    if(combine)declaration.element_chunk=!model.combiner_tile_ownership;
    auto map=ProjectTaskOwnership(declaration,node,model.stages[logical],threads);
    owners.emplace(node.name,Owner{physical,std::move(map)});
  }
  analysis::CouplingRelation result=analysis::CouplingRelation::FromIslText("{ [cs,c] -> [ps,p] : false }");
  for(auto const& edge:model.coupling_metrics.edges) {
    auto const& p=owners.at(edge.producer_task);auto const& c=owners.at(edge.consumer_task);
    auto relation=c.map.ApplyRange(edge.relation).ApplyRange(p.map.Reverse());
    auto* map=isl_map_read_from_str(analysis::SharedIslContext().raw(),relation.ToString().c_str());
    map=isl_map_insert_dims(map,isl_dim_in,0,1);map=isl_map_fix_si(map,isl_dim_in,0,c.stage);
    map=isl_map_insert_dims(map,isl_dim_out,0,1);map=isl_map_fix_si(map,isl_dim_out,0,p.stage);
    char* raw=isl_map_to_str(map);isl_map_free(map);
    if(!raw)throw std::runtime_error("CG ownership composition failed");
    result=result.Union(analysis::CouplingRelation::FromIslText(raw));free(raw);
  }
  // A fused epilogue's semantic phases are one physical task, not a self wait.
  auto identity=analysis::CouplingRelation::FromIslText("{ [s,t] -> [s,t] }");
  return result.Subtract(identity);
}
}

SymbolicProblem PrepareSymbolicProblem(mlir::ModuleOp module,TargetSpec const& target,
    ModelDims dims,int grid,int residency,int kappa,SymbolicPriceCache* cache) {
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
  auto symbolic_model=model;
  symbolic_model.dims.seq_parameter=model.seq_metric_parameter;
  symbolic_model.dims.past_parameter=model.past_metric_parameter;
  auto projection=ProjectRuntimeQueues(symbolic_model,runtime,po);
  std::vector<int> counts,offsets{0};
  for(auto const& stage:projection.stages){counts.push_back(int(stage.task_count.Eval(model.MetricBindings())));offsets.push_back(offsets.back()+counts.back());}
  std::optional<analysis::OperatorGraph> semantic_graph;
  semantic_graph=InstantiateModelTasks(model,geometry);
  projection.dependencies=ExactRuntimeDependencies(model,*semantic_graph,projection,threads);
  struct Prices {std::vector<double> task_ns,prefetch_ns;} input;input.task_ns.resize(offsets.back());
  input.prefetch_ns.resize(offsets.back());
  CostModel cost(target,model.dtype);
  auto theta=model.MetricBindings();
  std::ostringstream environment;
  environment<<target.ToJson()<<':'<<std::hexfloat<<model.LiveFootprintBytes()<<':'<<threads<<':'<<runtime.ownership_flags;
  for(auto const& [name,value]:theta.values)environment<<':'<<name<<'='<<value;
  // Keep the entire granularity environment in the key. This cache merges
  // repeated semantic stages, never guesses independence from other tiles.
  for(auto const& g:geometry)environment<<':'<<g.tile_m<<','<<g.tile_n<<','<<g.tile_k<<','<<g.stages<<','<<g.split_k;

  for (std::size_t s=0;s<counts.size();++s) {
    auto const& projected=projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto const& g=geometry.at(stage.IsCollective() ? stage.gemm : 0);
    std::ostringstream signature;signature<<environment.str()<<':'<<projected.combine<<':'<<counts[s]<<':'<<stage.width<<':'<<stage.extent<<':'<<stage.group<<':'<<g.tile_m<<','<<g.tile_n<<','<<g.tile_k<<','<<g.stages<<','<<g.split_k;
    for(auto const& sem:model.task_semantics)if(sem.stage==projected.logical_stage)
      signature<<'\n'<<analysis::SemanticSignature(sem.op);
    auto work_key=signature.str();auto key=work_key+"|residency="+std::to_string(residency);
    if(cache){auto found=cache->prices.find(key);if(found!=cache->prices.end()){
      if(found->second.size()!=std::size_t(counts[s]))throw std::logic_error("cached price count mismatch");
      std::copy(found->second.begin(),found->second.end(),input.task_ns.begin()+offsets[s]);continue;
    }}
    PreparedSymbolicWork prepared;
    auto cached_work=cache ? cache->work.find(work_key) : std::map<std::string,PreparedSymbolicWork>::iterator{};
    if(cache && cached_work!=cache->work.end())prepared=cached_work->second;
    else {
      prepared.coordinates.resize(counts[s]);
      if(projected.combine) {
        prepared.input=DeriveCombineTaskInput(model,projected.logical_stage,g,*semantic_graph,threads,
            runtime.ownership_flags & codegen::kCombinerTileOwnership,cost.options().fp32_partials);
        auto resources=codegen::ReadSimtTaskResources(codegen::TaskKind::kGemmCombine,threads);
        prepared.traits.threads=resources.threads;prepared.traits.smem_bytes=resources.shared_bytes;prepared.traits.shape_legal=true;
        for(int t=0;t<counts[s];++t)prepared.coordinates[t].Bind("q",t);
        if(prepared.input.work.task_count.Eval(theta)!=counts[s])throw std::invalid_argument("combine access ownership disagrees with projected count");
      } else {
        auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x){
          return x.stage==projected.logical_stage && (!stage.IsCollective() || x.op.kind==analysis::OperatorKind::kMatmul);
        });
        if(found==model.task_semantics.end())throw std::invalid_argument("stage lacks derived semantic task costs");
        prepared.input=DeriveModelTaskInput(model,*found,*semantic_graph,stage.IsCollective()?&g:nullptr);
        prepared.traits=ModelTaskTraits(model,projected.logical_stage,g);
        prepared.chunks=stage.IsCollective()?cost.Chunks(model.gemms.at(stage.gemm),g):1;
        if(prepared.input.scalar_access){for(int t=0;t<counts[s];++t)prepared.coordinates[t].Bind("q",t);}
        else {
          auto ownership=ProjectTaskOwnership(*found,prepared.input.task,stage,threads).BindParams(theta);
          auto names=ownership.RangeDimNames();
          for(auto const& [physical,logical]:ownership.Points()){
            if(physical.size()!=1 || physical[0]<0 || physical[0]>=counts[s])throw std::invalid_argument("task cost ownership disagrees with projected count");
            for(std::size_t i=0;i<names.size();++i)prepared.coordinates[physical[0]].Bind(names[i],logical[i]);
          }
        }
      }
      if(cache)cache->work.emplace(work_key,prepared);
    }
    std::vector<double> prefetch;PrefetchPricing pricing{0,&prefetch};
    auto prices=PriceTaskInstances(cost,prepared.input,prepared.traits,{residency},model,prepared.chunks,prepared.coordinates,residency,&pricing);
    if(cache)cache->prices.emplace(key,prices);
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
    bool all=oracle->reverse.IsAllBox({{0,result.spaces[p].count-1}},result.theta) &&
             oracle->forward.IsAllBox({{0,result.spaces[c].count-1}},result.theta);
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
