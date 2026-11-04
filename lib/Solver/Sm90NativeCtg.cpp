// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <algorithm>
#include <limits>
#include <sstream>

namespace tilemega::solver {
namespace {
analysis::CouplingRelation AttentionOwnership(analysis::OperatorNode const& task,
    analysis::ParamBinding const& binding) {
  // Sm90DenseAttentionBody::Decompose uses ((g * Batch + b) * Chunks + c).
  // Tensor storage is b-major; flattening storage axes would assign the
  // wrong physical task indices for B > 1.
  std::vector<int> order;
  for(auto const* name:{"g","b","m"})
    for(std::size_t i=0;i<task.output.axes.size();++i)
      if(task.output.axes[i].name==name)order.push_back(i);
  for(std::size_t i=0;i<task.output.axes.size();++i)
    if(std::find(order.begin(),order.end(),int(i))==order.end())order.push_back(i);
  std::string flat="0",bounds,coordinates;
  long stride=1;
  for(auto it=order.rbegin();it!=order.rend();++it) {
    auto const& axis=task.output.axes[*it];
    auto count=task.CoordinateExtent(*it).Eval(binding,{});
    if(task.IsTiled(*it)) {
      flat+="+"+std::to_string(stride)+"*"+axis.name;
      bounds+=" and 0 <= "+axis.name+" < "+std::to_string(count);
    }
    stride*=count;
  }
  for(auto const& name:task.Coordinates())coordinates+=(coordinates.empty()?"":",")+name;
  return analysis::CouplingRelation::FromIslText(
      "{ [q] -> ["+coordinates+"] : q = "+flat+bounds+" }");
}
std::string SpaceKey(analysis::SemanticOp const& semantic,
    analysis::OperatorNode const& task,analysis::Granularity const& granularity,int batch) {
  std::ostringstream key;
  key<<analysis::SemanticSignature(semantic)<<'\n'<<task.name.substr(semantic.name.size())
     <<"\nbatch="<<batch<<";past=575;";
  if(auto it=granularity.tiles.find(semantic.name);it!=granularity.tiles.end())
    for(auto const& [name,tile]:it->second)key<<name<<'='<<tile.ToString()<<';';
  if(auto it=granularity.reduction_chunk.find(semantic.name);it!=granularity.reduction_chunk.end())
    key<<"split="<<it->second.ToString()<<';';
  for(std::size_t axis=0;axis<task.output.axes.size();++axis)
    key<<task.output.axes[axis].name<<'='<<task.output.axes[axis].extent.ToString()
       <<'/'<<task.tile[axis].ToString()<<';';
  return key.str();
}
}

Sm90NativeCtg InstantiateSm90NativeCtg(frontend::ImportedSemantics const& input,
    Sm90DenseCoordinates const& c,int batch,mlir::MLIRContext& context,Sm90NativeCtgCache& cache) {
  if(c.moe)throw std::invalid_argument(
      "native MoE CTG requires explicit routing and combine semantics");
  SolverPhase total(&cache.timing,"native_ctg_total");
  auto imported=input;
  bool relift=false;
  for(auto& gemm:imported.plan.gemms) {
    if(gemm.epilogue==frontend::PlanGemm::Epilogue::kArgmaxPartial &&
       gemm.partial_tile_n!=unsigned(c.gemms[4].tile_n)) {
      gemm.partial_tile_n=c.gemms[4].tile_n;relift=true;
    }
    if(gemm.epilogue==frontend::PlanGemm::Epilogue::kSwiGLU &&
       gemm.interleave_u!=unsigned(c.swiglu_u)) {
      gemm.interleave_u=c.swiglu_u;relift=true;
    }
  }
  if(relift)imported.lifted=frontend::LiftSemantics(imported.plan,imported.lift_options);
  frontend::ImportOptions options;options.phase_batch=batch;
  for(auto const& gemm:imported.plan.gemms) {
    auto const& geometry=c.gemms[Sm90DenseFamily(imported.plan,gemm)];
    options.gemms.push_back({16,geometry.tile_n,geometry.TileK(),geometry.slots,1,kSm90DenseGemmImpl});
  }
  auto granularity=frontend::LaunchGranularity(imported.lifted,imported.plan,options.gemms);
  for(auto const& op:imported.lifted.ops)
    if(!c.argmax_stage && (op.role==frontend::OpRole::kServingArgmax ||
       (c.embed_la && op.role==frontend::OpRole::kEmbedding)))
      granularity.Tile(op.name,"m",analysis::ClosedForm::Constant(batch));
  // Native attention executes Chunks partial tasks and one group reducer.
  // The ordinary serving importer only describes the unsplit result. Use
  // the declared reduction transform so all outgoing uses name its reducer.
  if(c.attention.chunks>1)for(auto& op:imported.lifted.sem.ops) {
    auto origin=std::find_if(imported.lifted.ops.begin(),imported.lifted.ops.end(),
        [&](auto const& item){return item.name==op.name;});
    if(origin==imported.lifted.ops.end() || origin->role!=frontend::OpRole::kServingFusedAttention)
      continue;
    auto z=op.Dim("z");
    if(!z)throw std::invalid_argument("native attention has no KV reduction domain");
    op.reduction={"z","flash_combine",op.name+".partial",op.name+".combine",true,{}};
    granularity.Split(op.name,z->extent.CeilDiv(analysis::ClosedForm::Constant(c.attention.chunks)));
  }
  options.launch_granularity=granularity;options.combiner_tile_per_block=true;
  Sm90NativeCtg result;
  auto graph=analysis::Instantiate(imported.lifted.sem,granularity);
  std::vector<std::string> keys;
  std::map<std::string,int> index;
  std::map<std::pair<int,int>,std::string> edge_keys;
  std::string binding_key=imported.lift_options.batch_symbol+":"+
      imported.lift_options.past_symbol+":"+std::to_string(batch);
  bool cached=cache.bindings.count(binding_key);
  for(auto const& task:graph.nodes) {
    auto semantic=std::find_if(imported.lifted.sem.ops.begin(),imported.lifted.sem.ops.end(),
        [&](auto const& op){return op.name==task.name || op.reduction.combiner==task.name;});
    if(semantic==imported.lifted.sem.ops.end())
      throw std::invalid_argument("native task lacks a semantic cache identity");
    keys.push_back(SpaceKey(*semantic,task,granularity,batch));
    index.emplace(task.name,keys.size()-1);
    bool hit=cache.spaces.count(keys.back());cached &= hit;
    hit?++cache.space_hits:++cache.space_misses;
  }
  for(std::size_t consumer=0;consumer<graph.nodes.size();++consumer) {
    auto const& task=graph.nodes[consumer];
    for(std::size_t operand=0;operand<task.operands.size();++operand) {
      auto found=index.find(task.operands[operand].producer);
      if(found==index.end())continue;
      auto pair=std::make_pair(found->second,int(consumer));
      auto [entry,inserted]=edge_keys.emplace(pair,keys[pair.first]+"\n->\n"+keys[pair.second]);
      entry->second+="\noperand="+std::to_string(operand);
    }
  }
  for(auto const& [pair,key]:edge_keys) {
    bool hit=cache.edges.count(key);cached &= hit;
    hit?++cache.edge_hits:++cache.edge_misses;
  }
  // U2: an instantiated family/geometry contributes immutable ownership,
  // count and coupling text. Recombining those witnesses skips MLIR building.
  // A missing witness takes the exact importer path and populates the cache.
  if(cached) {
    SolverPhase phase(&cache.timing,"native_ctg_cached_composition");
    result.binding=cache.bindings.at(binding_key);
    for(std::size_t i=0;i<graph.nodes.size();++i) {
      auto item=cache.spaces.at(keys[i]);item.name=graph.nodes[i].name;
      auto origin=std::find_if(imported.lifted.ops.begin(),imported.lifted.ops.end(),
          [&](auto const& op){return op.name==item.name || op.name+".combine"==item.name;});
      item.stage=origin->stage;result.spaces.push_back(std::move(item));
    }
    for(auto const& [pair,key]:edge_keys)
      result.edges.push_back({pair.first,pair.second,cache.edges.at(key)});
    return result;
  }
  result.module=frontend::TorchExportImporter{}.InstantiateForGranularity(
      imported,context,options,&cache.coupling,nullptr,&cache.timing);
  ModelDims dims;dims.batch=batch;dims.seq=1;dims.past=575;dims.total=576;
  auto model=[&]{SolverPhase phase(&cache.timing,"native_ctg_model");
    return ModelDescription::FromCouplingGraph(*result.module,dims,"native");}();
  result.binding=model.MetricBindings();
  cache.bindings[binding_key]=result.binding;
  for(auto const& task:graph.nodes) {
    SolverPhase phase(&cache.timing,"native_ctg_ownership");
    auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
        [&](auto const& s){return s.op.name==task.name || s.op.reduction.combiner==task.name;});
    auto origin=std::find_if(imported.lifted.ops.begin(),imported.lifted.ops.end(),
        [&](auto const& op){return op.name==task.name || op.name+".combine"==task.name;});
    if(semantic==model.task_semantics.end() || origin==imported.lifted.ops.end())
      throw std::invalid_argument("native CTG task has no imported semantic origin");
    auto ownership=origin->role==frontend::OpRole::kServingFusedAttention?
        AttentionOwnership(task,result.binding):
        ProjectTaskOwnership(*semantic,task,model.stages.at(semantic->stage),160)
            .BindParams(result.binding);
    auto count=ownership.Reverse().ImageCard().Eval(result.binding);
    if(count<1 || count>std::numeric_limits<int>::max())
      throw std::invalid_argument("native CTG task count is outside the flow domain");
    auto role=task.name==semantic->op.reduction.combiner?
        frontend::OpRole::kServingMerge:origin->role;
    result.spaces.push_back({task.name,semantic->stage,int(count),role,std::move(ownership)});
    cache.spaces.insert_or_assign(keys[result.spaces.size()-1],result.spaces.back());
  }
  std::map<std::pair<int,int>,std::vector<analysis::CouplingRelation>> grouped;
  for(auto const& edge:model.coupling_metrics.edges) {
    SolverPhase phase(&cache.timing,"native_ctg_edges");
    int producer=index.at(edge.producer_task),consumer=index.at(edge.consumer_task);
    auto const& p=result.spaces[producer];auto const& q=result.spaces[consumer];
    auto relation=q.ownership.ApplyRange(edge.relation.BindParams(result.binding))
        .ApplyRange(p.ownership.Reverse()).Reverse();
    grouped[{producer,consumer}].push_back(std::move(relation));
  }
  for(auto const& [pair,relations]:grouped) {
    result.edges.push_back({pair.first,pair.second,relations.size()==1?relations.front():
        analysis::CouplingRelation::UnionAll(relations)});
    cache.edges.insert_or_assign(edge_keys.at(pair),result.edges.back().relation);
  }
  return result;
}

std::shared_ptr<std::vector<std::pair<int,int>> const> Sm90NativeReleaseTable(
    analysis::CouplingRelation const& relation,int producers,int consumers,int kappa,
    analysis::ParamBinding const& binding,Sm90NativeCtgCache& cache) {
  if(producers<1 || consumers<1 || kappa<1)
    throw std::invalid_argument("native release table requires positive task and event extents");
  auto bound=relation.BindParams(binding);
  std::ostringstream key;
  key<<bound.ToString()<<'\n'<<producers<<':'<<consumers<<':'<<kappa;
  auto found=cache.releases.find(key.str());
  if(found!=cache.releases.end()) {++cache.release_hits;return found->second;}
  ++cache.release_misses;
  auto oracle=cache.coupling.OracleFor(bound.ToString());
  auto sorted=std::make_shared<std::vector<std::pair<int,int>>>();
  for(int consumer=0;consumer<consumers;++consumer) {
    auto release=oracle->reverse.LinearRelease({consumer});
    if(release.maximum<0)continue;
    if(release.maximum>=producers)
      throw std::invalid_argument("native release exceeds the CTG producer extent");
    sorted->emplace_back(CoarsenRelease(release.maximum,producers,kappa),consumer);
  }
  std::sort(sorted->begin(),sorted->end());
  cache.releases.emplace(key.str(),sorted);
  return sorted;
}
} // namespace tilemega::solver
