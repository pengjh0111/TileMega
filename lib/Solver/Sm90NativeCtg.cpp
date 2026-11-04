// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
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

static Sm90NativeCtg InstantiateSm90NativeBaseCtg(frontend::ImportedSemantics const& input,
    Sm90DenseCoordinates const& c,int batch,mlir::MLIRContext& context,Sm90NativeCtgCache& cache) {
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

Sm90NativeCtg InstantiateSm90NativeCtg(frontend::ImportedSemantics const& input,
    Sm90DenseCoordinates const& c,int batch,mlir::MLIRContext& context,
    Sm90NativeCtgCache& cache,int workers) {
  if(c.moe && workers<1)
    throw std::invalid_argument("native MoE local routing needs the physical worker count");
  auto base=InstantiateSm90NativeBaseCtg(input,c,batch,context,cache);
  if(!c.moe)return base;
  if(!c.expert_fuse)
    throw std::invalid_argument("native MoE unfused schedule publication is not implemented");
  auto work=Sm90DenseWorkloadFromPlan(input.plan,batch);
  auto gu_domain=Sm90NativeMoeDomain(work.experts,work.top_k,batch,
      work.shapes[6][0]/c.gemms[6].tile_n);
  auto dn_domain=Sm90NativeMoeDomain(work.experts,work.top_k,batch,
      work.shapes[7][0]/c.gemms[7].tile_n);
  Sm90NativeCtg result;result.binding=base.binding;
  result.approximation=json::Object{{"kind",kSm90NativeMoeDomainApproximation},
      {"gate_up",gu_domain.ToJson()},{"down",dn_domain.ToJson()},
      {"routing",c.RouteName()},{"route_tasks",c.GlobalRoute()?1:workers},{"route_price","older-body RouteLa minus grid barrier"}};
  // The generic importer supplies the exact dense/attention projections.
  // Native expert axes and residual def-use are reconstructed below; its
  // original MoE module is not presented as the projected executable IR.
  auto add=[&](Sm90NativeCtgSpace item) {
    auto key="native-moe-projection-v1:"+std::to_string(int(item.role))+":"+item.ownership.ToString();
    auto found=cache.spaces.find(key);
    if(found!=cache.spaces.end()) {item.count=found->second.count;++cache.space_hits;}
    else {
      ++cache.space_misses;
      item.count=int(item.ownership.Reverse().ImageCard().Eval(result.binding));
      if(item.count<1)throw std::invalid_argument("empty native predictive task domain");
      cache.spaces.emplace(key,item);
    }
    result.spaces.push_back(std::move(item));return int(result.spaces.size())-1;
  };
  auto identity=[](int count) {
    return analysis::CouplingRelation::FromIslText("{ [t] -> [j] : t=j and 0<=t<"+
        std::to_string(count)+" }");
  };
  auto edge=[&](int producer,int consumer,analysis::CouplingRelation relation) {
    if(producer<0 || consumer<=producer)
      throw std::invalid_argument("native MoE dependency has no preceding producer");
    auto key="native-moe-projection-v1:"+relation.ToString();
    auto [found,inserted]=cache.edges.emplace(key,relation);
    inserted?++cache.edge_misses:++cache.edge_hits;
    result.edges.push_back({producer,consumer,found->second});
  };
  auto all=[&](int producer,int consumer) {
    if(producer<0)throw std::invalid_argument("native MoE input has no producer");
    edge(producer,consumer,analysis::CouplingRelation::FromIslText(
        "{ [p] -> [q] : 0<=p<"+std::to_string(result.spaces[producer].count)+
        " and 0<=q<"+std::to_string(result.spaces[consumer].count)+" }"));
  };
  std::map<int,int> mapped;
  int previous=-1,output=-1,router=-1,route=-1,gate_up=-1;
  for(int original=0;original<int(base.spaces.size());++original) {
    auto item=base.spaces[original];auto const& stage=input.plan.stages.at(item.stage);
    int family=stage.kind==frontend::PlanTaskKind::kGemm?
        int(Sm90DenseFamily(input.plan,input.plan.gemms.at(stage.gemm))):-1;
    if(family==6)item.ownership=gu_domain.ownership;
    if(family==7)item.ownership=dn_domain.ownership;
    int current=add(std::move(item));mapped.emplace(original,current);
    auto const& task=result.spaces[current];
    if(task.role==frontend::OpRole::kEmbedding) {
      if(previous>=0)throw std::invalid_argument("unexpected native MoE embedding position");
    }else if(family==0 || family==4 || task.role==frontend::OpRole::kServingArgmax) {
      // Combine writes the in-place residual consumed by the next layer/head.
      all(previous,current);
    }else if(task.role==frontend::OpRole::kServingFusedAttention ||
             task.role==frontend::OpRole::kServingMerge) {
      int found=0;
      for(auto const& source:base.edges)if(source.consumer==original) {
        edge(mapped.at(source.producer),current,source.relation);++found;
      }
      if(!found)throw std::invalid_argument("native attention has no CTG producer");
    }else if(family==1) {
      all(previous,current);output=current;
    }else if(family==5) {
      if(previous!=output)throw std::invalid_argument("router does not follow its residual input");
      all(output,current);router=current;
      auto local=result.spaces[current];local.name+=c.GlobalRoute()?".global_route":".local_route";
      local.role=frontend::OpRole::kMoERouting;local.ownership=identity(c.GlobalRoute()?1:workers);
      route=add(std::move(local));all(router,route);previous=route;continue;
    }else if(family==6) {
      if(previous!=route)throw std::invalid_argument("experts do not follow local routing");
      all(output,current);
      // Each CTA builds the route table it uses for its own grid-stride tasks.
      if(c.GlobalRoute())all(route,current);
      else edge(route,current,analysis::CouplingRelation::FromIslText(
          "{ [p] -> [q] : 0<=p<"+std::to_string(workers)+" and 0<=q<"+
          std::to_string(gu_domain.tasks)+" and q % "+std::to_string(workers)+" = p }"));
      gate_up=current;
    }else if(family==7) {
      if(previous!=gate_up)throw std::invalid_argument("down does not follow its gate_up");
      edge(gate_up,current,Sm90NativeMoeExpertCoupling(gu_domain,dn_domain));
      // A CTA with only down work still performs its own local route build.
      if(c.GlobalRoute())all(route,current);
      else edge(route,current,analysis::CouplingRelation::FromIslText(
          "{ [p] -> [q] : 0<=p<"+std::to_string(workers)+" and 0<=q<"+
          std::to_string(dn_domain.tasks)+" and (q+"+std::to_string(gu_domain.tasks)+") % "+
          std::to_string(workers)+" = p }"));
      auto combine=result.spaces[current];combine.name+=".combine";
      combine.role=frontend::OpRole::kMoECombine;combine.ownership=identity(dn_domain.tiles);
      int reduction=add(std::move(combine));
      edge(current,reduction,Sm90NativeMoeCombineCoupling(dn_domain));
      // The residual add reads the same column interval from O's output.
      int ot=c.gemms[1].tile_n,dt=c.gemms[7].tile_n;
      edge(output,reduction,analysis::CouplingRelation::FromIslText(
          "{ [p] -> [q] : 0<=p<"+std::to_string(result.spaces[output].count)+
          " and 0<=q<"+std::to_string(dn_domain.tiles)+" and p*"+std::to_string(ot)+
          "<(q+1)*"+std::to_string(dt)+" and q*"+std::to_string(dt)+"<(p+1)*"+
          std::to_string(ot)+" }"));
      previous=reduction;continue;
    }else throw std::invalid_argument("unrecognized native MoE predictive stage");
    previous=current;
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
