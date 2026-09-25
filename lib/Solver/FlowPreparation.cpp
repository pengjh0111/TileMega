// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/CacheServiceCurve.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Analysis/ISLContext.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/point.h>
#include <isl/val.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
namespace tilemega::solver {
namespace {
analysis::CouplingRelation ReadMap(isl_map* map) {
  char* text=isl_map_to_str(map);isl_map_free(map);
  if(!text)throw std::runtime_error("cannot serialize flow relation");
  auto result=analysis::CouplingRelation::FromIslText(text);free(text);return result;
}
std::string GeometryKey(BackendTraits const& t,int chunks) {
  return std::to_string(t.tile_m)+","+std::to_string(t.tile_n)+","+std::to_string(t.tile_k)+","+std::to_string(t.stages)+","+std::to_string(chunks);
}
}
SymbolicProblem PrepareFlowStructure(SymbolicProblem const& base,std::vector<GemmConfig> const& geometry,
    int workers,int kappa,analysis::CouplingCache& cache,FlowPreparationCache* prepared) {
  SymbolicProblem result;result.model=base.model;result.runtime=base.runtime;result.geometry=geometry;result.threads=base.threads;
  result.projection.options={workers,result.threads,kappa};result.projection.options.count_wait_entries=false;
  for(std::size_t i=0;i<geometry.size();++i){auto const& g=geometry[i];auto& r=result.runtime.gemms[i];r.tile_m=g.tile_m;r.tile_n=g.tile_n;r.tile_k=g.tile_k;r.stages=g.stages;r.split_k=g.split_k;}
  for(auto const& a:result.runtime.attention)if(a.chunks>1)throw std::invalid_argument("flow structure needs explicit expanded attention phases");
  auto graph=InstantiateModelTasks(result.model,geometry);auto theta=result.model.MetricBindings();
  analysis::SemanticGraph semantics;analysis::Granularity granularity;
  std::map<std::string,int> logical;
  for(auto const& sem:result.model.task_semantics){semantics.ops.push_back(sem.op);logical[sem.op.name]=sem.stage;if(!sem.op.reduction.combiner.empty())logical[sem.op.reduction.combiner]=sem.stage;
    auto const* node=graph.Find(sem.op.name);if(!node)throw std::runtime_error("missing flow task");
    for(std::size_t a=0;a<sem.op.result.axes.size();++a)granularity.Tile(sem.op.name,sem.op.result.axes[a].name,node->tile[a]);
    auto const& stage=result.model.stages[sem.stage];
    if(sem.op.reduction.splittable && stage.gemm>=0){auto const& g=geometry[stage.gemm];auto const* reduction=sem.op.Dim(sem.op.reduction.dim);long extent=reduction->extent.Eval({},{});long chunks=std::min<long>(g.split_k,(extent+g.tile_k-1)/g.tile_k);if(chunks>1)granularity.Split(sem.op.name,reduction->extent.CeilDiv(analysis::ClosedForm::Constant(chunks)));}
  }
  auto ownership=[&](ModelTaskSemantics const& sem,analysis::OperatorNode const& node,ModelStage const& stage){
    std::string signature;
    if(prepared){auto it=prepared->signatures.find(sem.op.name);if(it==prepared->signatures.end())it=prepared->signatures.emplace(sem.op.name,analysis::SemanticSignature(sem.op)).first;signature=it->second;}
    else signature=analysis::SemanticSignature(sem.op);
    std::ostringstream key;key<<signature<<':'<<sem.element_chunk<<':'<<int(stage.kind)<<':'<<stage.width<<':'<<stage.extent<<':'<<result.threads;
    for(std::size_t a=0;a<node.tile.size();++a)key<<':'<<node.tile[a].ToIslText()<<':'<<node.output.axes[a].extent.ToIslText()<<':'<<node.output.axes[a].origin.ToIslText();
    if(prepared){auto found=prepared->ownership.find(key.str());if(found!=prepared->ownership.end())return found->second;}
    auto map=ProjectTaskOwnership(sem,node,stage,result.threads);FlowPreparationCache::OwnershipEntry out{map,map.Reverse().ImageCard()};
    if(prepared)prepared->ownership.emplace(key.str(),out);return out;
  };
  std::vector<int> entry(result.model.stages.size()),done(entry);result.offsets={0};
  auto append=[&](int stage,bool combine,FlowPreparationCache::OwnershipEntry const& ownership){auto count=ownership.count;result.projection.stages.push_back({stage,combine,count});result.counts.push_back(count.Eval(theta));result.offsets.push_back(result.offsets.back()+result.counts.back());};
  for(std::size_t stage=0;stage<result.model.stages.size();++stage) {
    auto sem=std::find_if(result.model.task_semantics.begin(),result.model.task_semantics.end(),[&](auto const& s){return s.stage==int(stage) && (!result.model.stages[stage].IsCollective() || s.op.kind==analysis::OperatorKind::kMatmul);});
    if(sem==result.model.task_semantics.end())throw std::runtime_error("missing flow stage semantic");
    auto* node=graph.Find(sem->op.name);entry[stage]=result.counts.size();append(stage,false,ownership(*sem,*node,result.model.stages[stage]));done[stage]=entry[stage];
    if(auto* combine=graph.Find(sem->op.reduction.combiner)){auto owner=*sem;owner.element_chunk=!result.model.combiner_tile_ownership;done[stage]=result.counts.size();append(stage,true,ownership(owner,*combine,result.model.stages[stage]));}
  }
  struct Owner{int stage;analysis::CouplingRelation map;};std::map<std::string,Owner> owners;
  for(auto const& node:graph.nodes){int stage=logical.at(node.name);auto sem=std::find_if(result.model.task_semantics.begin(),result.model.task_semantics.end(),[&](auto const& s){return s.op.name==node.name || s.op.reduction.combiner==node.name;});
    auto declaration=*sem;bool combine=sem->op.reduction.combiner==node.name;if(combine)declaration.element_chunk=!result.model.combiner_tile_ownership;
    int physical=sem->op.kind==analysis::OperatorKind::kMatmul && !combine?entry[stage]:done[stage];owners.emplace(node.name,Owner{physical,ownership(declaration,node,result.model.stages[stage]).map});}
  auto edges=cache.Derive(semantics,graph,granularity,{});
  std::map<std::pair<int,int>,std::vector<analysis::CouplingRelation>> grouped;
  for(auto const& edge:edges){auto const& p=owners.at(edge.src.name);auto const& c=owners.at(edge.dst.name);if(p.stage==c.stage)continue;
    std::string key=c.map.ToString()+"\n"+edge.C.ToString()+"\n"+p.map.ToString();
    analysis::CouplingRelation relation;auto found=prepared?prepared->projected.find(key):std::map<std::string,analysis::CouplingRelation>::iterator{};
    if(prepared && found!=prepared->projected.end())relation=found->second;
    else{relation=c.map.ApplyRange(edge.C).ApplyRange(p.map.Reverse()).Reverse();if(prepared)prepared->projected.emplace(std::move(key),relation);}
    grouped[{p.stage,c.stage}].push_back(std::move(relation));}
  for(auto const& [pair,relations]:grouped)result.data_edges.push_back({pair.first,pair.second,(relations.size()==1?relations.front():analysis::CouplingRelation::UnionAll(relations))});
  return result;
}
PreparedFlow PrepareFlow(SymbolicProblem const& problem,analysis::DramFloor const& floor,
    TargetSpec const& target,int residency,HopCurve const& hop,
    analysis::CouplingCache& coupling,FlowPreparationCache& cache,bool colocate) {
  if(problem.model.dtype!=ScalarType::kBF16)throw std::invalid_argument("flow preparation requires BF16");
  auto target_key=target.ToJson();if(cache.target_key!=target_key){cache={};cache.target_key=std::move(target_key);}
  PreparedFlow result;auto& flow=result.flow;auto model=problem.model;model.metric_bindings.values.erase("Tm");model.metric_bindings.values.erase("Tn");auto theta=model.MetricBindings();
  flow.workers=target.res.num_sms*residency;SetFlowCalibration(flow,target,model.dtype,hop);
  auto bound=floor.Evaluate(theta);flow.dram_floor_ns=bound.dram_ns;flow.floor_ns=bound.floor_ns;
  auto const& cal=target.CalibrationFor("bf16");
  flow.all_external_miss=CacheServiceCurve(cal.l2_curve_bytes,cal.l2_curve_gbps).HitFraction(bound.read_bytes,cal.l2_gbps,cal.dram_gbps)==0;
  auto graph=InstantiateModelTasks(model,problem.geometry);
  CostModelOptions options;options.regime_a=true;CostModel cost(target,model.dtype,options);
  std::vector<std::string> signatures,geometry_keys;
  std::vector<int> order(problem.counts.size());int ordinal=0;
  for(auto const& logical:BuildVariantStageSchedule(problem.runtime.dependencies,model.stages.size()).schedule)
    for(std::size_t s=0;s<problem.counts.size();++s)if(problem.projection.stages[s].logical_stage==int(logical.stage))order[s]=ordinal++;
  for(std::size_t s=0;s<problem.counts.size();++s) {
    auto const& projected=problem.projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& sem){return sem.stage==projected.logical_stage && (!stage.IsCollective() || sem.op.kind==analysis::OperatorKind::kMatmul);});
    if(found==model.task_semantics.end())throw std::runtime_error("flow task has no semantics");
    auto semantic=*found;auto const& g=problem.geometry.at(stage.IsCollective()?stage.gemm:0);
    std::ostringstream space_key;space_key<<analysis::SemanticSignature(semantic.op)<<':'<<projected.combine<<':'<<residency<<':'<<problem.threads<<':'<<std::hexfloat<<bound.read_bytes;
    if(stage.IsCollective())space_key<<':'<<g.tile_m<<':'<<g.tile_n<<':'<<g.tile_k<<':'<<g.stages<<':'<<g.split_k;
    else for(auto const& [axis,tile]:semantic.tiles)space_key<<':'<<axis<<'='<<tile.ToIslText();
    for(auto const& [name,value]:std::map<std::string,long>(theta.values.begin(),theta.values.end()))space_key<<':'<<name<<'='<<value;
    for(auto const& operand:semantic.op.operands){auto f=floor.tensors.find(operand.tensor.name);if(f!=floor.tensors.end())space_key<<':'<<f->second.no_producer.ToString()<<':'<<f->second.writes.ToString()<<':'<<f->second.element_bytes;}
    auto output=floor.tensors.find(semantic.op.result.name);if(output!=floor.tensors.end())space_key<<":"<<output->second.external_writes.ToString();

    auto hit=cache.spaces.find(space_key.str());
    if(hit!=cache.spaces.end()) {
      ++cache.space_hits;auto space=hit->second.space;space.name=semantic.op.name+(projected.combine?".combine":"");space.order=order[s];
      if(space.count!=problem.counts[s])throw std::runtime_error("cached space count changed");
      if(hit->second.prices.coordinate_varying)result.varying_spaces.push_back(space.name);
      signatures.push_back(analysis::SemanticSignature(semantic.op)+(projected.combine?".combine":""));geometry_keys.push_back(hit->second.geometry_key);
      result.prices.push_back(hit->second.prices);flow.spaces.push_back(std::move(space));continue;
    }
    ++cache.space_misses;
    DerivedTaskInput input;BackendTraits traits;int chunks=1;
    if(projected.combine) {
      input=DeriveCombineTaskInput(model,projected.logical_stage,g,graph,problem.threads,problem.runtime.ownership_flags & codegen::kCombinerTileOwnership,cost.options().fp32_partials);
      auto resource=codegen::ReadSimtTaskResources(codegen::TaskKind::kGemmCombine,problem.threads);
      traits.threads=resource.threads;traits.smem_bytes=resource.shared_bytes;traits.shape_legal=true;
      // The combiner reads partials, not the original GEMM operands.
      semantic.op.name=input.task.name;semantic.op.kind=analysis::OperatorKind::kReduction;
      semantic.op.element_reads.clear();
    } else {
      input=DeriveModelTaskInput(model,semantic,graph,stage.IsCollective()?&g:nullptr);
      traits=ModelTaskTraits(model,projected.logical_stage,g);
      chunks=stage.IsCollective()?cost.Chunks(model.gemms.at(stage.gemm),g):1;
    }
    PiecePrices prices;
    try {BindTaskDramProvenance(input,semantic,floor,theta);
      prices=PriceBoundaryPieces(cost,input,semantic,traits,{residency},model,chunks,&cache.prices);
    }catch(std::exception const& e){throw std::runtime_error(input.task.name+": "+e.what());}
    FlowSpace space;space.name=input.task.name;space.category=projected.combine?"combine":semantic.op.arithmetic;
    if(!projected.combine && semantic.op.kind==analysis::OperatorKind::kMatmul && model.exported_tensors.count(semantic.op.result.name))space.category="lm_head";
    space.count=problem.counts[s];space.order=order[s];space.piece_of_task.assign(space.count,-1);
    analysis::CouplingRelation ownership;
    if(input.scalar_access)ownership=analysis::CouplingRelation::FromIslText("{ [q] -> [q] : 0<=q<"+std::to_string(space.count)+" }");
    else ownership=ProjectTaskOwnership(semantic,input.task,stage,problem.threads).BindParams(theta);
    for(std::size_t p=0;p<prices.pieces.size();++p) {
      auto const& piece=prices.pieces[p];space.pieces.push_back({int(piece.count.Eval(theta)),piece.parts});
      auto physical=ownership.ApplyRange(piece.domain).Reverse();
      // This enumerates the one-dimensional ownership image, never task-DAG
      // edges or operand elements. Price evaluation remains once per piece.
      auto* map=isl_map_read_from_str(analysis::SharedIslContext().raw(),physical.ToString().c_str());
      auto* set=isl_map_range(map);
      struct Fill {std::vector<int>* into;int piece;};Fill fill{&space.piece_of_task,int(p)};
      isl_set_foreach_point(set,[](isl_point* point,void* user)->isl_stat{
        auto* f=static_cast<Fill*>(user);auto* value=isl_point_get_coordinate_val(point,isl_dim_set,0);
        long t=isl_val_get_num_si(value);isl_val_free(value);isl_point_free(point);
        if(t<0 || t>=long(f->into->size()) || (*f->into)[t]!=-1)return isl_stat_error;
        (*f->into)[t]=f->piece;return isl_stat_ok;
      },&fill);isl_set_free(set);
    }
    if(std::find(space.piece_of_task.begin(),space.piece_of_task.end(),-1)!=space.piece_of_task.end())throw std::runtime_error("flow pieces do not cover runtime ownership: "+space.name);
    if(prices.coordinate_varying)result.varying_spaces.push_back(space.name);
    space.rank_ns=space.count?prices.total_isolated_ns/space.count:0;
    signatures.push_back(analysis::SemanticSignature(found->op)+(projected.combine?".combine":""));geometry_keys.push_back(GeometryKey(traits,chunks));
    cache.spaces.emplace(space_key.str(),FlowPreparationCache::SpaceEntry{space,prices,geometry_keys.back()});
    result.prices.push_back(std::move(prices));flow.spaces.push_back(std::move(space));
  }
  result.colocated_producer.assign(flow.spaces.size(),-1);
  auto data_edges=problem.data_edges;
  if(data_edges.empty()) {
    auto* raw=isl_map_read_from_str(analysis::SharedIslContext().raw(),problem.projection.dependencies.ToString().c_str());
    auto* pairs=isl_map_project_out(isl_map_copy(raw),isl_dim_in,1,1);pairs=isl_map_project_out(pairs,isl_dim_out,1,1);
    for(auto const& [cpoint,ppoint]:ReadMap(pairs).BindParams(theta).Points()) {
      int c=cpoint.at(0),p=ppoint.at(0);auto* edge=isl_map_fix_si(isl_map_copy(raw),isl_dim_in,0,c);edge=isl_map_fix_si(edge,isl_dim_out,0,p);
      edge=isl_map_project_out(edge,isl_dim_in,0,1);edge=isl_map_project_out(edge,isl_dim_out,0,1);edge=isl_map_reverse(edge);
      data_edges.push_back({p,c,ReadMap(edge)});
    }
    isl_map_free(raw);
  }
  for(auto const& data:data_edges) {
    int p=data.producer,c=data.consumer;auto const& relation=data.relation;auto oracle=coupling.OracleFor(relation.ToString());
    bool one=oracle->structure==analysis::EdgeStructure::OneToOne && oracle->forward.kind()==analysis::OracleKind::Unique && oracle->reverse.kind()==analysis::OracleKind::Unique;
    int previous=result.colocated_producer[c];if(colocate && one && (previous<0 || order[p]>order[previous]))result.colocated_producer[c]=p;
    int kappa=ProducerKappa(problem.projection.options,p);bool all=oracle->reverse.IsAllBox({{0,problem.counts[p]-1}},theta);
    std::string key=signatures[p]+"\n"+signatures[c]+"\n"+geometry_keys[p]+"|"+geometry_keys[c]+"|"+std::to_string(kappa)+"|"+relation.ToString();
    for(auto const& [name,value]:std::map<std::string,long>(theta.values.begin(),theta.values.end()))key+='|'+name+'='+std::to_string(value);
    auto it=cache.releases.find(key);
    std::shared_ptr<std::vector<std::pair<int,int>> const> sorted;
    if(it!=cache.releases.end()){++cache.release_hits;sorted=it->second;}else {
      ++cache.release_misses;auto values=std::make_shared<std::vector<std::pair<int,int>>>();bool nonprefix=false;
      for(int j=0;j<problem.counts[c];++j) {
        auto image=oracle->reverse.Query({j},theta);if(!image.Count())continue;
        int maximum=-1;
        if(image.rectangular){maximum=image.box.at(0).second;nonprefix|=image.box.at(0).first!=0;}
        else image.ForEach([&](auto const& point){maximum=std::max(maximum,int(point.at(0)));});
        nonprefix|=image.Count()!=maximum+1;
        values->emplace_back(CoarsenRelease(maximum,problem.counts[p],kappa),j);
      }
      if(nonprefix)++result.nonprefix_edges;
      std::sort(values->begin(),values->end());sorted=values;cache.releases.emplace(key,sorted);
    }
    flow.edges.push_back({p,c,kappa,all,false,std::move(sorted)});
  }
  for(auto& e:flow.edges)e.colocated=result.colocated_producer[e.consumer]==e.producer;
  std::vector<int> stages(flow.spaces.size());std::iota(stages.begin(),stages.end(),0);
  std::sort(stages.begin(),stages.end(),[&](int a,int b){return order[a]>order[b];});
  for(int s:stages){double tail=0;for(auto const& edge:flow.edges)if(edge.producer==s)tail=std::max(tail,flow.spaces[edge.consumer].rank_ns+flow.hop_ns);flow.spaces[s].rank_ns+=tail;}
  return result;
}
std::vector<TaskPriceParts> ExpandFlowPrices(PreparedFlow const& prepared) {
  std::vector<TaskPriceParts> result;
  for(auto const& s:prepared.flow.spaces)for(int p:s.piece_of_task)result.push_back(s.pieces.at(p).parts);
  return result;
}
void ApplyFlowPrices(SymbolicProblem& problem,PreparedFlow const& prepared,TargetSpec const& target,int residency) {
  auto parts=ExpandFlowPrices(prepared);problem.task_ns.clear();problem.prefetch_ns.clear();
  double rate=target.CalibrationFor("bf16").dram_gbps/(target.res.num_sms*residency);
  for(auto const& p:parts)problem.task_ns.push_back(IsolatedNs(p,rate));
}
}
