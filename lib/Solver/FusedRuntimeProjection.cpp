// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <limits>

#ifndef TILEMEGA_FUSED_RUNTIME_PROJECTION
#define TILEMEGA_FUSED_RUNTIME_PROJECTION 1
#endif

namespace tilemega::solver {
WrittenFusionProjection ProjectWrittenFusionQueues(mlir::ModuleOp module,
    ModelDims dims,RuntimeProjectionOptions options) {
  analysis::IslReferenceAudit audit(__func__);
  auto context=ModelDescription::FromFusionPhases(module,std::move(dims),"fusion-runtime-phases");
  auto source=codegen::ReadFusionSourcePlan(module);
  if (context.dims.IsSymbolic() &&
      ((!context.dims.seq_parameter.empty() && context.dims.seq_parameter!=context.seq_metric_parameter) ||
       (!context.dims.past_parameter.empty() && context.dims.past_parameter!=context.past_metric_parameter)))
    throw std::invalid_argument("symbolic fusion projection requires the CG dimension-role names");
  WrittenFusionProjection result;
  result.projection=ProjectRuntimeQueues(context,source,options);
  auto known=context.metric_bindings;
  if (context.dims.IsSymbolic()) {
    auto fixed=context;
    fixed.dims.seq_parameter.clear(); fixed.dims.past_parameter.clear();
    fixed.dims.total=fixed.dims.seq+fixed.dims.past;
    known=fixed.MetricBindings();
    std::set<std::string> symbolic{"L_s"};
    if (!context.dims.seq_parameter.empty()) {
      symbolic.insert("S"); symbolic.insert(context.seq_metric_parameter);
    }
    if (!context.dims.past_parameter.empty()) {
      symbolic.insert("P"); symbolic.insert("past"); symbolic.insert(context.past_metric_parameter);
    }
    for (auto const& name:symbolic) known.values.erase(name);
    for (auto const& [alias,canonical]:context.metric_aliases) {
      if (symbolic.count(canonical)) known.values.erase(alias);
    }
  } else known=context.MetricBindings();
  for (auto const& input:ReadFusedTaskInputs(module)) {
    auto const& p=input.semantics.at(0); auto const& c=input.semantics.at(1);
    if (c.stage!=p.stage+1)
      throw std::invalid_argument("runtime fusion requires adjacent distinct stages");
    auto po=ProjectTaskOwnership(p,input.phases.at(0).task,context.stages.at(p.stage),options.threads);
    auto co=ProjectTaskOwnership(c,input.phases.at(1).task,context.stages.at(c.stage),options.threads);
    auto mapping=co.ApplyRange(input.phase_maps.at(0)).ApplyRange(po.Reverse()).BindParams(known);
    int producer=-1,consumer=-1;
    for (int stage=0;stage<int(result.projection.stages.size());++stage) {
      if (result.projection.stages[stage].logical_stage==p.stage) producer=stage;
      if (result.projection.stages[stage].logical_stage==c.stage) consumer=stage;
    }
    auto live=[&](int stage) {
      return result.projection.tasks.IntersectRange("{ [s,t] : s="+std::to_string(stage)+" }")
          .ApplyRange(analysis::CouplingRelation::FromIslText("{ [s,t] -> [t] }")).ImageIdentity();
    };
    mapping=live(consumer).ApplyRange(mapping).ApplyRange(live(producer));
    result.projection=FuseProjectedQueues(result.projection,producer,consumer,mapping,options,true).projection;
    result.original_stages.emplace_back(p.stage,c.stage);
    result.consumer_to_producer.push_back(std::move(mapping));
  }
  // Retained scheduling provenance may overapproximate C, but must never
  // omit a dependency in the replacement CG. Check it after ownership and
  // stage projection; a malformed source-plan attribute must not weaken sync.
  std::vector<GemmConfig> configs;
  for (auto const& g:source.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  auto graph=InstantiateModelTasks(context,configs);
  struct Endpoint { int stage; analysis::CouplingRelation ownership; };
  std::map<std::string,Endpoint> endpoints;
  auto endpoint=[&](std::string const& symbol,ModelTaskSemantics const& semantic,
                    analysis::OperatorNode const& node) {
    int projected=-1;
    for (int s=0;s<int(result.projection.stages.size());++s)
      if (result.projection.stages[s].logical_stage==semantic.stage &&
          !result.projection.stages[s].combine) projected=s;
    if (projected<0) throw std::invalid_argument("fusion erased an externally required runtime phase");
    endpoints.emplace(symbol,Endpoint{projected,ProjectTaskOwnership(semantic,node,
        context.stages.at(semantic.stage),options.threads).BindParams(known)});
  };
  for (auto task:module.getOps<dialect::TileSpaceOp>()) {
    auto name=task.getOperatorName().str();
    auto semantic=std::find_if(context.task_semantics.begin(),context.task_semantics.end(),
        [&](auto const& entry) { return entry.op.name==name; });
    auto const* node=graph.Find(name);
    if (semantic==context.task_semantics.end() || !node)
      throw std::invalid_argument("replacement CG lacks runtime phase semantics");
    endpoint(task.getSymName().str(),*semantic,*node);
  }
  for (auto const& input:ReadFusedTaskInputs(module))
    endpoint(input.name,input.semantics.at(1),input.phases.at(1).task);
  for (auto edge:module.getOps<dialect::CouplingOp>()) {
    auto const& p=endpoints.at(edge.getSrc().str());
    auto const& c=endpoints.at(edge.getDst().str());
    if (p.stage==c.stage) continue;
    auto exact=c.ownership.ApplyRange(edge.getRelation().getMap().BindParams(known))
        .ApplyRange(p.ownership.Reverse());
    auto required=analysis::CouplingRelation::FromIslText(
        "{ [s="+std::to_string(c.stage)+",t] -> [t] }").ApplyRange(exact)
        .ApplyRange(analysis::CouplingRelation::FromIslText(
            "{ [t] -> [s="+std::to_string(p.stage)+",t] }"));
    auto live=result.projection.tasks.ImageIdentity();
    required=live.ApplyRange(required).ApplyRange(live);
    if (!required.IsSubset(result.projection.dependencies))
      throw std::invalid_argument("fused runtime projection omits replacement CG dependency "+
          edge.getSrc().str()+" -> "+edge.getDst().str()+": "+
          required.Subtract(result.projection.dependencies).ToString());
  }
  return result;
}
FusedRuntimeProjection FuseProjectedQueues(RuntimeProjection const& original,
    int producer,int consumer,analysis::CouplingRelation const& coupling,
    RuntimeProjectionOptions options,bool optional_producer) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_FUSED_RUNTIME_PROJECTION
  throw std::invalid_argument("fused runtime projection disabled");
#endif
  using R=analysis::CouplingRelation;
  auto parse=[](std::string const& text) { return R::FromIslText(text); };
  auto equal=[](R const& a,R const& b) { return a.IsSubset(b) && b.IsSubset(a); };
  if (producer<0 || consumer!=producer+1 || consumer>=int(original.stages.size()) ||
      options.grid<=0 || options.threads<=0 || options.kappa<0 ||
      options.grid!=original.options.grid || options.threads!=original.options.threads ||
      options.kappa!=original.options.kappa ||
      options.cg_split_task_order!=original.options.cg_split_task_order ||
      options.partition_worker_counts!=original.options.partition_worker_counts ||
      options.split_count_periods!=original.options.split_count_periods ||
      options.force_all_dependencies!=original.options.force_all_dependencies)
    throw std::invalid_argument("fusion projection requires an adjacent pair and matching runtime options");
  if (coupling.DomainDimNames().size()!=1 || coupling.RangeDimNames().size()!=1 ||
      !coupling.IsSingleValued())
    throw std::invalid_argument("fusion runtime ownership must map one consumer to one producer task");
  auto stage_set=[&](int stage) {
    return original.tasks.IntersectRange("{ [s,t] : s="+std::to_string(stage)+" }");
  };
  auto p=std::to_string(producer),c=std::to_string(consumer);
  auto p_tasks=stage_set(producer).ApplyRange(parse("{ [s,t] -> [t] }"));
  auto c_tasks=stage_set(consumer).ApplyRange(parse("{ [s,t] -> [t] }"));
  if (!equal(coupling.Image(),p_tasks.Image()) ||
      !(optional_producer ? coupling.Reverse().Image().IsSubset(c_tasks.Image()) :
                            equal(coupling.Reverse().Image(),c_tasks.Image())))
    throw std::invalid_argument("fusion runtime mapping does not cover both phase task spaces");
  auto internal=original.dependencies.IntersectDomain("{ [s,t] : s="+c+" }")
      .IntersectRange("{ [s,t] : s="+p+" }");
  auto old_c_to_p=parse("{ [s="+c+",t] -> [t] }").ApplyRange(coupling)
      .ApplyRange(parse("{ [t] -> [s="+p+",t] }"));
  if (!old_c_to_p.IsSubset(internal))
    throw std::invalid_argument("fusion ownership is not covered by original dependencies");
  auto external=original.dependencies.Subtract(internal);
  auto outgoing=external.IntersectRange("{ [s,t] : s="+p+" }");
  if (!outgoing.IsSubset(parse("{ [s,t] -> [a,b] : false }")) &&
      !coupling.Reverse().IsSingleValued())
    throw std::invalid_argument("replicated fusion producer has external consumers");
  // Remove the producer stage, with the consumer taking its index. All
  // other phase identities survive unchanged apart from stage renumbering.
  auto renumber=original.tasks.ImageIdentity().ApplyRange(parse(
      "{ [s,t] -> [n,t] : s!="+p+" and ((s<"+p+
      " and n=s) or (s>"+p+" and n=s-1)) }"));
  FusedRuntimeProjection out;
  auto& result=out.projection;
  result.options=options;
  result.task_binding=original.task_binding;
  result.tasks=original.tasks.ApplyRange(renumber);
  auto new_c_to_p=renumber.Reverse().ApplyRange(old_c_to_p);
  out.phase_tasks=renumber.Reverse().Union(new_c_to_p);
  result.dependencies=out.phase_tasks.ApplyRange(external).ApplyRange(out.phase_tasks.Reverse());
  auto ordinary=original.ordinary_dependencies.value_or(original.dependencies).Subtract(internal);
  result.ordinary_dependencies=out.phase_tasks.ApplyRange(ordinary).ApplyRange(out.phase_tasks.Reverse());
  auto all_new=result.tasks.Image();
  if (!result.dependencies.Image().IsSubset(all_new) ||
      !result.dependencies.Reverse().Image().IsSubset(all_new))
    throw std::invalid_argument("fusion dependency escapes replacement task space");
  auto backwards=result.dependencies.Subtract(parse("{ [s,t] -> [a,b] : a<s }"));
  if (!backwards.IsSubset(parse("{ [s,t] -> [a,b] : false }")))
    throw std::invalid_argument("fusion introduces a same-stage or backward dependency");
  auto stage_count=int(original.stages.size())-1;
  for (int stage=0;stage<stage_count;++stage) {
    auto tasks=result.tasks.IntersectRange("{ [s,t] : s="+std::to_string(stage)+" }");
    auto record=original.stages[stage<producer ? stage : stage+1];
    record.task_count=tasks.ImageCard();
    result.stages.push_back(record);
    result.runtime_task_refs=result.runtime_task_refs.Add(record.task_count);
    auto worker_zero=tasks.IntersectRange("{ [s,t] : t % "+std::to_string(options.grid)+" = 0 }");
    result.max_worker_task_refs=result.max_worker_task_refs.Add(worker_zero.ImageCard());
  }
  auto requested=parse("{ [s,t] -> [w,pstage,kind,g] : false }");
  auto waits=requested;
  auto ownership=[&](int old,int next) {
    return out.phase_tasks.IntersectDomain("{ [s,t] : s="+std::to_string(next)+" }")
        .IntersectRange("{ [s,t] : s="+std::to_string(old)+" }")
        .Reverse().ProjectRange(0,1).Reverse().ProjectRange(0,1);
  };
  auto task_count=[&](int stage) {
    auto count=result.tasks.BindParams(result.task_binding)
        .IntersectRange("{ [s,t] : s="+std::to_string(stage)+" }").ImageCard().Eval({});
    if(count<=0 || std::uint64_t(count)>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("bound fused task count overflows");
    return std::uint32_t(count);
  };
  for(auto const& edge:original.runtime_tables) {
    if(edge.producer==producer && edge.consumer==consumer)continue;
    auto new_stage=[&](int old){return old<=producer?old:old-1;};
    int np=new_stage(edge.producer),nc=new_stage(edge.consumer);
    auto exact=ownership(edge.consumer,nc).ApplyRange(edge.table.linear_relation)
        .ApplyRange(ownership(edge.producer,np).Reverse()).BindParams(result.task_binding);
    result.runtime_tables.push_back({np,nc,analysis::BuildDependencyTableLinear(
        exact,task_count(np),task_count(nc))});
  }
  std::set<std::pair<int,int>> counted_pairs;
  std::uint64_t counter_offset=0;
  for(auto const& edge:original.runtime_counted) {
    if(options.force_all_dependencies)continue;
    if(edge.producer==producer && edge.consumer==consumer)continue;
    auto new_stage=[&](int old){return old<=producer?old:old-1;};
    int np=new_stage(edge.producer),nc=new_stage(edge.consumer);
    auto cm=ownership(edge.consumer,nc),pm=ownership(edge.producer,np);
    // Replicating a counted writer would publish a logical unit twice.
    // Splitting a target requires a new dispatch contract, not copied counts.
    if(!cm.IsSingleValued() || !cm.Reverse().IsSingleValued() ||
        !pm.IsSingleValued() || !pm.Reverse().IsSingleValued())
      throw std::invalid_argument("counted fusion requires bijective phase ownership");
    auto contract=edge.contract;
    contract.contributions.target_units=cm.ApplyRange(contract.contributions.target_units);
    auto targets=contract.contributions.target_units.Reverse().Image().ImageIdentity();
    long count=targets.ImageCard().Eval({});
    if(count<=0 || std::uint64_t(count)>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("counted fused target count overflows");
    auto domain=parse("{ [c] -> [c] : 0<=c<"+std::to_string(count)+" }");
    if(!equal(domain,targets))throw std::invalid_argument("counted fused targets are not dense");
    auto names=contract.contributions.target_units.DomainDimNames();
    std::vector<analysis::ParamBinding> coordinates(count);
    for(long i=0;i<count;++i)coordinates[i].Bind(names.at(0),i);
    auto values=contract.contributions.target_units.BoundTaskCard().EvalPoints({},coordinates);
    contract.contributions.expected.clear();
    for(auto value:values) {
      if(value<=0 || std::uint64_t(value)>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("counted fused threshold overflows");
      contract.contributions.expected.push_back(value);
    }
    auto old_sources=parse("{ [p] -> [p] : 0<=p<"+std::to_string(contract.producers)+" }");
    if(!equal(pm.Image().ImageIdentity(),old_sources))
      throw std::invalid_argument("counted fusion omits a logical writer");
    contract.producers=pm.Reverse().ImageCard().Eval({});
    contract.conservative_relation=cm.ApplyRange(contract.conservative_relation).ApplyRange(pm.Reverse());
    (void)analysis::BuildDependencyTableLinear(contract.conservative_relation,contract.producers,count);
    result.runtime_counted.push_back({np,nc,std::move(contract)});
    counted_pairs.emplace(edge.producer,edge.consumer);
    auto owned=parse("{ [s="+std::to_string(nc)+",t] -> [w,pstage="+
        std::to_string(np)+",kind=3,g] : 0<=t<"+std::to_string(count)+
        " and w=t%"+std::to_string(options.grid)+" and g="+std::to_string(counter_offset)+"+t }");
    requested=requested.Union(owned);waits=waits.Union(owned);
    counter_offset+=count;
    if(counter_offset>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("counted fused counter storage overflows");
  }
  // Aggregate versus fine is an edge policy, not inferred from the size of
  // a newly composed dependency window. Preserve it through phase mapping.
  for (int old_p=0;old_p<int(original.stages.size());++old_p)
    for (int old_c=old_p+1;old_c<int(original.stages.size());++old_c) {
      if (old_p==producer && old_c==consumer) continue;
      if(counted_pairs.count({old_p,old_c})) {
        auto ordinary=original.requested_events.IntersectDomain("{ [s,t] : s="+std::to_string(old_c)+" }")
            .IntersectRange("{ [w,pstage,kind,g] : pstage="+std::to_string(old_p)+" and kind!=3 }");
        if(ordinary.IsSubset(parse("{ [s,t] -> [w,pstage,kind,g] : false }")))continue;
      }
      auto edge=ordinary.IntersectDomain("{ [s,t] : s="+std::to_string(old_c)+" }")
          .IntersectRange("{ [s,t] : s="+std::to_string(old_p)+" }");
      if (edge.IsSubset(parse("{ [s,t] -> [a,b] : false }"))) continue;
      auto projected=out.phase_tasks.ApplyRange(edge).ApplyRange(out.phase_tasks.Reverse());
      auto policy=original.requested_events.IntersectDomain("{ [s,t] : s="+std::to_string(old_c)+" }")
          .IntersectRange("{ [w,pstage,kind,g] : pstage="+std::to_string(old_p)+" and kind=0 }");
      bool aggregate=!policy.IsSubset(parse("{ [s,t] -> [w,pstage,kind,g] : false }"));
      if (!aggregate && options.kappa==0)
        throw std::invalid_argument("aggregate runtime projection lacks event policy");
      auto groups=projected.ApplyRange(parse(aggregate
          ? "{ [pstage,p] -> [pstage,kind=0,g=0] }"
          : "{ [pstage,p] -> [pstage,kind="+std::to_string(options.kappa==1 ? 2 : 1)+
            ",g] : g=floord(p,"+std::to_string(options.kappa)+") }"));
      auto owned=groups.RangeProduct(parse("{ [s,t] -> [w] : w=t%"+
          std::to_string(options.grid)+" }"))
          .ApplyRange(parse("{ [pstage,kind,g,w] -> [w,pstage,kind,g] }"));
      requested=requested.Union(owned);
      if (options.kappa==1 && !aggregate)
        owned=owned.IntersectRange("{ [w,pstage,kind,g] : g % "+std::to_string(options.grid)+" != w }");
      waits=waits.Union(owned);
    }
  result.requested_events=std::move(requested);
  result.waits=std::move(waits);
  result.runtime_wait_entries=result.waits.ImageCard();
  return out;
}

std::vector<codegen::DependencyRecord> RebuildBoundRuntimeDependencies(
    RuntimeProjection const& projection) {
  analysis::IslReferenceAudit audit(__func__);
  using R=analysis::CouplingRelation;
  auto parse=[](std::string const& text){return R::FromIslText(text);};
  auto tasks=projection.tasks.BindParams(projection.task_binding);
  auto ordinary=projection.ordinary_dependencies.value_or(projection.dependencies)
      .BindParams(projection.task_binding);
  auto count=[&](int stage) {
    if(stage<0 || stage>=int(projection.stages.size()))
      throw std::invalid_argument("bound dependency references unknown stage");
    auto owned=tasks.IntersectRange("{ [s,t] : s="+std::to_string(stage)+" }")
        .ApplyRange(parse("{ [s,t] -> [t] }"));
    long n=owned.ImageCard().Eval({});
    if(n<=0 || std::uint64_t(n)>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("bound dependency task count overflows");
    auto dense=parse("{ [] -> [t] : 0<=t<"+std::to_string(n)+" }");
    if(!dense.IsSubset(owned.Image()) || !owned.Image().IsSubset(dense))
      throw std::invalid_argument("bound dependency task ids are not dense");
    return std::uint32_t(n);
  };
  std::vector<codegen::DependencyRecord> result;
  auto pairs=ordinary.ProjectRange(1,1).Reverse().ProjectRange(1,1);
  for(auto const& [p,c]:pairs.Points()) {
    if(p.at(0)>=c.at(0))throw std::invalid_argument("bound dependency is not forward");
    auto exact=ordinary.IntersectDomain("{ [s,t] : s="+std::to_string(c[0])+" }")
        .IntersectRange("{ [s,t] : s="+std::to_string(p[0])+" }")
        .ApplyRange(parse("{ [s,t] -> [t] }"))
        .Reverse().ApplyRange(parse("{ [s,t] -> [t] }")).Reverse();
    codegen::DependencyRecord edge{std::uint32_t(p[0]),std::uint32_t(c[0]),{}};
    edge.table=analysis::BuildDependencyTableLinear(exact,count(p[0]),count(c[0]));
    result.push_back(std::move(edge));
  }
  for(auto const& source:projection.runtime_counted) {
    if(source.producer>=source.consumer ||
        count(source.producer)!=source.contract.producers ||
        count(source.consumer)!=source.contract.contributions.expected.size())
      throw std::invalid_argument("bound counted contract differs from task ownership");
    codegen::DependencyRecord edge{std::uint32_t(source.producer),std::uint32_t(source.consumer),{}};
    edge.counted=source.contract;
    result.push_back(std::move(edge));
  }
  std::stable_sort(result.begin(),result.end(),[](auto const& a,auto const& b) {
    return std::tie(a.consumer,a.producer)<std::tie(b.consumer,b.producer);
  });
  return result;
}
}  // namespace tilemega::solver
