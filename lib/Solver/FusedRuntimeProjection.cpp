// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <stdexcept>

#ifndef TILEMEGA_FUSED_RUNTIME_PROJECTION
#define TILEMEGA_FUSED_RUNTIME_PROJECTION 1
#endif

namespace tilemega::solver {
FusedRuntimeProjection FuseProjectedQueues(RuntimeProjection const& original,
    int producer,int consumer,analysis::CouplingRelation const& coupling,
    RuntimeProjectionOptions options) {
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
      !equal(coupling.Reverse().Image(),c_tasks.Image()))
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
  result.tasks=original.tasks.ApplyRange(renumber);
  auto new_c_to_p=renumber.Reverse().ApplyRange(old_c_to_p);
  out.phase_tasks=renumber.Reverse().Union(new_c_to_p);
  result.dependencies=out.phase_tasks.ApplyRange(external).ApplyRange(out.phase_tasks.Reverse());
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
  // Aggregate versus fine is an edge policy, not inferred from the size of
  // a newly composed dependency window. Preserve it through phase mapping.
  for (int old_p=0;old_p<int(original.stages.size());++old_p)
    for (int old_c=old_p+1;old_c<int(original.stages.size());++old_c) {
      if (old_p==producer && old_c==consumer) continue;
      auto edge=external.IntersectDomain("{ [s,t] : s="+std::to_string(old_c)+" }")
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
}  // namespace tilemega::solver
