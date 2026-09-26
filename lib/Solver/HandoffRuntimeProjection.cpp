// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/HandoffRuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <stdexcept>
#include <set>
namespace tilemega::solver {
HandoffRuntimeProjection ProjectHandoffRuntime(
    analysis::CouplingRelation const& tasks,
    analysis::CouplingRelation const& dependencies,int stage_count,
    std::vector<RuntimeHandoff> const& handoffs) {
  analysis::IslReferenceAudit audit(__func__);
  using R=analysis::CouplingRelation;
  auto parse=[](std::string const& text){return R::FromIslText(text);};
  auto equal=[](R const& a,R const& b){return a.IsSubset(b)&&b.IsSubset(a);};
  auto live=tasks.ImageIdentity();
  auto stage=[&](int s){return live.IntersectDomain("{ [s,t] : s="+std::to_string(s)+" }");};
  auto coordinates=[&](int s){return stage(s).ApplyRange(parse("{ [s,t] -> [t] }"));};
  auto empty=parse("{ [s,t] -> [p,u] : false }");
  auto internal=empty,extra=empty;
  std::set<int> removed,owners;
  for(auto const& h:handoffs) {
    if(h.producer<0 || h.consumer<=h.producer || h.consumer>=stage_count)
      throw std::invalid_argument("handoff runtime stages must be producer-before-consumer");
    if(h.kind=="event")continue;
    if(h.kind!="recompute" && h.kind!="last_arriver" && h.kind!="smem_direct")
      throw std::invalid_argument("unsupported runtime handoff kind");
    auto mapping=coordinates(h.consumer).ApplyRange(h.consumer_to_producer)
        .ApplyRange(coordinates(h.producer).Reverse());
    auto required=dependencies.IntersectDomain("{ [s,t] : s="+std::to_string(h.consumer)+" }")
        .IntersectRange("{ [s,t] : s="+std::to_string(h.producer)+" }");
    // Windows may include conservative extra producers, but no true phase
    // predecessor may be absent from the original execution dependency.
    if(!mapping.IsSubset(required) || mapping.IsSubset(empty))
      throw std::invalid_argument("handoff mapping escapes the source execution dependency");
    if(!equal(mapping.Image(),coordinates(h.producer).Reverse().Image())) {
      // A last-arriver consumer may omit empty producer blocks.
      if(h.kind!="last_arriver")
        throw std::invalid_argument("handoff mapping omits live producer tasks");
    }
    if(!equal(mapping.Reverse().Image(),coordinates(h.consumer).Reverse().Image()))
      throw std::invalid_argument("handoff mapping omits live consumer tasks");
    if(h.kind=="last_arriver" && !h.consumer_to_producer.Reverse().IsSingleValued())
      throw std::invalid_argument("a partial task cannot arrive at multiple output tickets");
    if(h.kind=="smem_direct" && (!h.consumer_to_producer.IsSingleValued() || !h.consumer_to_producer.Reverse().IsSingleValued()))
      throw std::invalid_argument("direct handoff requires bijective task ownership");
    int owner=h.kind=="last_arriver"?h.producer:h.consumer;
    int erased=h.kind=="last_arriver"?h.consumer:h.producer;
    if(removed.count(owner) || owners.count(erased))
      throw std::invalid_argument("overlapping runtime handoffs require composed phase maps");
    owners.insert(owner);
    if(h.kind=="last_arriver" || !h.retain_producer)removed.insert(erased);
    internal=internal.Union(required);
    extra=extra.Union(h.kind=="last_arriver"?mapping.Reverse():mapping);
  }
  HandoffRuntimeProjection result;
  auto renumber=empty;
  for(int old=0;old<stage_count;++old)if(!removed.count(old)) {
    int next=result.surviving_stages.size();result.surviving_stages.push_back(old);
    renumber=renumber.Union(stage(old).ApplyRange(parse("{ [s,t] -> [n="+std::to_string(next)+",t] }")));
  }
  result.tasks=tasks.ApplyRange(renumber);
  result.phases=renumber.Reverse().ApplyRange(live.Union(extra));
  auto external=dependencies.Subtract(internal);
  // If a removed phase feeds an unrewritten consumer, replication is valid
  // only when the composed phase map retains all required producers.
  auto referenced=external.Image().Union(external.Reverse().Image());
  if(!referenced.IsSubset(result.phases.Image()))
    throw std::invalid_argument("handoff dropped an externally referenced phase");
  result.dependencies=result.phases.ApplyRange(external).ApplyRange(result.phases.Reverse());
  if(!result.dependencies.IsSubset(parse("{ [s,t] -> [p,u] : p<s }")))
    throw std::invalid_argument("handoff introduced a self or backward execution dependency");
  return result;
}
}
