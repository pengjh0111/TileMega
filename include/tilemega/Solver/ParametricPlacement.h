// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <string>
#include <ostream>

namespace tilemega::solver {
struct ParametricPlacement {
  analysis::CouplingRelation tasks;       // [] -> [stage,task]
  analysis::CouplingRelation dependencies;// consumer -> producer, including grouped waits
  analysis::CouplingRelation pi_sigma;    // [stage,task] -> [worker,slot]
  analysis::CouplingRelation rank;        // [stage,task] -> [stage-order,task-order]
  analysis::CouplingRelation grid_limit;  // [] -> [grid,resident_limit]
  std::string family;
};
struct ParametricProof {
  bool total=false,bijective=false,dense=false,acyclic=false,resident=false;
  analysis::CouplingRelation queue_edges,ordered_edges;
  std::string error;
  bool passed() const { return total&&bijective&&dense&&acyclic&&resident; }
};
/// Exact universal checks over the parameter domain carried by the maps.
/// A strict finite lexicographic rank proves acyclicity of dependency and
/// FIFO edges together; no bounded set of sampled theta values is a proof.
inline ParametricProof ProveParametricPlacement(ParametricPlacement const& p,std::ostream* diagnostic=nullptr,int finite_grid=0) {
  using analysis::CouplingRelation;
  auto map=[](char const* s){return CouplingRelation::FromIslText(s);};
  ParametricProof proof;
  auto note=[&](char const* step){if(diagnostic)*diagnostic<<"ISL_CHECK "<<step<<std::endl;};
  try {
    note("total");
    auto domain=p.pi_sigma.Reverse().Image();
    proof.total=p.pi_sigma.IsSingleValued() && domain.IsSubset(p.tasks) && p.tasks.IsSubset(domain);
    note("bijective");
    proof.bijective=p.pi_sigma.Reverse().IsSingleValued();
    auto occupied=p.pi_sigma.Image();
    auto preceding=occupied.ApplyRange(map("{ [w,s] -> [w,s-1] : s>0 }"));
    note("dense");
    proof.dense=occupied.IsSubset(map("{ [] -> [w,s] : w>=0 and s>=0 }"));
    if(finite_grid>0) {
      auto limits=p.grid_limit.Points();
      if(limits.size()!=1 || limits[0].second.size()!=2 || limits[0].second[0]!=finite_grid)
        throw std::invalid_argument("finite dense certificate requires a completely bound grid");
      // Enumerate the entire bound ISL image, not sampled parameter values.
      // Canonical slot runs avoid repeated Presburger elimination across all
      // stage pieces sharing the same non-power-of-two worker modulus.
      std::vector<std::vector<long>> worker_slots(finite_grid);
      for(auto const& point:occupied.Points()) {
        auto const& where=point.second;
        if(where.size()!=2 || where[0]<0 || where[0]>=finite_grid || where[1]<0)
          throw std::invalid_argument("bound occupied image outside resident workers");
        worker_slots[where[0]].push_back(where[1]);
      }
      std::string runs="{ ";bool first=true;
      for(int w=0;w<finite_grid;++w) {
        auto& slots=worker_slots[w];std::sort(slots.begin(),slots.end());
        slots.erase(std::unique(slots.begin(),slots.end()),slots.end());
        for(std::size_t i=0;i<slots.size();) {
          auto lo=slots[i],hi=lo;while(++i<slots.size() && slots[i]==hi+1)hi=slots[i];
          if(!first)runs+="; ";first=false;
          runs+="[] -> ["+std::to_string(w)+",s] : "+std::to_string(lo)+"<=s<="+std::to_string(hi);
        }
      }
      runs+=first ? "[] -> [w,s] : false }" : " }";
      auto canonical=CouplingRelation::FromIslText(runs);
      proof.dense=proof.dense && canonical.ApplyRange(map("{ [w,s] -> [w,s-1] : s>0 }")).IsSubset(canonical);
    } else proof.dense=proof.dense && preceding.IsSubset(occupied);
    note("resident");
    auto limits=p.grid_limit.ApplyRange(map("{ [g,r] -> [g,r] : 0<g<=r }"));
    proof.resident=p.grid_limit.IsSingleValued() && p.grid_limit.IsSubset(limits) &&
        p.tasks.Reverse().Image().IsSubset(p.grid_limit.Reverse().Image());
    auto worker_count=p.pi_sigma.Image().RangeProduct(p.grid_limit);
    proof.resident=proof.resident && worker_count.IsSubset(
        map("{ [] -> [w,s,g,r] : 0<=w<g and g<=r }"));
    note("queue_edges");
    proof.queue_edges=p.pi_sigma.ApplyRange(map("{ [w,s] -> [w,s+1] }"))
        .ApplyRange(p.pi_sigma.Reverse());
    // Map only existing edges to ranks. Constructing every ordered pair of
    // tasks first forms a quadratic relation unrelated to the input graph.
    note("ranked_edges");
    auto ranked_edges=p.rank.Reverse().ApplyRange(
        p.dependencies.Reverse().Union(proof.queue_edges)).ApplyRange(p.rank);
    auto lex=map("{ [a,b] -> [c,d] : a<c or (a=c and b<d) }");
    proof.ordered_edges=p.dependencies.Reverse().Union(proof.queue_edges);
    auto rank_domain=p.rank.Reverse().Image();
    note("acyclic");
    proof.acyclic=p.rank.IsSingleValued() && p.tasks.IsSubset(rank_domain) &&
        ranked_edges.IsSubset(lex);
    if (!proof.passed()) proof.error="unproved totality, dense bijection, resident limit, or union rank";
  } catch (std::exception const& e) {proof.error=e.what();}
  return proof;
}

inline MaterializedPlan EvaluateParametricPlacement(ParametricPlacement const& p,
    analysis::ParamBinding const& theta,std::vector<int> const& counts,int grid) {
  MaterializedPlan plan;plan.owner.resize(counts.size());plan.slot.resize(counts.size());
  for (std::size_t s=0;s<counts.size();++s) {
    plan.owner[s].assign(counts[s],-1);plan.slot[s].assign(counts[s],-1);
  }
  for (auto const& [task,where]:p.pi_sigma.BindParams(theta).Points()) {
    if (task.size()!=2 || where.size()!=2 || task[0]<0 || task[0]>=long(counts.size()) ||
        task[1]<0 || task[1]>=counts[task[0]] || where[0]<0 || where[0]>=grid || where[1]<0)
      throw std::invalid_argument("symbolic Plan evaluation outside runtime domain");
    auto& owner=plan.owner[task[0]][task[1]];
    if (owner!=-1) throw std::invalid_argument("symbolic Plan evaluates more than once");
    owner=int(where[0]);plan.slot[task[0]][task[1]]=int(where[1]);
  }
  std::string error;
  if (!BuildPlanQueues(counts,grid,&plan,&error)) throw std::invalid_argument(error);
  return plan;
}
} // namespace tilemega::solver
