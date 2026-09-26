// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/HandoffAccess.h>
#include <tilemega/Analysis/ISLContext.h>
#include "IslUtil.h"
#include <stdexcept>
namespace tilemega::analysis {
namespace {
bool Equal(CouplingRelation const& a,CouplingRelation const& b) {
  return a.IsSubset(b) && b.IsSubset(a);
}
void Require(bool yes,char const* reason) {if(!yes)throw std::invalid_argument(reason);}
bool OneRow(CouplingRelation const& access) {
  auto map=isl_util::ReadMap(SharedIslContext().raw(),access.ToString());
  int rank=isl_map_dim(map.get(),isl_dim_out);
  if(rank<2)return true; // A vector operand has no row axis.
  map=isl_util::Map(isl_map_project_out(map.release(),isl_dim_out,rank-1,1));
  return isl_map_is_single_valued(map.get())==isl_bool_true;
}
HandoffAccessProof Compose(TaskAccesses const& p,TaskAccesses const& c,std::set<std::string> const& tensors) {
  Require(!tensors.empty(),"handoff needs an intermediate tensor");
  HandoffAccessProof proof;
  for(auto const& name:tensors) {
    Require(p.writes.count(name) && c.reads.count(name),"handoff tensor is not a read/write edge");
    auto const& read=c.reads.at(name);auto const& write=p.writes.at(name);
    auto edge=read.ApplyRange(write.Reverse());
    Require(read.IsSubset(edge.ApplyRange(write)),"handoff read has unsupplied elements");
    proof.consumer_to_producer=proof.consumer_to_producer.Union(edge);
  }
  auto const& edge=proof.consumer_to_producer;
  for(auto const& [name,write]:c.writes)
    Require(write.Reverse().Image().IsSubset(edge.Reverse().Image()),"handoff does not cover every consumer task");
  proof.composed.writes=c.writes;
  for(auto const& [name,read]:p.reads)proof.composed.reads[name]=edge.ApplyRange(read);
  for(auto const& [name,read]:c.reads)if(!tensors.count(name))
    proof.composed.reads[name]=proof.composed.reads[name].Union(read);
  return proof;
}
}
HandoffAccessProof ProveRecompute(TaskAccesses const& p,TaskAccesses const& c,
    std::set<std::string> const& tensors,TaskAccesses const& available) {
  IslReferenceAudit audit(__func__);auto proof=Compose(p,c,tensors);
  for(auto const& [name,write]:p.writes) {
    Require(tensors.count(name),"recompute producer has an unhandled side write");
    Require(OneRow(write),"recompute producer writes multiple rows per task");
  }
  for(auto const& [name,read]:p.reads) {
    Require(OneRow(read),"recompute reduction spans multiple rows");
    Require(available.reads.count(name) &&
        proof.consumer_to_producer.ApplyRange(read).IsSubset(available.reads.at(name)),
        "recompute input is not available before consumer execution");
  }
  return proof;
}
HandoffAccessProof ProveLastArriver(TaskAccesses const& p,TaskAccesses const& c,
    std::map<std::string,CouplingRelation> const& collapse,std::string const& output) {
  IslReferenceAudit audit(__func__);std::set<std::string> tensors;
  for(auto const& [name,map]:collapse)tensors.insert(name);
  auto proof=Compose(p,c,tensors);auto const& edge=proof.consumer_to_producer;
  Require(edge.Reverse().IsSingleValued(),"a partial tile belongs to multiple reduction consumers");
  Require(c.writes.count(output),"reduction output is missing");
  for(auto const& [name,map]:collapse) {
    auto const& read=c.reads.at(name);auto const& write=p.writes.at(name);
    Require(Equal(read,edge.ApplyRange(write)),"last arriver must read complete producer tiles");
    Require(Equal(read.ApplyRange(map),c.writes.at(output)),"partial projection does not cover the output tile");
    auto fibre=c.writes.at(output).ApplyRange(map.Reverse()).ApplyRange(write.Reverse());
    Require(Equal(edge,fibre),"last arriver omits members of the reduction fibre");
  }
  return proof;
}
HandoffAccessProof ProveDirectHandoff(TaskAccesses const& p,TaskAccesses const& c,
    std::set<std::string> const& tensors,CouplingRelation const& pp,CouplingRelation const& cp) {
  IslReferenceAudit audit(__func__);auto proof=Compose(p,c,tensors);auto const& edge=proof.consumer_to_producer;
  Require(edge.IsSingleValued() && edge.Reverse().IsSingleValued(),"direct handoff requires Unique in both directions");
  Require(pp.IsSingleValued() && cp.IsSingleValued(),"placement must be a function");
  auto next=CouplingRelation::FromIslText("{ [worker,slot] -> [worker,next] : next=slot+1 }");
  Require(Equal(edge.ApplyRange(pp).ApplyRange(next),cp),"direct handoff tasks are not adjacent on the same worker");
  return proof;
}
} // namespace tilemega::analysis
