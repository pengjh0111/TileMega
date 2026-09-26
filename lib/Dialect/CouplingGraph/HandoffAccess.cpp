// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/ExecOps.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/SymbolTable.h>
#include <stdexcept>
#include <algorithm>
namespace tilemega::dialect {
namespace {
using analysis::CouplingRelation;
using analysis::TaskAccesses;
TaskAccesses Access(TileSpaceOp space,bool elements=false) {
  if(!space.getSemantic())throw std::invalid_argument("handoff requires L-sem on both tasks");
  auto op=analysis::DecodeSemanticOp(space.getSemantic()->str());
  analysis::Granularity g;
  for(auto const& axis:op.result.axes) {
    auto value=space.getGranularity().getAs<mlir::StringAttr>(axis.name);
    if(!value)throw std::invalid_argument("handoff lacks a task granularity");
    g.Tile(op.name,axis.name,elements?analysis::ClosedForm::Constant(1):
        analysis::ClosedForm::Parse(value.getValue().str()));
  }
  analysis::SemanticGraph sem;sem.ops.push_back(op);
  auto graph=analysis::Instantiate(sem,g);auto const& task=*graph.Find(op.name);
  TaskAccesses out;
  out.writes[op.result.name]=analysis::ElementAccess(task,analysis::BuildWriteMap(task),{},analysis::AccessDomain::kPhysicalTensor);
  for(auto const& w:op.additional_writes)
    out.writes[w.tensor.name]=out.writes[w.tensor.name].Union(analysis::ExactElementRead(op,task,{w.tensor,w.map,w.nonnegative},{}));
  if(!op.element_reads.empty()) {
    for(auto const& r:op.element_reads)out.reads[r.tensor.name]=out.reads[r.tensor.name].Union(analysis::ExactElementRead(op,task,r,{}));
  }else for(std::size_t i=0;i<task.operands.size();++i) {
    auto r=analysis::BuildReadMap(task,i);
    out.reads[r.tensor.name]=out.reads[r.tensor.name].Union(analysis::ElementAccess(task,r,{},analysis::AccessDomain::kPhysicalTensor));
  }
  return out;
}
}
analysis::TaskAccesses HandoffTaskAccesses(TileSpaceOp task) {return Access(task);}
analysis::HandoffAccessProof VerifyHandoffAccess(HandoffOp handoff) {
  analysis::IslReferenceAudit audit(__func__);
  auto plan=handoff->getParentOfType<PlanOp>();
  if(!plan)throw std::invalid_argument("handoff must be nested in tmexec.plan");
  auto graph=mlir::SymbolTable::lookupNearestSymbolFrom<GraphOp>(plan->getParentOp(),plan.getGraphAttr());
  if(!graph)throw std::invalid_argument("handoff plan references an unknown graph");
  auto edge=mlir::dyn_cast_or_null<CouplingOp>(mlir::SymbolTable::lookupSymbolIn(graph,handoff.getCoupling()));
  if(!edge)throw std::invalid_argument("handoff references an unknown coupling");
  auto p=mlir::dyn_cast_or_null<TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getSrc()));
  auto c=mlir::dyn_cast_or_null<TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getDst()));
  if(!p || !c)throw std::invalid_argument("handoff needs unfused source and destination tasks");
  auto kind=handoff.getKind();
  if(kind=="event")return {};
  std::map<std::string,TaskAccesses> accesses;
  for(auto s:graph.getBody().front().getOps<TileSpaceOp>())accesses.emplace(s.getSymName().str(),Access(s));
  auto const& pa=accesses.at(p.getSymName().str());auto const& ca=accesses.at(c.getSymName().str());
  std::set<std::string> middle;CouplingRelation relation;
  for(auto const& [name,write]:pa.writes)if(auto r=ca.reads.find(name);r!=ca.reads.end()) {
    middle.insert(name);relation=relation.Union(r->second.ApplyRange(write.Reverse()));
  }
  if(middle.empty())throw std::invalid_argument("handoff has no exact shared tensor access");
  if(kind=="recompute") {
    TaskAccesses available;
    for(auto const& [name,read]:pa.reads) {
      auto required=relation.ApplyRange(read);CouplingRelation all_writes;
      for(auto const& [task,a]:accesses)if(auto w=a.writes.find(name);w!=a.writes.end())all_writes=all_writes.Union(w->second.Image());
      // Turn the no-producer image back into consumer-indexed accesses.
      auto external=required.Image().Subtract(all_writes);
      available.reads[name]=required.ApplyRange(external.ImageIdentity());
      // A dependency path proves availability; paths through the removed
      // producer become direct incoming edges in ApplyHandoffs.
      std::map<std::string,CouplingRelation> ancestors;
      auto identity=ca.writes.begin()->second.Reverse().Image().ImageIdentity();
      ancestors[c.getSymName().str()]=identity;
      std::vector<TileSpaceOp> ordered;
      for(auto task:graph.getBody().front().getOps<TileSpaceOp>())ordered.push_back(task);
      std::sort(ordered.begin(),ordered.end(),[](auto x,auto y){return x.getStage()>y.getStage();});
      for(auto dst:ordered) {
        auto reached=ancestors.find(dst.getSymName().str());
        if(reached==ancestors.end())continue;
        for(auto incoming:graph.getBody().front().getOps<CouplingOp>()) {
          if(incoming.getDst()!=dst.getSymName())continue;
          auto src=accesses.find(incoming.getSrc().str());
          if(src==accesses.end())continue;
          auto mapping=reached->second.ApplyRange(incoming.getRelation().getMap());
          ancestors[src->first]=ancestors[src->first].Union(mapping);
          if(incoming.getSrc()!=p.getSymName())
            if(auto w=src->second.writes.find(name);w!=src->second.writes.end())
              available.reads[name]=available.reads[name].Union(mapping.ApplyRange(w->second));
        }
      }
    }
    return analysis::ProveRecompute(pa,ca,middle,available);
  }
  if(kind=="last_arriver") {
    auto elements=Access(c,true);auto op=analysis::DecodeSemanticOp(c.getSemantic()->str());
    std::map<std::string,CouplingRelation> collapse;
    for(auto const& name:middle)collapse[name]=elements.reads.at(name).Reverse().ApplyRange(elements.writes.at(op.result.name));
    return analysis::ProveLastArriver(pa,ca,collapse,op.result.name);
  }
  if(kind=="smem_direct") {
    // These relations are the plan's authoritative pi/sigma, not annotations
    // supplied by the handoff decision. Lowering must consume the same maps.
    CouplingRelation pp,cp;
    for(auto place:plan.getBody().front().getOps<PlacementOp>()) {
      auto position=place->getAttrOfType<CouplingMapAttr>("position");
      if(!position)continue;
      if(place.getTask()==p.getSymName())pp=position.getMap();
      if(place.getTask()==c.getSymName())cp=position.getMap();
    }
    if(pp.empty() || cp.empty())throw std::invalid_argument("direct handoff requires actual placement position relations");
    return analysis::ProveDirectHandoff(pa,ca,middle,pp,cp);
  }
  throw std::invalid_argument("unknown edge handoff kind");
}
mlir::LogicalResult HandoffOp::verify() {
  try {VerifyHandoffAccess(*this);return mlir::success();}
  catch(std::exception const& e){return emitOpError(e.what());}
}
}
