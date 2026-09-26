// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/ExecOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <stdexcept>
namespace tilemega::dialect {
namespace {
using namespace mlir;
using analysis::CouplingRelation;
template<class T>T Symbol(Operation* scope,llvm::StringRef name) {
  return dyn_cast_or_null<T>(SymbolTable::lookupSymbolIn(scope,name));
}
bool Equal(CouplingRelation const& a,CouplingRelation const& b) {return a.IsSubset(b) && b.IsSubset(a);}
DictionaryAttr Maps(OpBuilder& b,std::map<std::string,CouplingRelation> const& maps) {
  NamedAttrList out;for(auto const& [name,map]:maps)out.set(name,CouplingMapAttr::get(b.getContext(),map));
  return out.getDictionary(b.getContext());
}
void RewriteRecompute(GraphOp graph,PlanOp plan,HandoffOp decision,analysis::HandoffAccessProof const& proof) {
  auto edge=Symbol<CouplingOp>(graph,decision.getCoupling());
  if(!edge)throw std::invalid_argument("overlapping handoffs require a new solve after the first rewrite");
  auto p=Symbol<TileSpaceOp>(graph,edge.getSrc()),c=Symbol<TileSpaceOp>(graph,edge.getDst());
  if(!p || !c)throw std::invalid_argument("handoff phase was already rewritten");
  auto pa=HandoffTaskAccesses(p),ca=HandoffTaskAccesses(c);
  auto identity=ca.writes.begin()->second.Reverse().Image().ImageIdentity();
  auto name=p.getSymName().str()+"__"+c.getSymName().str();
  OpBuilder b(graph.getContext());b.setInsertionPoint(c);
  OperationState state(c.getLoc(),FusedTileSpaceOp::getOperationName());
  state.addAttribute("sym_name",b.getStringAttr(name));
  state.addAttribute("phase_semantics",b.getArrayAttr({p.getSemanticAttr(),c.getSemanticAttr()}));
  state.addAttribute("phase_granularities",b.getArrayAttr({p.getGranularity(),c.getGranularity()}));
  state.addAttribute("phase_stages",b.getDenseI64ArrayAttr({int64_t(p.getStage()),int64_t(c.getStage())}));
  state.addAttribute("phase_maps",b.getArrayAttr({CouplingMapAttr::get(b.getContext(),proof.consumer_to_producer),CouplingMapAttr::get(b.getContext(),identity)}));
  state.addAttribute("reads",Maps(b,proof.composed.reads));state.addAttribute("writes",Maps(b,proof.composed.writes));
  state.addAttribute("task_count",MetricAttr::get(b.getContext(),identity.ImageCard()));
  state.addAttribute("handoff_kind",b.getStringAttr("recompute"));
  state.addAttribute("handoff_plan",FlatSymbolRefAttr::get(b.getContext(),plan.getSymName()));
  state.addAttribute("handoff_coupling",b.getStringAttr(decision.getCoupling()));
  b.create(state);
  std::vector<CouplingOp> edges;for(auto e:graph.getBody().front().getOps<CouplingOp>())edges.push_back(e);
  bool retain=false;for(auto e:edges)if(e.getSrc()==p.getSymName() && e.getDst()!=c.getSymName())retain=true;
  std::set<std::string> old_events;
  auto rewrite=[&](CouplingOp e,CouplingRelation const& relation,llvm::StringRef src,llvm::StringRef dst) {
    old_events.insert(e.getEvent().str());
    e.setSrcAttr(FlatSymbolRefAttr::get(b.getContext(),src));e.setDstAttr(FlatSymbolRefAttr::get(b.getContext(),dst));
    e.setRelationAttr(CouplingMapAttr::get(b.getContext(),relation));
    e.setWaitAttr(MetricAttr::get(b.getContext(),relation.Card()));
    e.setFanoutAttr(MetricAttr::get(b.getContext(),relation.FanoutCard()));
    e.setCountAttr(MetricAttr::get(b.getContext(),relation.Reverse().ImageCard()));
    e->removeAttr("wait_map");e->removeAttr("coupling_attrs");
    e.setSyncKindAttr(SyncKindAttr::get(b.getContext(),b.getStringAttr("global")));
    auto event_name=e.getSymName().str()+"__handoff_event";
    OperationState event(e.getLoc(),EventTensorOp::getOperationName());
    auto extent=MetricAttr::get(b.getContext(),relation.ImageCard());
    event.addAttribute("sym_name",b.getStringAttr(event_name));
    event.addAttribute("event_type",TypeAttr::get(RankedTensorType::get({ShapedType::kDynamic},b.getI32Type())));
    event.addAttribute("extent",extent);event.addAttribute("dims",b.getArrayAttr({extent}));
    b.setInsertionPoint(e);b.create(event);e.setEventAttr(FlatSymbolRefAttr::get(b.getContext(),event_name));
  };
  for(auto e:edges) {
    if(e.getSrc()==p.getSymName() && e.getDst()==c.getSymName()) {old_events.insert(e.getEvent().str());e.erase();continue;}
    if(e.getDst()==p.getSymName()) {
      auto relation=proof.consumer_to_producer.ApplyRange(e.getRelation().getMap());
      auto clone=cast<CouplingOp>(e->clone());clone.setSymNameAttr(b.getStringAttr(e.getSymName().str()+"__handoff"));
      clone.setVolumeAttr(MetricAttr::get(b.getContext(),e.getVolume().getValue().SumAlong(proof.consumer_to_producer)));
      graph.getBody().front().push_back(clone);rewrite(clone,relation,e.getSrc(),name);
      if(!retain){old_events.insert(e.getEvent().str());e.erase();}
    }else if(e.getSrc()==c.getSymName() || e.getDst()==c.getSymName()) {
      auto src=e.getSrc()==c.getSymName()?llvm::StringRef(name):e.getSrc();
      auto dst=e.getDst()==c.getSymName()?llvm::StringRef(name):e.getDst();
      rewrite(e,e.getRelation().getMap(),src,dst);
    }
  }
  if(!retain)p.erase();c.erase();
  for(auto const& event:old_events) {
    bool used=false;for(auto e:graph.getBody().front().getOps<CouplingOp>())used|=e.getEvent()==event;
    if(!used)if(auto op=Symbol<EventTensorOp>(graph,event))op.erase();
  }
}
struct ApplyPass:PassWrapper<ApplyPass,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyPass)
  llvm::StringRef getArgument()const final{return "tilemega-apply-handoffs";}
  llvm::StringRef getDescription()const final{return "Rewrite access-proved edge handoffs into CG task phases";}
  void runOnOperation()override{try{ApplyHandoffs(getOperation());}catch(std::exception const& e){getOperation().emitError(e.what());signalPassFailure();}}
};
}
void VerifyWrittenHandoff(FusedTileSpaceOp task) {
  auto module=task->getParentOfType<mlir::ModuleOp>();
  auto plan_ref=task->getAttrOfType<mlir::FlatSymbolRefAttr>("handoff_plan");
  auto edge_name=task->getAttrOfType<mlir::StringAttr>("handoff_coupling");
  auto plan=plan_ref?Symbol<PlanOp>(module,plan_ref.getValue()):PlanOp{};
  if(!plan || !edge_name)throw std::invalid_argument("written handoff lacks its source decision");
  HandoffOp decision;
  for(auto h:plan.getBody().front().getOps<HandoffOp>())if(h.getCoupling()==edge_name.getValue())decision=h;
  if(!decision || decision.getKind()!=task->getAttrOfType<mlir::StringAttr>("handoff_kind").getValue())
    throw std::invalid_argument("written handoff differs from the solver decision");
  auto proof=VerifyHandoffAccess(decision);
  auto graph=Symbol<GraphOp>(module,plan.getGraph());auto edge=Symbol<CouplingOp>(graph,decision.getCoupling());
  auto p=Symbol<TileSpaceOp>(graph,edge.getSrc()),c=Symbol<TileSpaceOp>(graph,edge.getDst());
  if(task.getPhaseSemantics()!=mlir::ArrayAttr::get(module.getContext(),{p.getSemanticAttr(),c.getSemanticAttr()}) ||
     task.getPhaseGranularities()!=mlir::ArrayAttr::get(module.getContext(),{p.getGranularity(),c.getGranularity()}))
    throw std::invalid_argument("handoff phases differ from the source access proof");
  if(!Equal(mlir::cast<CouplingMapAttr>(task.getPhaseMaps()[0]).getMap(),proof.consumer_to_producer))
    throw std::invalid_argument("handoff phase map differs from the exact coupling");
  auto ca=HandoffTaskAccesses(c);
  auto identity=ca.writes.begin()->second.Reverse().Image().ImageIdentity();
  if(!Equal(mlir::cast<CouplingMapAttr>(task.getPhaseMaps()[1]).getMap(),identity))
    throw std::invalid_argument("handoff consumer phase is not the identity");
  for(auto const& pair:{std::make_pair(task.getReads(),&proof.composed.reads),std::make_pair(task.getWrites(),&proof.composed.writes)}) {
    if(pair.first.size()!=pair.second->size())throw std::invalid_argument("written handoff access set differs");
    for(auto const& [name,map]:*pair.second) {
      auto attr=pair.first.getAs<CouplingMapAttr>(name);
      if(!attr || !Equal(attr.getMap(),map))throw std::invalid_argument("written handoff access relation differs");
    }
  }
}
void ApplyHandoffs(mlir::ModuleOp module) {
  analysis::IslReferenceAudit audit(__func__);
  if(mlir::failed(mlir::verify(module)))throw std::invalid_argument("handoff pass requires verified IR");
  mlir::OwningOpRef<mlir::ModuleOp> copy(mlir::cast<mlir::ModuleOp>(module->clone()));
  for(auto plan:copy->getOps<PlanOp>()) {
    auto source=Symbol<GraphOp>(*copy,plan.getGraph());
    if(!source)throw std::invalid_argument("plan graph missing");
    std::vector<std::pair<HandoffOp,analysis::HandoffAccessProof>> selected;
    for(auto h:plan.getBody().front().getOps<HandoffOp>())if(h.getKind()!="event")selected.emplace_back(h,VerifyHandoffAccess(h));
    if(selected.empty())continue;
    auto name=source.getSymName().str()+"__handoff";
    if(SymbolTable::lookupSymbolIn(*copy,name))throw std::invalid_argument("handoff graph already exists");
    auto rewritten=cast<GraphOp>(source->clone());rewritten.setSymNameAttr(StringAttr::get(module.getContext(),name));
    copy->getBody()->push_back(rewritten);
    for(auto& [h,proof]:selected) {
      if(h.getKind()=="recompute")RewriteRecompute(rewritten,plan,h,proof);
      else throw std::invalid_argument("selected handoff has no graph rewrite yet");
    }
    (*copy)->setAttr("tmexec.active_graph",FlatSymbolRefAttr::get(module.getContext(),name));
    (*copy)->setAttr("tilemega.handoff_pending_lowering",UnitAttr::get(module.getContext()));
  }
  if(mlir::failed(mlir::verify(*copy)))throw std::invalid_argument("handoff result failed CG verification");
  module->setAttrs((*copy)->getAttrs());module.getBodyRegion().takeBody(copy->getBodyRegion());
}
void RegisterHandoffPass(){mlir::PassRegistration<ApplyPass>();}
}
