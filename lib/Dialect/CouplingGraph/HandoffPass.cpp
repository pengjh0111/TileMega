// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/ExecOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>
namespace tilemega::dialect {
namespace {
using namespace mlir;
using analysis::CouplingRelation;
template<class T>T Symbol(Operation* scope,llvm::StringRef name) {
  return dyn_cast_or_null<T>(SymbolTable::lookupSymbolIn(scope,name));
}
std::string UniqueSymbol(Operation* scope,std::string const& base) {
  if(!SymbolTable::lookupSymbolIn(scope,base))return base;
  for(unsigned suffix=1;;++suffix) {
    auto next=base+"_"+std::to_string(suffix);
    if(!SymbolTable::lookupSymbolIn(scope,next))return next;
  }
}
bool Equal(CouplingRelation const& a,CouplingRelation const& b) {return a.IsSubset(b) && b.IsSubset(a);}
CouplingRelation TaskIdentity(analysis::TaskAccesses const& accesses) {
  CouplingRelation domain;
  for(auto const& [name,write]:accesses.writes)domain=domain.Union(write.Reverse().Image());
  return domain.ImageIdentity();
}
DictionaryAttr Maps(OpBuilder& b,std::map<std::string,CouplingRelation> const& maps) {
  NamedAttrList out;for(auto const& [name,map]:maps)out.set(name,CouplingMapAttr::get(b.getContext(),map));
  return out.getDictionary(b.getContext());
}
void BindNormPrologue(GraphOp graph,TileSpaceOp producer,TileSpaceOp consumer) {
  auto module=graph->getParentOfType<ModuleOp>();
  auto model=module->getAttrOfType<DictionaryAttr>("tilemega.model_plan");
  if(!model)return; // Minimal access-proof fixtures have no runtime model.
  auto stages=dyn_cast_or_null<ArrayAttr>(model.get("stages"));
  auto gemms=dyn_cast_or_null<ArrayAttr>(model.get("gemms"));
  if(!stages || !gemms || producer.getStage()>=stages.size() ||
     consumer.getStage()>=stages.size())
    throw std::invalid_argument("recompute handoff lacks serving stage descriptors");
  auto p=cast<DictionaryAttr>(stages[producer.getStage()]);
  auto c=cast<DictionaryAttr>(stages[consumer.getStage()]);
  auto pk=cast<StringAttr>(p.get("kind")).getValue();
  auto ck=cast<StringAttr>(c.get("kind")).getValue();
  if(pk!="kRMSNorm" || ck!="kGemm")return;
  auto operands=cast<DenseI64ArrayAttr>(p.get("operands"));
  auto gemm=cast<IntegerAttr>(c.get("gemm")).getInt();
  if(gemm<0 || gemm>=int64_t(gemms.size()))
    throw std::invalid_argument("norm prologue GEMM index outside model");
  auto selected=cast<DictionaryAttr>(gemms[gemm]);
  if(operands.size()<3 || operands[0]<0 || operands[1]<0 ||
     operands[2]!=cast<IntegerAttr>(selected.get("a")).getInt())
    throw std::invalid_argument("norm prologue does not feed the GEMM A operand");
  OpBuilder b(graph.getContext());
  NamedAttrList updated(selected);
  updated.set("norm_input",b.getI64IntegerAttr(operands[0]));
  updated.set("norm_weight",b.getI64IntegerAttr(operands[1]));
  std::vector<Attribute> changed(gemms.begin(),gemms.end());
  changed[gemm]=updated.getDictionary(graph.getContext());
  NamedAttrList plan(model);plan.set("gemms",b.getArrayAttr(changed));
  module->setAttr("tilemega.model_plan",plan.getDictionary(graph.getContext()));
}
void RewriteForward(GraphOp graph,PlanOp plan,HandoffOp decision,analysis::HandoffAccessProof const& proof) {
  auto edge=Symbol<CouplingOp>(graph,decision.getCoupling());
  if(!edge)throw std::invalid_argument("overlapping handoffs require a new solve after the first rewrite");
  auto p=Symbol<TileSpaceOp>(graph,edge.getSrc()),c=Symbol<TileSpaceOp>(graph,edge.getDst());
  if(!p || !c)throw std::invalid_argument("handoff phase was already rewritten");
  if(decision.getKind()=="recompute")BindNormPrologue(graph,p,c);
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
  state.addAttribute("handoff_kind",b.getStringAttr(decision.getKind()));
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
    auto event_name=UniqueSymbol(graph,e.getSymName().str()+"__handoff_event");
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
      auto clone=cast<CouplingOp>(e->clone());clone.setSymNameAttr(b.getStringAttr(UniqueSymbol(graph,e.getSymName().str()+"__handoff")));
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
analysis::TaskAccesses LastArriverAccesses(analysis::TaskAccesses const& p,
    analysis::TaskAccesses const& c,CouplingRelation const& to_consumer) {
  auto out=p;
  for(auto const& [name,map]:c.reads)out.reads[name]=out.reads[name].Union(to_consumer.ApplyRange(map));
  for(auto const& [name,map]:c.writes)out.writes[name]=out.writes[name].Union(to_consumer.ApplyRange(map));
  return out;
}
void RewriteLastArriver(GraphOp graph,PlanOp plan,HandoffOp decision,analysis::HandoffAccessProof const& proof) {
  auto edge=Symbol<CouplingOp>(graph,decision.getCoupling());
  if(!edge)throw std::invalid_argument("last-arriver coupling was already rewritten");
  auto p=Symbol<TileSpaceOp>(graph,edge.getSrc()),c=Symbol<TileSpaceOp>(graph,edge.getDst());
  if(!p || !c)throw std::invalid_argument("last-arriver phases were already rewritten");
  auto pa=HandoffTaskAccesses(p),ca=HandoffTaskAccesses(c);
  auto to_consumer=proof.consumer_to_producer.Reverse();
  // The second phase is conditional. Partial stores remain global, and only
  // the last ticket holder executes the projected consumer phase.
  auto accesses=LastArriverAccesses(pa,ca,to_consumer);
  auto identity=TaskIdentity(pa);
  auto name=p.getSymName().str()+"__"+c.getSymName().str();
  OpBuilder b(graph.getContext());b.setInsertionPoint(p);
  auto ticket_name=name+"__tickets";
  OperationState ticket(p.getLoc(),EventTensorOp::getOperationName());
  auto extent=MetricAttr::get(b.getContext(),proof.consumer_to_producer.Reverse().ImageCard());
  ticket.addAttribute("sym_name",b.getStringAttr(ticket_name));
  ticket.addAttribute("event_type",TypeAttr::get(RankedTensorType::get({ShapedType::kDynamic},b.getI32Type())));
  ticket.addAttribute("extent",extent);ticket.addAttribute("dims",b.getArrayAttr({extent}));
  ticket.addAttribute("protocol",b.getStringAttr("last_arriver"));
  ticket.addAttribute("triggers",MetricAttr::get(b.getContext(),proof.consumer_to_producer.Card()));
  b.create(ticket);
  OperationState state(p.getLoc(),FusedTileSpaceOp::getOperationName());
  state.addAttribute("sym_name",b.getStringAttr(name));
  state.addAttribute("phase_semantics",b.getArrayAttr({p.getSemanticAttr(),c.getSemanticAttr()}));
  state.addAttribute("phase_granularities",b.getArrayAttr({p.getGranularity(),c.getGranularity()}));
  state.addAttribute("phase_stages",b.getDenseI64ArrayAttr({int64_t(p.getStage()),int64_t(c.getStage())}));
  state.addAttribute("phase_maps",b.getArrayAttr({CouplingMapAttr::get(b.getContext(),identity),CouplingMapAttr::get(b.getContext(),to_consumer)}));
  state.addAttribute("phase_conditions",b.getArrayAttr({b.getStringAttr("always"),b.getStringAttr("last_arriver")}));
  state.addAttribute("ticket",FlatSymbolRefAttr::get(b.getContext(),ticket_name));
  state.addAttribute("reads",Maps(b,accesses.reads));state.addAttribute("writes",Maps(b,accesses.writes));
  state.addAttribute("task_count",MetricAttr::get(b.getContext(),identity.ImageCard()));
  state.addAttribute("handoff_kind",b.getStringAttr("last_arriver"));
  state.addAttribute("handoff_plan",FlatSymbolRefAttr::get(b.getContext(),plan.getSymName()));
  state.addAttribute("handoff_coupling",b.getStringAttr(decision.getCoupling()));b.create(state);
  std::vector<CouplingOp> edges;for(auto e:graph.getBody().front().getOps<CouplingOp>())edges.push_back(e);
  std::set<std::string> old_events;
  for(auto e:edges) {
    bool srcp=e.getSrc()==p.getSymName(),srcc=e.getSrc()==c.getSymName();
    bool dstp=e.getDst()==p.getSymName(),dstc=e.getDst()==c.getSymName();
    if(!srcp && !srcc && !dstp && !dstc)continue;
    old_events.insert(e.getEvent().str());
    if(srcp && dstc){e.erase();continue;}
    auto relation=e.getRelation().getMap();
    if(dstc) {
      relation=to_consumer.ApplyRange(relation);
      e.setVolumeAttr(MetricAttr::get(b.getContext(),e.getVolume().getValue().SumAlong(to_consumer)));
    }
    if(srcc)relation=relation.ApplyRange(proof.consumer_to_producer);
    if(srcp || srcc)e.setSrcAttr(FlatSymbolRefAttr::get(b.getContext(),name));
    if(dstp || dstc)e.setDstAttr(FlatSymbolRefAttr::get(b.getContext(),name));
    e.setRelationAttr(CouplingMapAttr::get(b.getContext(),relation));
    e.setWaitAttr(MetricAttr::get(b.getContext(),relation.Card()));e.setFanoutAttr(MetricAttr::get(b.getContext(),relation.FanoutCard()));
    e.setCountAttr(MetricAttr::get(b.getContext(),relation.Reverse().ImageCard()));
    e->removeAttr("wait_map");e->removeAttr("coupling_attrs");
    e.setSyncKindAttr(SyncKindAttr::get(b.getContext(),b.getStringAttr("global")));
    auto event_name=UniqueSymbol(graph,e.getSymName().str()+"__handoff_event");
    OperationState event(e.getLoc(),EventTensorOp::getOperationName());auto size=MetricAttr::get(b.getContext(),relation.ImageCard());
    event.addAttribute("sym_name",b.getStringAttr(event_name));event.addAttribute("event_type",TypeAttr::get(RankedTensorType::get({ShapedType::kDynamic},b.getI32Type())));
    event.addAttribute("extent",size);event.addAttribute("dims",b.getArrayAttr({size}));b.setInsertionPoint(e);b.create(event);
    e.setEventAttr(FlatSymbolRefAttr::get(b.getContext(),event_name));
  }
  p.erase();c.erase();
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
  if(task.getPhaseStages().size()!=2 || task.getPhaseStages()[0]!=int64_t(p.getStage()) || task.getPhaseStages()[1]!=int64_t(c.getStage()))
    throw std::invalid_argument("handoff phase stage identities differ from the source proof");
  if(task.getPhaseSemantics()!=mlir::ArrayAttr::get(module.getContext(),{p.getSemanticAttr(),c.getSemanticAttr()}) ||
     task.getPhaseGranularities()!=mlir::ArrayAttr::get(module.getContext(),{p.getGranularity(),c.getGranularity()}))
    throw std::invalid_argument("handoff phases differ from the source access proof");
  analysis::TaskAccesses expected;
  CouplingRelation phase0,phase1;
  if(decision.getKind()=="last_arriver") {
    auto pa=HandoffTaskAccesses(p),ca=HandoffTaskAccesses(c);
    phase0=TaskIdentity(pa);phase1=proof.consumer_to_producer.Reverse();
    expected=LastArriverAccesses(pa,ca,phase1);
    auto conditions=task->getAttrOfType<mlir::ArrayAttr>("phase_conditions");
    auto ticket_ref=task->getAttrOfType<mlir::FlatSymbolRefAttr>("ticket");
    auto ticket=ticket_ref?mlir::SymbolTable::lookupNearestSymbolFrom<EventTensorOp>(task,ticket_ref):EventTensorOp{};
    if(!conditions || conditions.size()!=2 || conditions[0]!=mlir::StringAttr::get(module.getContext(),"always") ||
       conditions[1]!=mlir::StringAttr::get(module.getContext(),"last_arriver") || !ticket)
      throw std::invalid_argument("last-arriver phase lacks its ticket condition");
    auto triggers=ticket->getAttrOfType<MetricAttr>("triggers");
    auto protocol=ticket->getAttrOfType<mlir::StringAttr>("protocol");
    if(!protocol || protocol.getValue()!="last_arriver" || !triggers || !triggers.getValue().Add(proof.consumer_to_producer.Card().Scale(-1)).IsZero() ||
       !ticket.getExtent().getValue().Add(proof.consumer_to_producer.Reverse().ImageCard().Scale(-1)).IsZero())
      throw std::invalid_argument("last-arriver ticket differs from the complete reduction fibre");
  }else {
    phase0=proof.consumer_to_producer;auto ca=HandoffTaskAccesses(c);
    phase1=ca.writes.begin()->second.Reverse().Image().ImageIdentity();expected=proof.composed;
  }
  if(!Equal(mlir::cast<CouplingMapAttr>(task.getPhaseMaps()[0]).getMap(),phase0) ||
     !Equal(mlir::cast<CouplingMapAttr>(task.getPhaseMaps()[1]).getMap(),phase1))
    throw std::invalid_argument("handoff phase maps differ from the access proof");
  for(auto const& pair:{std::make_pair(task.getReads(),&expected.reads),std::make_pair(task.getWrites(),&expected.writes)}) {
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
      if(h.getKind()=="recompute" || h.getKind()=="smem_direct")RewriteForward(rewritten,plan,h,proof);
      else if(h.getKind()=="last_arriver")RewriteLastArriver(rewritten,plan,h,proof);
      else throw std::invalid_argument("unknown handoff rewrite");
    }
    (*copy)->setAttr("tmexec.active_graph",FlatSymbolRefAttr::get(module.getContext(),name));
    (*copy)->setAttr("tilemega.handoff_pending_lowering",UnitAttr::get(module.getContext()));
  }
  if(mlir::failed(mlir::verify(*copy)))throw std::invalid_argument("handoff result failed CG verification");
  module->setAttrs((*copy)->getAttrs());module.getBodyRegion().takeBody(copy->getBodyRegion());
}
void LowerServingHandoffStages(mlir::ModuleOp module) {
  if(!module->hasAttr("tilemega.handoff_pending_lowering"))return;
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto pages=module->getAttrOfType<mlir::DictionaryAttr>("tmexec.pages");
  if(!model)
    throw std::invalid_argument("serving handoff lowering requires a model stage table");
  auto source_stages=model.getAs<mlir::ArrayAttr>("stages");
  if(!source_stages)throw std::invalid_argument("serving handoff lacks runtime stages");
  std::vector<mlir::NamedAttrList> stages;
  stages.reserve(source_stages.size());
  for(auto attr:source_stages)
    stages.emplace_back(mlir::cast<mlir::DictionaryAttr>(attr));
  mlir::OpBuilder b(module.getContext());
  auto stage_kind=[&](int s) {
    if(s<0 || s>=int(stages.size()))
      throw std::invalid_argument("handoff stage lies outside runtime model");
    auto kind=stages[s].get("kind");
    if(!kind)return std::string{};
    return mlir::cast<mlir::StringAttr>(kind).getValue().str();
  };
  std::set<int> claimed;
  int lowered=0;
  for(auto plan:module.getOps<PlanOp>()) {
    auto graph=Symbol<GraphOp>(module,plan.getGraph());
    if(!graph)throw std::invalid_argument("handoff source graph is missing");
    for(auto handoff:plan.getBody().front().getOps<HandoffOp>()) {
      if(handoff.getKind()=="event")continue;
      (void)VerifyHandoffAccess(handoff);
      auto edge=Symbol<CouplingOp>(graph,handoff.getCoupling());
      auto producer=edge?Symbol<TileSpaceOp>(graph,edge.getSrc()):TileSpaceOp{};
      auto consumer=edge?Symbol<TileSpaceOp>(graph,edge.getDst()):TileSpaceOp{};
      if(!producer || !consumer)
        throw std::invalid_argument("handoff source task space is missing");
      int p=producer.getStage(),c=consumer.getStage();
      if(!claimed.insert(c).second)
        throw std::invalid_argument("runtime reducer is claimed by multiple handoffs");
      if(handoff.getKind()=="recompute") {
        if(!pages)throw std::invalid_argument("nonpaged recompute handoff is unsupported");
        if(stage_kind(p)!="kRMSNorm" || stage_kind(c)!="kGemm")
          throw std::invalid_argument("serving recompute currently requires RMSNorm to GEMM");
        bool other_consumer=false;
        for(auto outgoing:graph.getBody().front().getOps<CouplingOp>())
          if(outgoing.getSrc()==producer.getSymName() &&
             outgoing.getDst()!=consumer.getSymName())other_consumer=true;
        if(!other_consumer)stages[p].set("handoff_elided",b.getBoolAttr(true));
      }else if(handoff.getKind()=="last_arriver") {
        if(stage_kind(p)=="kFusedAttention" && stage_kind(c)=="kAttentionMerge") {
          stages[p].set("handoff_reduce_stage",b.getI64IntegerAttr(c));
          stages[c].set("handoff_elided",b.getBoolAttr(true));
        } else if(pages && stage_kind(p)=="kGemm" && stage_kind(c)=="kArgmaxReduce") {
          stages[p].set("handoff_reduce_stage",b.getI64IntegerAttr(c));
          stages[c].set("handoff_elided",b.getBoolAttr(true));
        } else if(p==c && stage_kind(p)=="kGemm") {
          // Split-K's combine is expanded after the logical stage table is
          // read. The sentinel resolves to that generated reducer slot.
          stages[p].set("handoff_reduce_stage",b.getI64IntegerAttr(
              codegen::kHandoffAutoCombine));
        } else throw std::invalid_argument(
            "serving last-arriver requires attention merge or split-K combine");
      }else
        throw std::invalid_argument("serving smem_direct needs a distinct page handoff schedule");
      ++lowered;
    }
  }
  if(!lowered)throw std::invalid_argument("pending handoff marker has no runtime decision");
  std::vector<mlir::Attribute> encoded;
  encoded.reserve(stages.size());
  for(auto& stage:stages)encoded.push_back(stage.getDictionary(module.getContext()));
  mlir::NamedAttrList updated(model);
  updated.set("stages",b.getArrayAttr(encoded));
  module->setAttr("tilemega.model_plan",updated.getDictionary(module.getContext()));
  module->removeAttr("tilemega.handoff_pending_lowering");
  module->setAttr("tmexec.runtime_handoff_lowering",b.getStringAttr("conservative_stage_slots"));
  if(mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("lowered handoff stage table failed IR verification");
}
ServingHandoffSelection SelectServingHandoffs(mlir::ModuleOp module,
    unsigned selected_classes) {
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  bool paged=module->hasAttr("tmexec.pages");
  if(!model || (!module->hasAttr("tmexec.pages") && (selected_classes&~12u)))
    throw std::invalid_argument("serving handoff selection requires a model and supported reduction classes");
  if(module->hasAttr("tmexec.runtime_handoff_lowering"))
    throw std::invalid_argument("serving handoffs were already selected");
  auto stages=model.getAs<mlir::ArrayAttr>("stages");
  if(!stages)throw std::invalid_argument("serving handoff selection lacks stages");
  auto runtime_gemms=module->getAttrOfType<mlir::ArrayAttr>("tilemega.gemm_runtime");
  auto gemm_has_split=[&](int stage) {
    auto desc=mlir::cast<mlir::DictionaryAttr>(stages[stage]);
    auto index=desc.getAs<mlir::IntegerAttr>("gemm");
    if(!index || !runtime_gemms || index.getInt()<0 ||
       index.getInt()>=static_cast<std::int64_t>(runtime_gemms.size()))
      throw std::invalid_argument("argmax handoff lacks GEMM runtime geometry");
    auto geometry=mlir::cast<mlir::DictionaryAttr>(runtime_gemms[index.getInt()]);
    auto split=geometry.getAs<mlir::IntegerAttr>("split_k");
    if(!split)throw std::invalid_argument("argmax handoff lacks split_k");
    return split.getInt()>1;
  };
  auto kind=[&](int stage)->std::string {
    if(stage<0 || stage>=int(stages.size()))return {};
    auto value=mlir::cast<mlir::DictionaryAttr>(stages[stage]).getAs<mlir::StringAttr>("kind");
    return value?value.getValue().str():std::string{};
  };
  mlir::OpBuilder b(module.getContext());
  b.setInsertionPointToEnd(module.getBody());
  mlir::OperationState graph_state(b.getUnknownLoc(),GraphOp::getOperationName());
  graph_state.addAttribute("sym_name",b.getStringAttr("serving_handoff_source"));
  graph_state.addRegion()->push_back(new mlir::Block);
  auto graph=mlir::cast<GraphOp>(b.create(graph_state));
  for(auto& op:*module.getBody())
    if(mlir::isa<TileSpaceOp,CouplingOp,EventTensorOp>(op))
      graph.getBody().front().push_back(op.clone());
  mlir::OperationState plan_state(b.getUnknownLoc(),PlanOp::getOperationName());
  plan_state.addAttribute("sym_name",b.getStringAttr("serving_handoff_plan"));
  plan_state.addAttribute("graph",mlir::FlatSymbolRefAttr::get(module.getContext(),
      graph.getSymName()));
  plan_state.addRegion()->push_back(new mlir::Block);
  b.setInsertionPointToEnd(module.getBody());
  auto plan=mlir::cast<PlanOp>(b.create(plan_state));
  b.setInsertionPointToEnd(&plan.getBody().front());
  ServingHandoffSelection selected;
  std::set<std::pair<int,int>> pairs;
  std::set<int> claimed_consumers;
  // Prefer row recompute when it and a split-K reducer would claim the same
  // logical GEMM stage. A later solver phase can compose both handoffs.
  for(int pass=0;pass<2;++pass)
  for(auto edge:graph.getBody().front().getOps<CouplingOp>()) {
    auto p=Symbol<TileSpaceOp>(graph,edge.getSrc());
    auto c=Symbol<TileSpaceOp>(graph,edge.getDst());
    if(!p || !c)continue;
    auto pair=std::make_pair(int(p.getStage()),int(c.getStage()));
    if(pairs.count(pair) || claimed_consumers.count(pair.second))continue;
    std::string choice;
    if(paged && (selected_classes&1) && kind(pair.first)=="kRMSNorm" &&
       kind(pair.second)=="kGemm") {
      bool shared=false;
      for(auto other:graph.getBody().front().getOps<CouplingOp>())
        if(other.getSrc()==p.getSymName() && other.getDst()!=c.getSymName())
          shared=true;
      if(!shared)choice="recompute";
    }
    else if((selected_classes&(2|8)) && kind(pair.first)=="kFusedAttention" &&
            kind(pair.second)=="kAttentionMerge")
      choice="last_arriver";
    else if((selected_classes&4) && pair.first==pair.second &&
            kind(pair.first)=="kGemm")
      choice="last_arriver";
    else if(paged && (selected_classes&2) && kind(pair.first)=="kGemm" &&
            kind(pair.second)=="kArgmaxReduce" && !gemm_has_split(pair.first))
      choice="last_arriver";
    else continue;
    if((pass==0)!=(choice=="recompute"))continue;
    if(choice=="last_arriver" && pair.first==pair.second) {
      // Promote only this solved split-K pair. Other proofs and the legacy
      // importer retain their source semantics.
      auto partial=p->getAttrOfType<StringAttr>("split_access_semantic");
      auto combine=c->getAttrOfType<StringAttr>("split_access_semantic");
      if(!partial || !combine)continue;
      p->setAttr("semantic",partial);c->setAttr("semantic",combine);
      // A partial workspace has exactly one reader: its reducer. Redirecting
      // any other reader would publish uncombined values as logical output.
      bool escapes=false;
      for(auto other:graph.getBody().front().getOps<CouplingOp>())
        if(other.getSrc()==p.getSymName() && other.getDst()!=c.getSymName())escapes=true;
      if(escapes)throw std::invalid_argument("split-K partial workspace has another consumer");
    }
    mlir::OperationState decision(edge.getLoc(),HandoffOp::getOperationName());
    decision.addAttribute("coupling",mlir::FlatSymbolRefAttr::get(module.getContext(),
        edge.getSymName()));
    decision.addAttribute("kind",b.getStringAttr(choice));
    auto handoff=mlir::cast<HandoffOp>(b.create(decision));
    try {(void)VerifyHandoffAccess(handoff);}
    catch(std::invalid_argument const& error) {
      if(std::getenv("TILEMEGA_HANDOFF_TRACE"))
        llvm::errs()<<"HANDOFF_REJECT producer="<<pair.first<<" consumer="
                    <<pair.second<<" kind="<<choice<<" reason="<<error.what()<<"\n";
      handoff.erase();continue;
    }
    pairs.insert(pair);claimed_consumers.insert(pair.second);
    if(choice=="recompute")++selected.recompute;
    else ++selected.last_arriver;
  }
  if(!selected.recompute && !selected.last_arriver) {
    plan.erase();graph.erase();return selected;
  }
  ApplyHandoffs(module);
  LowerServingHandoffStages(module);
  return selected;
}
void RegisterHandoffPass(){mlir::PassRegistration<ApplyPass>();}
}
