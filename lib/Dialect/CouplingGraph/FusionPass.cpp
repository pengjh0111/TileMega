// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <stdexcept>

#ifndef TILEMEGA_CG_FUSION_PASS
#define TILEMEGA_CG_FUSION_PASS 1
#endif

namespace tilemega::dialect {
namespace {
void Rewrite(mlir::ModuleOp module,std::string const& producer,std::string const& consumer) {
  using namespace mlir;
  auto model=solver::ModelDescription::FromCouplingGraph(module,{0,0,0},"fusion-pass");
  auto plan=codegen::ReadRuntimePlan(module);
  std::vector<solver::GemmConfig> configs;
  for (auto const& g:plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  auto candidate=solver::DeriveLogicalFusionCandidate(model,configs,producer,consumer);
  TaskSpaceOp p,c;
  for (auto task:module.getOps<TaskSpaceOp>()) {
    if (task.getOperatorName()==producer) p=task;
    if (task.getOperatorName()==consumer) c=task;
  }
  if (!p || !c || !p.getSemantic() || !c.getSemantic()) throw std::invalid_argument("fusion phase semantics are missing");
  std::string ps=p.getSymName().str(),cs=c.getSymName().str(),name=ps+"__"+cs;
  if (SymbolTable::lookupSymbolIn(module,name)) throw std::invalid_argument("fused task symbol already exists");
  auto const& mapping=candidate.accesses.consumer_to_producer;
  auto const& cw=candidate.consumer_accesses.writes.begin()->second;
  auto identity=cw.ApplyRange(cw.Reverse());
  if (!identity.IsSingleValued()) throw std::invalid_argument("consumer writes do not define unique task ownership");
  OpBuilder builder(module.getContext()); builder.setInsertionPoint(p);
  auto maps=[&](auto const& accesses) {
    NamedAttrList attributes;
    for (auto const& [tensor,map]:accesses) attributes.set(tensor,CouplingMapAttr::get(module.getContext(),map));
    return attributes.getDictionary(module.getContext());
  };
  OperationState fused(p.getLoc(),FusedTaskSpaceOp::getOperationName());
  fused.addAttribute("sym_name",builder.getStringAttr(name));
  fused.addAttribute("phase_semantics",builder.getArrayAttr({p.getSemanticAttr(),c.getSemanticAttr()}));
  fused.addAttribute("phase_maps",builder.getArrayAttr({CouplingMapAttr::get(module.getContext(),mapping),
      CouplingMapAttr::get(module.getContext(),identity)}));
  fused.addAttribute("reads",maps(candidate.accesses.task.reads));
  fused.addAttribute("writes",maps(candidate.accesses.task.writes));
  fused.addAttribute("task_count",MetricAttr::get(module.getContext(),candidate.accesses.task_count));
  builder.create(fused);
  std::vector<Operation*> erase;
  std::set<std::string> replaced_events;
  for (auto edge:module.getOps<CouplingOp>()) {
    auto src=edge.getSrc().str(),dst=edge.getDst().str();
    bool source=src==ps || src==cs,destination=dst==ps || dst==cs;
    if (!source && !destination) continue;
    replaced_events.insert(edge.getEvent().str());
    if (src==ps && dst==cs) { erase.push_back(edge); continue; }
    auto relation=edge.getRelation().getMap();
    // I1: coordinate composition preserves the closed parameterized C.
    if (dst==ps) relation=mapping.ApplyRange(relation);
    if (src==ps) relation=relation.ApplyRange(mapping.Reverse());
    edge.setRelationAttr(CouplingMapAttr::get(module.getContext(),relation));
    edge.setWaitAttr(MetricAttr::get(module.getContext(),relation.Card()));
    edge.setFanoutAttr(MetricAttr::get(module.getContext(),relation.FanoutCard()));
    if (destination) {
      edge.setDstAttr(FlatSymbolRefAttr::get(module.getContext(),name));
      edge.setCountAttr(MetricAttr::get(module.getContext(),candidate.accesses.task_count));
    }
    if (source) edge.setSrcAttr(FlatSymbolRefAttr::get(module.getContext(),name));
    if (!relation.Card().SumDomain().Add(relation.FanoutCard().SumDomain().Scale(-1)).IsZero())
      throw std::invalid_argument("fused graph violates sum(wait) == sum(fanout)");
    // The old window/placement/sync selection no longer describes this task.
    edge->removeAttr("wait_map");
    edge.setSyncKindAttr(SyncKindAttr::get(module.getContext(),builder.getStringAttr("global")));
    std::string event_name=edge.getSymName().str()+"__fused_event";
    if (SymbolTable::lookupSymbolIn(module,event_name)) throw std::invalid_argument("fused event symbol already exists");
    auto extent=MetricAttr::get(module.getContext(),relation.ImageCard());
    OperationState event(edge.getLoc(),EventTensorOp::getOperationName());
    event.addAttribute("sym_name",builder.getStringAttr(event_name));
    event.addAttribute("event_type",TypeAttr::get(RankedTensorType::get({ShapedType::kDynamic},builder.getI32Type())));
    event.addAttribute("extent",extent);
    event.addAttribute("dims",builder.getArrayAttr({extent}));
    builder.setInsertionPoint(edge); builder.create(event);
    edge.setEventAttr(FlatSymbolRefAttr::get(module.getContext(),event_name));
  }
  for (auto op:erase) op->erase();
  erase.clear();
  for (auto op:module.getOps<PlacementOp>()) if (op.getTask()==ps || op.getTask()==cs) erase.push_back(op);
  for (auto op:module.getOps<ImplementationOp>()) if (op.getTask()==ps || op.getTask()==cs) erase.push_back(op);
  for (auto op:erase) op->erase();
  for (auto const& event_name:replaced_events) {
    bool used=false;
    for (auto edge:module.getOps<CouplingOp>()) used|=edge.getEvent()==event_name;
    if (!used) if (auto event=SymbolTable::lookupSymbolIn(module,event_name)) event->erase();
  }
  p.erase(); c.erase();
  module->setAttr("tilemega.fusion_pending_lowering",builder.getUnitAttr());
  if (failed(verify(module))) throw std::invalid_argument("fused CG verification failed");
}

struct FusionPass : mlir::PassWrapper<FusionPass,mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)
  FusionPass()=default;
  FusionPass(FusionPass const& other):PassWrapper(other) {}
  mlir::Pass::Option<std::string> producer{*this,"producer",llvm::cl::desc("Selected producer L-task")};
  mlir::Pass::Option<std::string> consumer{*this,"consumer",llvm::cl::desc("Selected consumer L-task")};
  llvm::StringRef getArgument() const final { return "tilemega-fuse-task-pair"; }
  llvm::StringRef getDescription() const final { return "Apply a selected adjacent L-task fusion"; }
  void runOnOperation() override {
    try { FuseTaskPair(getOperation(),producer,consumer); }
    catch (std::exception const& error) { getOperation().emitError(error.what()); signalPassFailure(); }
  }
};
}
void FuseTaskPair(mlir::ModuleOp module,std::string const& producer,std::string const& consumer) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_CG_FUSION_PASS
  throw std::invalid_argument("CG fusion pass is disabled");
#endif
  if (!module || producer.empty() || consumer.empty() || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("fusion requires verified CG and an explicit task pair");
  mlir::OwningOpRef<mlir::ModuleOp> clone(llvm::cast<mlir::ModuleOp>(module->clone()));
  Rewrite(*clone,producer,consumer);
  module->setAttrs((*clone)->getAttrs());
  module.getBodyRegion().takeBody(clone->getBodyRegion());
}
void RegisterFusionPass() { mlir::PassRegistration<FusionPass>(); }
}
