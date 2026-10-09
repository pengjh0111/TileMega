// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/BoundDependencyForm.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Codegen/RuntimeDependencyCodec.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <stdexcept>
#include <limits>

#ifndef TILEMEGA_CG_FUSION_PASS
#define TILEMEGA_CG_FUSION_PASS 1
#endif

namespace tilemega::dialect {
namespace {
void Rewrite(mlir::ModuleOp module,std::string const& producer,std::string const& consumer,
             solver::ModelFusionCandidate const& candidate,
             std::map<std::string,analysis::OperatorNode>& tasks) {
  using namespace mlir;
  TileSpaceOp p,c;
  for (auto task:module.getOps<TileSpaceOp>()) {
    if (task.getOperatorName()==producer) p=task;
    if (task.getOperatorName()==consumer) c=task;
  }
  if (!p || !c || !p.getSemantic() || !c.getSemantic()) throw std::invalid_argument("fusion phase semantics are missing");
  std::string ps=p.getSymName().str(),cs=c.getSymName().str(),name=ps+"__"+cs;
  if (SymbolTable::lookupSymbolIn(module,name)) throw std::invalid_argument("fused task symbol already exists");
  auto const& mapping=candidate.accesses.consumer_to_producer;
  auto const& cw=candidate.consumer_accesses.writes.begin()->second;
  auto identity=cw.ApplyRange(cw.Reverse());
  if(auto const& access=candidate.consumer.task.element_access) {
    auto const& semantic=access->semantic;
    // Scatter addresses overlap in the I2 envelope; task ownership does not.
    identity=analysis::ProjectTaskElements(semantic,candidate.consumer.task,access->partition,
        semantic.task_space,semantic.task_map,{},{}).Reverse().ImageIdentity();
  }
  if (!identity.IsSingleValued()) throw std::invalid_argument("consumer writes do not define unique task ownership");
  OpBuilder builder(module.getContext()); builder.setInsertionPoint(p);
  auto maps=[&](auto const& accesses) {
    NamedAttrList attributes;
    for (auto const& [tensor,map]:accesses) attributes.set(tensor,CouplingMapAttr::get(module.getContext(),map));
    return attributes.getDictionary(module.getContext());
  };
  OperationState fused(p.getLoc(),FusedTileSpaceOp::getOperationName());
  fused.addAttribute("sym_name",builder.getStringAttr(name));
  fused.addAttribute("phase_semantics",builder.getArrayAttr({p.getSemanticAttr(),c.getSemanticAttr()}));
  fused.addAttribute("phase_granularities",builder.getArrayAttr({p.getGranularity(),c.getGranularity()}));
  fused.addAttribute("phase_stages",builder.getDenseI64ArrayAttr({int64_t(p.getStage()),int64_t(c.getStage())}));
  fused.addAttribute("phase_maps",builder.getArrayAttr({CouplingMapAttr::get(module.getContext(),mapping),
      CouplingMapAttr::get(module.getContext(),identity)}));
  fused.addAttribute("reads",maps(candidate.accesses.task.reads));
  fused.addAttribute("writes",maps(candidate.accesses.task.writes));
  fused.addAttribute("task_count",MetricAttr::get(module.getContext(),candidate.accesses.task_count));
  builder.create(fused);
  if (!tasks.empty()) {
    tasks.erase(ps);tasks.erase(cs);
    tasks.emplace(name,candidate.consumer.task);
  }
  analysis::ParamBinding metric_binding;
  for(auto key:{"tilemega.theta","tilemega.g"})
    if(auto values=module->getAttrOfType<DictionaryAttr>(key))
      for(auto item:values)
        if(auto value=llvm::dyn_cast<IntegerAttr>(item.getValue()))
          metric_binding.Bind(item.getName().str(),value.getInt());
  std::vector<Operation*> erase;
  std::set<std::string> replaced_events;
  for (auto edge:module.getOps<CouplingOp>()) {
    auto src=edge.getSrc().str(),dst=edge.getDst().str();
    bool source=src==ps || src==cs,destination=dst==ps || dst==cs;
    if (!source && !destination) continue;
    replaced_events.insert(edge.getEvent().str());
    if (src==ps && dst==cs) { erase.push_back(edge); continue; }
    auto relation=edge.getRelation().getMap();
    bool exact=edge->hasAttr("shared_elements");
    if (exact) {
      auto old_shared=edge->getAttrOfType<CouplingMapAttr>("shared_elements").getMap();
      auto old_reads=edge->getAttrOfType<CouplingMapAttr>("coupled_reads").getMap();
      auto all_elements=edge->getAttrOfType<CouplingMapAttr>("consumer_elements");
      auto old_elements=all_elements?all_elements.getMap():old_reads;
      auto consumer_phase=dst==ps ? mapping : dst==cs ? identity :
          old_elements.Reverse().ImageIdentity();
      auto producer_phase=src==ps ? mapping : src==cs ? identity : relation.ImageIdentity();
      auto shared=consumer_phase.FlatProduct(producer_phase).ApplyRange(old_shared);
      auto reads=consumer_phase.ApplyRange(old_reads);
      edge->setAttr("shared_elements",CouplingMapAttr::get(module.getContext(),shared));
      edge->setAttr("coupled_reads",CouplingMapAttr::get(module.getContext(),reads));
      edge->setAttr("consumer_elements",CouplingMapAttr::get(module.getContext(),
          consumer_phase.ApplyRange(old_elements)));
      edge.setVolumeAttr(MetricAttr::get(module.getContext(),shared.BoundTaskCard()));
      edge->setAttr("interface_elements",MetricAttr::get(module.getContext(),
          reads.BoundTaskCard().SumDomain().Add(reads.Image().BoundTaskCard().Scale(-1))));
      if (auto box=edge->getAttrOfType<CouplingMapAttr>("read_box"))
        edge->setAttr("read_box",CouplingMapAttr::get(module.getContext(),
            consumer_phase.ApplyRange(box.getMap())));
    }
    // I1: coordinate composition preserves the closed parameterized C.
    if (dst==ps) relation=mapping.ApplyRange(relation);
    if (src==ps) relation=relation.ApplyRange(mapping.Reverse());
    edge.setRelationAttr(CouplingMapAttr::get(module.getContext(),relation));
    auto wait=exact?relation.BoundTaskCard():relation.Card();
    auto fanout=exact?relation.Reverse().BoundTaskCard():relation.FanoutCard();
    edge.setWaitAttr(MetricAttr::get(module.getContext(),wait));
    edge.setFanoutAttr(MetricAttr::get(module.getContext(),fanout));
    if (destination) {
      edge.setDstAttr(FlatSymbolRefAttr::get(module.getContext(),name));
      edge.setCountAttr(MetricAttr::get(module.getContext(),candidate.accesses.task_count));
    }
    if (source) edge.setSrcAttr(FlatSymbolRefAttr::get(module.getContext(),name));
    if (exact?!wait.SumDomain().SemanticallyEqual(fanout.SumDomain(),metric_binding):
        !wait.SumDomain().Add(fanout.SumDomain().Scale(-1)).IsZero())
      throw std::invalid_argument("fused graph violates sum(wait) == sum(fanout)");
    // The old window/placement/sync selection no longer describes this task.
    edge->removeAttr("wait_map");
    std::optional<std::uint32_t> event_slots;
    if (auto old=edge->getAttrOfType<DictionaryAttr>("dependency_geometry")) {
      auto binding=old.getAs<DictionaryAttr>("binding");
      if (!binding || tasks.empty())
        throw std::invalid_argument("fused bound dependency lacks task geometry");
      analysis::ParamBinding known;
      for (auto item:binding) {
        auto integer=llvm::dyn_cast<IntegerAttr>(item.getValue());
        if (!integer)throw std::invalid_argument("fused dependency binding is not integral");
        known.Bind(item.getName().str(),integer.getInt());
      }
      auto const& producer_task=tasks.at(source?name:src);
      auto const& consumer_task=tasks.at(destination?name:dst);
      analysis::CouplingEdge logical;logical.C=relation;
      auto bound=analysis::BindExactTaskDependency(logical,producer_task,consumer_task,known);
      auto pc=analysis::LinearizeTaskCoordinates(producer_task,relation.RangeDimNames(),known,"_tm_p");
      auto cc=analysis::LinearizeTaskCoordinates(consumer_task,relation.DomainDimNames(),known,"_tm_c");
      auto count=[&](analysis::OperatorNode const& task) {
        long count=task.Count().Eval(known,{});
        if(count<=0 || std::uint64_t(count)>std::numeric_limits<std::uint32_t>::max())
          throw std::invalid_argument("fused dependency task count overflows runtime IDs");
        return std::uint32_t(count);
      };
      auto producers=count(producer_task),consumers=count(consumer_task);
      edge->setAttr("dependency_geometry",EncodeBoundTaskGeometry(builder,
          {bound.encoded_relation,producers,consumers},pc,cc,known));
      edge->removeAttr("dependency_table");
      if (edge->hasAttr("dependency_counted")) {
        edge->setAttr("dependency_counted",RebindBoundCountedScatter(builder,edge,known));
        edge->removeAttr("wait_map");
      } else if (bound.table)
        edge->setAttr("dependency_table",EncodeBoundDependencyTable(builder,*bound.table,pc,cc,known));
      else edge->setAttr("wait_map",builder.getStringAttr(bound.window.ToString()));
      // Runtime event IDs use the complete producer linearization, including
      // holes in this edge's image. Image cardinality alone can underallocate.
      event_slots=producers;
    }
    edge.setSyncKindAttr(SyncKindAttr::get(module.getContext(),builder.getStringAttr("global")));
    std::string event_name=edge.getSymName().str()+"__fused_event";
    if (SymbolTable::lookupSymbolIn(module,event_name)) {
      if(edge.getEvent()!=event_name)
        throw std::invalid_argument("fused event symbol already exists");
      // A batch may rebase both endpoints of this same external edge.
      event_name+="__"+name;
      if(SymbolTable::lookupSymbolIn(module,event_name))
        throw std::invalid_argument("rebound fused event symbol already exists");
    }
    auto extent=MetricAttr::get(module.getContext(),event_slots ?
        analysis::QuasiPolynomial::Constant(*event_slots) : relation.ImageCard());
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
  FuseTaskPairs(module,{{producer,consumer}});
}
void FuseTaskPairs(mlir::ModuleOp module,
    std::vector<std::pair<std::string,std::string>> const& pairs) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_CG_FUSION_PASS
  throw std::invalid_argument("CG fusion pass is disabled");
#endif
  if (!module || pairs.empty() || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("fusion requires verified CG and an explicit task pair");
  auto model=solver::ModelDescription::FromCouplingGraph(module,{0,0,0},"fusion-pass");
  auto plan=codegen::ReadRuntimePlan(module);
  std::vector<solver::GemmConfig> configs;
  for (auto const& g:plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  std::set<std::string> selected;
  std::vector<solver::ModelFusionCandidate> candidates;
  for (auto const& [producer,consumer]:pairs) {
    if (producer.empty() || consumer.empty() || !selected.insert(producer).second ||
        !selected.insert(consumer).second)
      throw std::invalid_argument("selected fusion intervals overlap or have missing identities");
    candidates.push_back(solver::DeriveLogicalFusionCandidate(model,configs,producer,consumer));
  }
  mlir::OwningOpRef<mlir::ModuleOp> clone(llvm::cast<mlir::ModuleOp>(module->clone()));
  mlir::OpBuilder builder(module.getContext());
  std::vector<mlir::Attribute> source_dependencies;
  for (auto const& edge:plan.dependencies) {
    source_dependencies.push_back(codegen::EncodeRuntimeDependency(builder,edge));
  }
  (*clone)->setAttr("tilemega.fusion_source_dependencies",builder.getArrayAttr(source_dependencies));
  if(std::any_of(plan.dependencies.begin(),plan.dependencies.end(),
      [](auto const& edge){return edge.table.has_value() || edge.counted.has_value();})) {
    mlir::NamedAttrList binding;
    for(auto const& [name,value]:plan.task_binding.values)
      binding.set(name,builder.getI64IntegerAttr(value));
    (*clone)->setAttr("tilemega.fusion_source_task_binding",binding.getDictionary(module.getContext()));
  }
  (*clone)->setAttr("tilemega.fusion_source_cluster",builder.getI64IntegerAttr(plan.cluster_dim));
  std::map<std::string,analysis::OperatorNode> task_nodes;
  bool bound=llvm::any_of((*clone).getOps<CouplingOp>(),[](CouplingOp edge) {
    return edge->hasAttr("dependency_geometry");
  });
  if (bound) {
    auto graph=solver::InstantiateModelTasks(model,configs);
    for (auto task:(*clone).getOps<TileSpaceOp>()) {
      auto node=graph.Find(task.getOperatorName().str());
      if (!node)throw std::invalid_argument("fusion dependency lacks its semantic task");
      task_nodes.emplace(task.getSymName().str(),*node);
    }
  }
  for (std::size_t i=0;i<pairs.size();++i)
    Rewrite(*clone,pairs[i].first,pairs[i].second,candidates[i],task_nodes);
  module->setAttrs((*clone)->getAttrs());
  module.getBodyRegion().takeBody(clone->getBodyRegion());
}
void RegisterFusionPass() { mlir::PassRegistration<FusionPass>(); }
}
