// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Verifier.h>
#include <stdexcept>

namespace tilemega::solver {
#ifndef TILEMEGA_FUSED_TASK_INPUT
#define TILEMEGA_FUSED_TASK_INPUT 1
#endif
std::vector<FusedTaskInput> ReadFusedTaskInputs(mlir::ModuleOp module) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_FUSED_TASK_INPUT
  throw std::invalid_argument("fused task input disabled");
#endif
  if (!module || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("fused task input requires verified CG");
  std::vector<FusedTaskInput> result;
  for (auto task:module.getOps<dialect::FusedTaskSpaceOp>()) {
    FusedTaskInput input;
    input.name=task.getSymName().str();
    input.task_count=task.getTaskCount().getValue();
    analysis::SemanticGraph semantics;
    analysis::Granularity granularity;
    for (auto [text,tiles,stage]:llvm::zip(task.getPhaseSemantics(),
         task.getPhaseGranularities(),task.getPhaseStages())) {
      ModelTaskSemantics phase;
      phase.op=analysis::DecodeSemanticOp(llvm::cast<mlir::StringAttr>(text).getValue().str());
      phase.stage=stage;
      auto dictionary=llvm::cast<mlir::DictionaryAttr>(tiles);
      phase.element_chunk=dictionary.getAs<mlir::StringAttr>("ownership")=="element_chunk";
      for (auto const& axis:phase.op.result.axes) {
        auto tile=analysis::ClosedForm::Parse(dictionary.getAs<mlir::StringAttr>(axis.name).getValue().str());
        phase.tiles.emplace(axis.name,tile);
        granularity.Tile(phase.op.name,axis.name,tile);
      }
      semantics.ops.push_back(phase.op);
      input.semantics.push_back(std::move(phase));
    }
    auto graph=analysis::Instantiate(semantics,granularity);
    std::vector<analysis::MixedArithmeticPhase> arithmetic;
    for (auto [phase,attribute]:llvm::zip(input.semantics,task.getPhaseMaps())) {
      auto mapping=llvm::cast<dialect::CouplingMapAttr>(attribute).getMap();
      auto const* node=graph.Find(phase.op.name);
      if (!node) throw std::invalid_argument("fused phase absent from task graph");
      auto work=analysis::DeriveTaskWork(phase.op,*node,{});
      analysis::ArithmeticInputs args;
      args.reduction=work.task_reduce_extent;
      args.total=work.reduce_extent;
      args.width=node->tile.at(phase.op.result.axes.size()-1).Eval({},{});
      args.dtype=phase.op.dtype;
      auto signature=analysis::InstantiateArithmetic(phase.op.arithmetic,args);
      analysis::RequireArithmeticImplementation(signature);
      auto write=analysis::ElementAccess(*node,analysis::BuildWriteMap(*node),{},
                                        analysis::AccessDomain::kPhysicalTensor);
      if (!mapping.Image().IsSubset(write.Reverse().Image()))
        throw std::invalid_argument("fusion phase map exceeds its original task domain");
      // Pull each phase's output work into consumer coordinates. Repeated
      // producer execution is counted once per consumer, not once globally.
      auto pulled=signature;
      auto pull=[&](analysis::QuasiPolynomial const& value) {
        try { (void)value.Eval({}); return value; }
        catch (std::out_of_range const&) { return value.SumAlong(mapping); }
      };
      pulled.flops_per_output_element.numerator=pull(signature.flops_per_output_element.numerator);
      pulled.transcendental_per_output_element.numerator=pull(signature.transcendental_per_output_element.numerator);
      arithmetic.push_back({std::move(pulled),mapping.ApplyRange(write).Card()});
      input.phases.push_back({*node,std::move(work),std::move(signature),node->Coordinates(),
                              std::nullopt,std::nullopt});
      input.phase_maps.push_back(std::move(mapping));
    }
    input.arithmetic=analysis::ComposeArithmetic(std::move(arithmetic));
    for (auto item:task.getReads())
      input.accesses.reads.emplace(item.getName().str(),llvm::cast<dialect::CouplingMapAttr>(item.getValue()).getMap());
    for (auto item:task.getWrites())
      input.accesses.writes.emplace(item.getName().str(),llvm::cast<dialect::CouplingMapAttr>(item.getValue()).getMap());
    if (input.phases.size()!=2)
      throw std::invalid_argument("multi-phase fusion needs sequential access composition");
    auto producer=DeriveModelTaskAccesses(input.semantics[0],input.phases[0]);
    auto consumer=DeriveModelTaskAccesses(input.semantics[1],input.phases[1]);
    std::set<std::string> internal,retained;
    for (auto const& [tensor,write]:producer.writes) {
      if (consumer.reads.count(tensor)) internal.insert(tensor);
      if (input.accesses.writes.count(tensor)) retained.insert(tensor);
    }
    auto observe=[&](analysis::SemanticOp const& op) {
      for (auto const& operand:op.operands)
        if (producer.writes.count(operand.tensor.name)) retained.insert(operand.tensor.name);
      for (auto const& read:op.element_reads)
        if (producer.writes.count(read.tensor.name)) retained.insert(read.tensor.name);
    };
    for (auto other:module.getOps<dialect::TaskSpaceOp>())
      if (auto text=other.getSemantic()) observe(analysis::DecodeSemanticOp(text->str()));
    for (auto other:module.getOps<dialect::FusedTaskSpaceOp>())
      if (other!=task) for (auto text:other.getPhaseSemantics())
        observe(analysis::DecodeSemanticOp(llvm::cast<mlir::StringAttr>(text).getValue().str()));
    auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
    if (plan) {
      auto outputs=plan.getAs<mlir::ArrayAttr>("outputs"),buffers=plan.getAs<mlir::ArrayAttr>("buffers");
      if (!outputs || !buffers) throw std::invalid_argument("fusion output plan is incomplete");
      for (auto output:outputs) {
        auto entry=llvm::dyn_cast<mlir::DictionaryAttr>(output);
        auto index=entry ? entry.getAs<mlir::IntegerAttr>("buffer") : mlir::IntegerAttr{};
        if (!index || index.getInt()<0 || index.getInt()>=static_cast<int64_t>(buffers.size()))
          throw std::invalid_argument("fusion exported buffer index is invalid");
        auto buffer=llvm::dyn_cast<mlir::DictionaryAttr>(buffers[index.getInt()]);
        auto name=buffer ? buffer.getAs<mlir::StringAttr>("name") : mlir::StringAttr{};
        if (!name) throw std::invalid_argument("fusion exported tensor name is missing");
        if (producer.writes.count(name.getValue().str())) retained.insert(name.getValue().str());
      }
    }
    auto composed=analysis::ComposeFusionAccesses(producer,consumer,internal,retained);
    auto equal=[](auto const& a,auto const& b) { return a.IsSubset(b) && b.IsSubset(a); };
    if (!equal(composed.consumer_to_producer,input.phase_maps[0]))
      throw std::invalid_argument("fusion phase map differs from semantic coupling");
    auto const& write=consumer.writes.begin()->second;
    if (!equal(write.ApplyRange(write.Reverse()),input.phase_maps[1]))
      throw std::invalid_argument("fusion consumer map is not its task identity");
    auto same_accesses=[&](auto const& a,auto const& b) {
      if (a.size()!=b.size()) return false;
      for (auto const& [tensor,map]:a) {
        auto found=b.find(tensor);
        if (found==b.end() || !equal(map,found->second)) return false;
      }
      return true;
    };
    if (!same_accesses(input.accesses.reads,composed.task.reads) ||
        !same_accesses(input.accesses.writes,composed.task.writes))
      throw std::invalid_argument("fusion physical accesses differ from phase semantics");
    result.push_back(std::move(input));
  }
  return result;
}

ModelFusionCandidate DeriveWrittenFusionCandidate(FusedTaskInput const& input,
    ModelDescription const& context,std::vector<GemmConfig> const& configs) {
  analysis::IslReferenceAudit audit(__func__);
  if (input.semantics.size()!=2 || input.phases.size()!=2 || input.phase_maps.size()!=2 ||
      configs.size()!=context.gemms.size())
    throw std::invalid_argument("written fusion pricing requires a complete selected pair");
  analysis::OperatorGraph graph;
  for (auto const& phase:input.phases) graph.nodes.push_back(phase.task);
  auto derive=[&](std::size_t index) {
    auto const& semantic=input.semantics[index];
    if (semantic.stage<0 || semantic.stage>=static_cast<int>(context.stages.size()))
      throw std::invalid_argument("fusion phase stage lies outside pricing context");
    auto const& stage=context.stages[semantic.stage];
    GemmConfig const* config=nullptr;
    if (semantic.op.kind==analysis::OperatorKind::kMatmul) {
      if (stage.gemm<0 || stage.gemm>=static_cast<int>(configs.size()))
        throw std::invalid_argument("fusion collective phase has no selected GEMM");
      config=&configs[stage.gemm];
      if (config->split_k!=1 || semantic.op.result.axes.size()!=2 ||
          !input.phases[index].task.tile[0].IsLiteral(config->tile_m) ||
          !input.phases[index].task.tile[1].IsLiteral(config->tile_n))
        throw std::invalid_argument("selected GEMM differs from written fusion granularity");
    }
    return DeriveModelTaskInput(context,semantic,graph,config,false);
  };
  auto producer=derive(0),consumer=derive(1);
  auto pa=DeriveModelTaskAccesses(input.semantics[0],producer);
  auto ca=DeriveModelTaskAccesses(input.semantics[1],consumer);
  std::set<std::string> internal,external;
  for (auto const& [tensor,write]:pa.writes) {
    if (ca.reads.count(tensor)) internal.insert(tensor);
    if (input.accesses.writes.count(tensor)) external.insert(tensor);
  }
  auto accesses=analysis::ComposeFusionAccesses(pa,ca,internal,external);
  if (!accesses.consumer_to_producer.IsSubset(input.phase_maps[0]) ||
      !input.phase_maps[0].IsSubset(accesses.consumer_to_producer))
    throw std::invalid_argument("written fusion mapping differs from price input");
  return {std::move(producer),std::move(consumer),std::move(accesses),input.arithmetic,
          std::move(pa),std::move(ca)};
}
}  // namespace tilemega::solver
