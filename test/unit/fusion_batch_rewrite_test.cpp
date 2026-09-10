// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <algorithm>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <stdexcept>

int main() try {
  using namespace tilemega;
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  int errors=0,identities=0,tasks=0;
  for (auto name:{"gqa2","mha4"}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+
        "/docs/experiments/SEQSCAN/raw/export/"+name+".json",context);
    auto model=solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},name);
    auto plan=codegen::ReadRuntimePlan(*module);
    std::vector<std::pair<std::string,std::string>> pairs;
    for (std::size_t i=1;i<model.task_semantics.size();++i) {
      auto const& p=model.task_semantics[i-1]; auto const& c=model.task_semantics[i];
      if (p.op.arithmetic=="rope" && c.op.arithmetic=="kv_append") pairs.emplace_back(p.op.name,c.op.name);
    }
    auto print=[&] { std::string text; llvm::raw_string_ostream output(text); module->print(output); return output.str(); };
    auto original=print();
    auto reject=[&](auto bad) {
      int refs=isl.ReferenceCount(); bool caught=false;
      try { dialect::FuseTaskPairs(*module,bad); }
      catch (std::invalid_argument const&) { caught=true; }
      if (!caught || refs!=isl.ReferenceCount() || print()!=original)
        throw std::runtime_error("fusion batch failure mutated module or retained references");
      ++errors;
    };
    auto overlap=pairs; overlap.push_back(pairs.front()); reject(overlap);
    auto missing=pairs; missing.push_back({"missing","missing_consumer"}); reject(missing);
    dialect::FuseTaskPairs(*module,pairs);
    auto inputs=solver::ReadFusedTaskInputs(*module);
    if (inputs.size()!=pairs.size()) throw std::runtime_error("batch rewrite lost selected fusion intervals");
    tasks+=inputs.size();
    auto phases=solver::ModelDescription::FromFusionPhases(*module,{4,3,7},name);
    auto source_plan=codegen::ReadFusionSourcePlan(*module);
    bool refused=false;
    auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
    solver::CostModel cost(target);
    try {
      (void)cost.Evaluate(phases,solver::GemmConfig{},solver::Residency{2});
    } catch (std::invalid_argument const&) { refused=true; }
    if (!refused) throw std::runtime_error("phase context was accepted as a fused model price");
    if (!phases.fusion_phase_context || phases.task_semantics.size()!=model.task_semantics.size() ||
        !phases.coupling_metrics.edges.empty() || source_plan.dependencies.size()!=plan.dependencies.size() ||
        source_plan.parameter_ranges!=plan.parameter_ranges || source_plan.ownership_flags!=plan.ownership_flags)
      throw std::runtime_error("fusion source projection changed phase geometry or parameter bounds");
    for (auto const& semantic:model.task_semantics) {
      auto found=std::find_if(phases.task_semantics.begin(),phases.task_semantics.end(),
          [&](auto const& phase) { return phase.op.name==semantic.op.name; });
      if (found==phases.task_semantics.end() || found->op.Serialize()!=semantic.op.Serialize() ||
          found->stage!=semantic.stage || found->element_chunk!=semantic.element_chunk ||
          found->tiles.size()!=semantic.tiles.size())
        throw std::runtime_error("fusion source projection lost a semantic phase");
      for (auto const& [axis,tile]:semantic.tiles)
        if (found->tiles.at(axis).ToString()!=tile.ToString())
          throw std::runtime_error("fusion source projection changed a tile");
    }
    for (std::size_t i=0;i<plan.dependencies.size();++i) {
      auto const& a=plan.dependencies[i]; auto const& b=source_plan.dependencies[i];
      if (a.producer!=b.producer || a.consumer!=b.consumer || a.window!=b.window)
        throw std::runtime_error("fusion source projection changed an original dependency");
    }
    for (auto edge:module->getOps<dialect::CouplingOp>()) {
      auto theta=model.MetricBindings();
      if (edge.getWait().getValue().SumDomain().Eval(theta)!=edge.getFanout().getValue().SumDomain().Eval(theta))
        throw std::runtime_error("batch fusion incidence identity failed");
      ++identities;
    }
  }
  if (isl.ReferenceCount()) throw std::runtime_error("batch fusion leaked references");
  std::cout << "FUSION_BATCH tasks=" << tasks << " identities=" << identities << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
