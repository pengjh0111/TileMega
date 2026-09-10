// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Builders.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <stdexcept>

int main() try {
  using namespace tilemega;
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  int checks=0,errors=0;
  auto print=[](mlir::ModuleOp module) {
    std::string text; llvm::raw_string_ostream out(text); module.print(out); return out.str();
  };
  for (auto model:{"gqa2","mha4"}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+
        "/docs/experiments/SEQSCAN/raw/export/"+model+".json",context);
    auto description=solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},model);
    auto plan=codegen::ReadRuntimePlan(*module);
    std::vector<solver::GemmConfig> configs;
    for (auto const& g:plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
    auto candidate=solver::DeriveLogicalFusionCandidate(description,configs,"l0.s05.rope","l0.s06.append");
    auto before=print(*module);
    auto reject=[&](auto action) {
      int refs=isl.ReferenceCount(); bool caught=false;
      try { action(); } catch (std::invalid_argument const&) { caught=true; }
      if (!caught || isl.ReferenceCount()!=refs) throw std::runtime_error("unaudited fusion rewrite rejection");
      ++errors;
    };
    reject([&] { dialect::FuseTaskPair(*module,"l0.s00.norm","l0.s06.append"); });
    if (print(*module)!=before) throw std::runtime_error("failed fusion mutated original CG");
    auto count=[](auto range) { return std::distance(range.begin(),range.end()); };
    auto tasks=count(module->getOps<dialect::TaskSpaceOp>());
    auto edges=count(module->getOps<dialect::CouplingOp>());
    dialect::FuseTaskPair(*module,"l0.s05.rope","l0.s06.append");
    if (mlir::failed(mlir::verify(*module)) || count(module->getOps<dialect::TaskSpaceOp>())!=tasks-2 ||
        count(module->getOps<dialect::FusedTaskSpaceOp>())!=1 || count(module->getOps<dialect::CouplingOp>())!=edges-1)
      throw std::runtime_error("fusion did not replace task graph and remove exactly one internal edge");
    auto task=*module->getOps<dialect::FusedTaskSpaceOp>().begin();
    mlir::Builder builder(&context);
    auto reject_attribute=[&](llvm::StringRef name,mlir::Attribute replacement) {
      int refs=isl.ReferenceCount();
      auto saved=task->getAttr(name);
      if (replacement) task->setAttr(name,replacement); else task->removeAttr(name);
      bool failed=mlir::failed(mlir::verify(*module));
      task->setAttr(name,saved);
      if (!failed || isl.ReferenceCount()!=refs)
        throw std::runtime_error("invalid phase metadata accepted or retained ISL references");
      ++errors;
    };
    reject_attribute("phase_granularities",{});
    reject_attribute("phase_granularities",builder.getArrayAttr({}));
    reject_attribute("phase_stages",builder.getDenseI64ArrayAttr({-1,6}));
    reject_attribute("phase_stages",builder.getDenseI64ArrayAttr({5}));
    if (task.getWrites().get("l0.k_rot") || !task.getWrites().get("l0.full_k") || task.getReads().get("l0.k_rot"))
      throw std::runtime_error("internal tensor escaped the composed task");
    reject([&] { (void)codegen::CouplingGraphToCUDA{}.Lower(*module); });
    reject([&] { (void)solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},model); });
    auto saved=task.getTaskCountAttr();
    task.setTaskCountAttr(dialect::MetricAttr::get(&context,analysis::QuasiPolynomial::Constant(0)));
    if (mlir::succeeded(mlir::verify(*module))) throw std::runtime_error("false fused task count accepted");
    task.setTaskCountAttr(saved);
    auto inputs=solver::ReadFusedTaskInputs(*module);
    if (inputs.size()!=1 || inputs.front().phases.size()!=2)
      throw std::runtime_error("fused arithmetic reader lost phases");
    auto reads=task.getReads();
    task->setAttr("reads",builder.getDictionaryAttr({}));
    reject([&] { (void)solver::ReadFusedTaskInputs(*module); });
    task->setAttr("reads",reads);
    auto const& input=inputs.front();
    for (std::size_t phase=0;phase<2;++phase) {
      auto const& expected=candidate.arithmetic.phases[phase];
      auto const& actual=input.arithmetic.phases[phase];
      if (!actual.output_elements.SemanticallyEqual(expected.output_elements,{}) ||
          actual.arithmetic.flops_use_mma!=expected.arithmetic.flops_use_mma ||
          !actual.arithmetic.flops_per_output_element.numerator.SemanticallyEqual(
              expected.arithmetic.flops_per_output_element.numerator,{}) ||
          actual.arithmetic.flops_per_output_element.denominator!=expected.arithmetic.flops_per_output_element.denominator)
        throw std::runtime_error("written fusion arithmetic differs from selected candidate");
    }
    for (auto edge:module->getOps<dialect::CouplingOp>())
      for (long seq:{1L,4L,128L,512L,2048L}) for (long past:{0L,3L,512L}) {
        description.dims={int(seq),int(past),int(seq+past)};
        auto theta=description.MetricBindings();
        auto wait=edge.getWait().getValue().SubstituteParams(theta).SumDomain().Eval(theta);
        auto fanout=edge.getFanout().getValue().SubstituteParams(theta).SumDomain().Eval(theta);
        if (wait!=fanout) throw std::runtime_error("rewired graph incidence identity failed at "+edge.getSymName().str());
        ++checks;
      }
  }
  if (isl.ReferenceCount()) throw std::runtime_error("fusion rewrite retained ISL references");
  std::cout << "FUSION_REWRITE edge_identities=" << checks << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
