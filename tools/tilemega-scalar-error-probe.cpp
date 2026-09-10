// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=2) throw std::invalid_argument("usage: tilemega-scalar-error-probe REPO");
  analysis::IslContext context;
  int branches=0;
  {
    mlir::MLIRContext mlir;
    mlir.getOrLoadDialect<dialect::CGDialect>();
    auto cg=frontend::TorchExportImporter{}.Import(
        std::string(argv[1])+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",mlir);
    auto model=ModelDescription::FromCouplingGraph(*cg,{4,3,7},"gqa2");
    std::vector<GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
    auto graph=InstantiateModelTasks(model,configs);
    auto target=TargetSpec::FromJson(std::string(argv[1])+"/configs/targets/sm_89.json");
    CostModel cost(target,model.dtype);
    auto reject=[&](char const* name,auto&& action) {
      auto before=context.ReferenceCount();
      std::string error;
      try { action(); } catch (std::exception const& e) { error=e.what(); }
      auto after=context.ReferenceCount();
      if (error.empty() || before!=after) throw std::runtime_error(std::string("unverified rejection: ")+name);
      std::cout << name << "\tbefore=" << before << "\tafter=" << after << "\t" << error << '\n';
      ++branches;
    };
    for (auto const& semantic:model.task_semantics) {
      auto const& stage=model.stages.at(semantic.stage);
      if (stage.kind!=StageKind::kRoPE) continue;
      auto task=*graph.Find(semantic.op.name);
      auto input=DeriveModelTaskInput(model,semantic,graph,nullptr);
      BackendTraits traits; traits.threads=kTensorBF16Threads;
      reject("threads",[&] { ProjectScalarTaskOwnership(semantic,task,stage,0); });
      reject("rank",[&] { auto t=task; t.output.axes.pop_back(); ProjectScalarTaskOwnership(semantic,t,stage,traits.threads); });
      reject("collective",[&] { auto s=stage; s.kind=StageKind::kGemm; ProjectScalarTaskOwnership(semantic,task,s,traits.threads); });
      reject("element_tile",[&] { auto s=semantic; s.element_chunk=true; auto t=task;
        t.tile[0]=analysis::ClosedForm::Constant(2); ProjectScalarTaskOwnership(s,t,stage,traits.threads); });
      reject("odd_rotation",[&] { auto s=semantic; s.element_chunk=true; auto t=task;
        t.tile[0]=t.tile[1]=analysis::ClosedForm::Constant(1); auto st=stage; st.width=3;
        ProjectScalarTaskOwnership(s,t,st,traits.threads); });
      reject("missing_signature",[&] { auto s=semantic; s.op.arithmetic="absent_signature"; DeriveModelTaskInput(model,s,graph,nullptr); });
      reject("missing_flow",[&] { auto broken=input; broken.scalar_flow.reset(); cost.TaskCostNs(broken,traits,{2},model,1); });
      reject("coordinate_contract",[&] { auto broken=input; broken.cost_coordinates={"wrong"}; cost.TaskCostNs(broken,traits,{2},model,1); });
      reject("reduction_threads",[&] { auto t=traits; t.threads=3; cost.TaskCostNs(input,t,{2},model,1); });
      reject("cyclic_flow",[&] { auto broken=input; broken.scalar_flow->nodes.front().inputs={0}; cost.TaskCostNs(broken,traits,{2},model,1); });
      reject("empty_flow",[&] { auto broken=input; broken.scalar_flow->nodes.clear(); cost.TaskCostNs(broken,traits,{2},model,1); });
      reject("unbound_theta",[&] { auto m=model; m.dims=ModelDims::Symbolic("S",3); cost.TaskCostNs(input,traits,{2},m,1); });
      reject("missing_stage_semantics",[&] { auto m=model; m.task_semantics.clear(); cost.TaskStageNs(m,semantic.stage,configs.front(),{2}); });
      break;
    }
  }
  if (branches!=13 || context.ReferenceCount()) throw std::runtime_error("incomplete scalar rejection audit");
  std::cout << "SCALAR_ERRORS branches=" << branches << " reference_delta=0\nISL_CONTEXT remaining=0\n";
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 2; }
