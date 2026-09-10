// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/FusionResources.h>
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
    if (model.exported_tensors.empty()) throw std::runtime_error("CG exported tensor visibility was lost");
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
      analysis::ParamBinding point; point.Bind("q",0);
      double instance=cost.TaskInstanceNs(input,traits,{2},model,1,point,1);
      if (!(instance>0)) throw std::runtime_error("nonpositive real scalar instance");
      analysis::FusionAccesses replicated;
      replicated.consumer_to_producer=analysis::CouplingRelation::FromIslText(
          "{ [c] -> [q] : 0<=c<2 and q=0 }");
      replicated.fanout=replicated.consumer_to_producer.FanoutCard();
      if (FusionRecomputeNs(replicated,cost,input,traits,{2},model,1,1)!=instance)
        throw std::runtime_error("fanout two must charge one real task, not the whole stage");
      replicated.consumer_to_producer=analysis::CouplingRelation::FromIslText("{ [c=0] -> [q=0] }");
      replicated.fanout=replicated.consumer_to_producer.FanoutCard();
      if (FusionRecomputeNs(replicated,cost,input,traits,{2},model,1,1)!=0)
        throw std::runtime_error("fanout one must not charge recomputation");
      reject("instance_occupancy",[&] { cost.TaskInstanceNs(input,traits,{2},model,1,point,3); });
      reject("instance_rank",[&] { cost.TaskInstanceNs(input,traits,{2},model,1,{},1); });
      reject("instance_negative",[&] { auto p=point; p.Bind("q",-1); cost.TaskInstanceNs(input,traits,{2},model,1,p,1); });
      reject("instance_domain",[&] { auto p=point; p.Bind("q",1000000); cost.TaskInstanceNs(input,traits,{2},model,1,p,1); });
      reject("negative_memory",[&] { TaskMemoryTraffic memory; memory.local_read_bytes=-1;
        cost.TaskInstanceNs(input,traits,{2},model,1,point,1,&memory); });
      reject("infinite_memory",[&] { TaskMemoryTraffic memory;
        memory.global_read_bytes=std::numeric_limits<double>::infinity();
        cost.TaskInstanceNs(input,traits,{2},model,1,point,1,&memory); });
      reject("local_operand",[&] { TaskMemoryTraffic memory; memory.local_read_operands.insert(-1);
        cost.TaskInstanceNs(input,traits,{2},model,1,point,1,&memory); });
      reject("collective_local_input",[&] { TaskMemoryTraffic memory; memory.local_read_bytes=2;
        auto t=traits; t.stages=3; cost.TaskInstanceNs(input,t,{2},model,1,point,1,&memory); });
      reject("fusion_fanout",[&] { auto f=replicated; f.fanout=f.fanout.Scale(2); FusionRecomputeNs(f,cost,input,traits,{2},model,1,1); });
      reject("fusion_nonadjacent",[&] { DeriveModelFusionCandidate(model,configs,0,2); });
      reject("fusion_missing_semantics",[&] { auto m=model; m.task_semantics.clear(); DeriveModelFusionCandidate(m,configs,0,1); });
      reject("logical_fusion_missing",[&] { DeriveLogicalFusionCandidate(model,configs,"absent",semantic.op.name); });
      reject("logical_fusion_nonadjacent",[&] { DeriveLogicalFusionCandidate(model,configs,
          model.task_semantics.front().op.name,model.task_semantics.back().op.name); });
      reject("logical_fusion_reverse",[&] { DeriveLogicalFusionCandidate(model,configs,
          model.task_semantics.back().op.name,model.task_semantics.front().op.name); });
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
  if (branches!=27 || context.ReferenceCount()) throw std::runtime_error("incomplete scalar rejection audit");
  std::cout << "SCALAR_ERRORS branches=" << branches << " reference_delta=0\nISL_CONTEXT remaining=0\n";
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 2; }
