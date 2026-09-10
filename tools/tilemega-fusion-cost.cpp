// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=4) throw std::invalid_argument("usage: tilemega-fusion-cost REPO MODEL COMPILED_REGISTERS");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=frontend::TorchExportImporter{}.Import(std::string(argv[1])+
      "/docs/experiments/SEQSCAN/raw/export/"+argv[2]+".json",context);
  auto model=ModelDescription::FromCouplingGraph(*module,{4,3,7},argv[2]);
  auto target=TargetSpec::FromJson(std::string(argv[1])+"/configs/targets/sm_89.json");
  CostModel cost(target,model.dtype);
  int registers=std::stoi(argv[3]);
  if (registers<=0) throw std::invalid_argument("compiled register resource is required");
  std::vector<GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
  int accepted=0,errors=0;
  std::cout << std::setprecision(17) << "model\tproducer\tconsumer\texisting_epilogue\tproducer_tasks\tconsumer_tasks\trecompute_tasks\tproducer_waves\tconsumer_waves\tseparate_ns\tfused_ns\trecompute_ns\tglobal_bytes_before\tglobal_bytes_after\tlocal_bytes\tshared_bytes\tregisters\n";
  for (std::size_t index=1;index<model.task_semantics.size();++index) {
    auto const& ps=model.task_semantics[index-1];
    auto const& cs=model.task_semantics[index];
    std::optional<ModelFusionCandidate> candidate;
    try { candidate=DeriveLogicalFusionCandidate(model,configs,ps.op.name,cs.op.name); }
    catch (std::invalid_argument const&) { continue; }
    auto traits=[&](DerivedTaskInput const& input,ModelTaskSemantics const& semantic) {
      if (input.task.kind==analysis::OperatorKind::kMatmul) return TensorBF16Traits(128,128,16,3);
      BackendTraits result; result.threads=kTensorBF16Threads;
      auto kind=static_cast<codegen::TaskKind>(model.stages[semantic.stage].kind);
      if (kind==codegen::TaskKind::kGemm) kind=codegen::TaskKind::kElementwise;
      result.smem_bytes=sizeof(float)*codegen::SimtSharedElements(kind,result.threads,TILEMEGA_ATTENTION_MAX_TOTAL);
      return result;
    };
    auto p=traits(candidate->producer,ps),c=traits(candidate->consumer,cs);
    std::map<std::string,int> types;
    for (auto const& [name,access]:candidate->accesses.intermediate_tiles) types.emplace(name,2);
    auto resources=DeriveFusionResources(candidate->accesses,model.MetricBindings(),types,p,registers,c,registers);
    auto price=PriceFusionTasks(*candidate,cost,p,c,{2},model);
    if (!(price.fused_ns>0 && price.separate_ns>0) ||
        price.global_bytes_after>=price.global_bytes_before || price.local_bytes<=0)
      throw std::runtime_error("fusion physical traffic was not redirected");
    std::cout << argv[2] << '\t' << ps.op.name << '\t' << cs.op.name << '\t' << (ps.stage==cs.stage)
              << '\t' << price.producer_tasks << '\t' << price.consumer_tasks << '\t' << price.recomputed_tasks
              << '\t' << price.producer_waves << '\t' << price.consumer_waves << '\t' << price.separate_ns
              << '\t' << price.fused_ns << '\t' << price.recompute_ns << '\t' << price.global_bytes_before
              << '\t' << price.global_bytes_after << '\t' << price.local_bytes << '\t' << resources.shared_bytes
              << '\t' << resources.registers << '\n';
    auto reject=[&](auto action) {
      int before=isl.ReferenceCount(); bool rejected=false;
      try { action(); } catch (std::invalid_argument const&) { rejected=true; }
      if (!rejected || before!=isl.ReferenceCount()) throw std::runtime_error("fusion pricing error retained references");
      ++errors;
    };
    reject([&] { auto bad=*candidate; bad.accesses.fanout=bad.accesses.fanout.Scale(2);
      PriceFusionTasks(bad,cost,p,c,{2},model); });
    reject([&] { auto wrong=c; wrong.threads/=2; PriceFusionTasks(*candidate,cost,p,wrong,{2},model); });
    reject([&] { PriceFusionTasks(*candidate,cost,p,c,{0},model); });
    if (ps.stage==cs.stage) {
      // Counterfactual tile domain only: give each consumer one output row
      // while retaining the producer tile, so fanout is genuinely nonunit.
      auto replicated=model;
      auto& semantic=replicated.task_semantics[index];
      semantic.stage=replicated.stages.size();
      auto stage=model.stages[cs.stage]; stage.kind=StageKind::kElementwise; stage.gemm=-1;
      replicated.stages.push_back(stage);
      semantic.tiles[semantic.op.domain.front().name]=analysis::ClosedForm::Constant(1);
      auto pair=DeriveLogicalFusionCandidate(replicated,configs,ps.op.name,cs.op.name);
      auto extra=PriceFusionTasks(pair,cost,p,c,{2},replicated);
      if (extra.recomputed_tasks<=0 || extra.recompute_ns<=0 ||
          extra.global_bytes_after<=price.global_bytes_after)
        throw std::runtime_error("nonunit fanout failed to charge replicated producer work");
      std::cerr << "FUSION_FANOUT producer=" << extra.producer_tasks << " consumer=" << extra.consumer_tasks
                << " recomputed=" << extra.recomputed_tasks << " recompute_ns=" << std::setprecision(17)
                << extra.recompute_ns << " fused_ns=" << extra.fused_ns << " global_bytes=" << extra.global_bytes_after << '\n';
    }
    ++accepted;
  }
  if (!accepted || isl.ReferenceCount()) throw std::runtime_error("fusion task prices missing or leaked");
  std::cerr << "FUSION_COST candidates=" << accepted << " error_branches=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
