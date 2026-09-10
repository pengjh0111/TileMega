// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=5) throw std::invalid_argument("usage: tilemega-fusion-events EXPORT TARGET SEQ PAST");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=frontend::TorchExportImporter{}.Import(argv[1],context);
  int seq=std::stoi(argv[3]),past=std::stoi(argv[4]);
  auto model=ModelDescription::FromCouplingGraph(*module,{seq,past,seq+past},argv[1]);
  auto plan=codegen::ReadRuntimePlan(*module);
  auto target=TargetSpec::FromJson(argv[2]);
  std::vector<GemmConfig> configs;
  for (auto const& g:plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  int threads=model.dtype==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
  int cases=0;
  std::cout << std::unitbuf << std::setprecision(17)
      << "producer\tconsumer\tseq\tpast\tkappa\trefs_before\trefs_after\twaits_before\twaits_after\tevent_before_ns\tevent_after_ns\tdelta_ns\n";
  for (std::size_t index=1;index<model.task_semantics.size();++index) {
    auto const& p=model.task_semantics[index-1]; auto const& c=model.task_semantics[index];
    if (p.stage<0 || c.stage!=p.stage+1 || model.stages[p.stage].gemm>=0 || model.stages[c.stage].gemm>=0) continue;
    std::optional<ModelFusionCandidate> candidate;
    try { candidate=DeriveLogicalFusionCandidate(model,configs,p.op.name,c.op.name); }
    catch (std::invalid_argument const&) { continue; }
    auto po=ProjectScalarTaskOwnership(p,candidate->producer.task,model.stages[p.stage],threads);
    auto co=ProjectScalarTaskOwnership(c,candidate->consumer.task,model.stages[c.stage],threads);
    auto relation=co.ApplyRange(candidate->accesses.consumer_to_producer).ApplyRange(po.Reverse())
        .BindParams(model.MetricBindings());
    auto runtime_candidate=DeriveModelFusionCandidate(model,configs,p.stage,c.stage);
    auto actual=runtime_candidate.accesses.consumer_to_producer.BindParams(model.MetricBindings());
    if (!actual.IsSubset(relation) || !relation.IsSubset(actual))
      throw std::runtime_error("runtime physical fusion differs from logical ownership projection");
    for (int kappa:{0,1}) {
      RuntimeProjectionOptions options{target.res.num_sms*2,threads,kappa};
      auto before=ProjectRuntimeQueues(model,plan,options);
      int ps=-1,cs=-1;
      for (int s=0;s<int(before.stages.size());++s) {
        if (before.stages[s].logical_stage==p.stage) ps=s;
        if (before.stages[s].logical_stage==c.stage) cs=s;
      }
      auto after=FuseProjectedQueues(before,ps,cs,relation,options);
      CostModelOptions prices; prices.l2_events=true; prices.kappa=kappa;
      CostModel cost(target,model.dtype,prices);
      auto traits=[&](int stage) {
        BackendTraits traits; traits.threads=threads;
        traits.smem_bytes=sizeof(float)*codegen::SimtSharedElements(
            static_cast<codegen::TaskKind>(model.stages.at(stage).kind),threads,TILEMEGA_ATTENTION_MAX_TOTAL);
        return traits;
      };
      auto task_price=PriceFusionTasks(runtime_candidate,cost,traits(p.stage),traits(c.stage),{2},model);
      std::cerr << "RUNTIME_FUSION_PRICE producer=" << p.op.name << " consumer=" << c.op.name
                << " separate_ns=" << task_price.separate_ns << " fused_ns=" << task_price.fused_ns
                << " recompute_ns=" << task_price.recompute_ns << " runtime_tasks=" << task_price.consumer_tasks << '\n';
      auto priced=model;
      AttachProjectedEventMetrics(priced,plan,before);
      double old=cost.EventNs(priced,configs,{2},before.stages.size());
      AttachProjectedEventMetrics(priced,plan,after.projection);
      double now=cost.EventNs(priced,configs,{2},after.projection.stages.size());
      std::cout << p.op.name << '\t' << c.op.name << '\t' << seq << '\t' << past << '\t' << kappa
                << '\t' << before.runtime_task_refs.Eval({}) << '\t' << after.projection.runtime_task_refs.Eval({})
                << '\t' << before.runtime_wait_entries.Eval({}) << '\t' << after.projection.runtime_wait_entries.Eval({})
                << '\t' << old << '\t' << now << '\t' << now-old << '\n';
      if (!(now<old)) throw std::runtime_error("fusion event prediction did not decrease");
      ++cases;
    }
  }
  if (!cases || isl.ReferenceCount()) throw std::runtime_error("no fusion event prices or retained references");
  std::cerr << "FUSION_EVENTS cases=" << cases << " remaining=0 gpu_verified=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
