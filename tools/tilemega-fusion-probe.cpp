// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>

int main(int argc,char** argv) try {
  if (argc!=2) throw std::invalid_argument("usage: tilemega-fusion-probe EXPORT.json");
  tilemega::analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  auto module=tilemega::frontend::TorchExportImporter{}.Import(argv[1],context);
  auto model=tilemega::solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},argv[1]);
  std::vector<tilemega::solver::GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
  std::cout << "producer\tconsumer\tproducer_stage\tconsumer_stage\tstatus\ttask_count\trecompute_tasks\tconservation\treason\n";
  int accepted=0,rejected=0;
  for (std::size_t stage=1;stage<model.task_semantics.size();++stage) {
    auto const& producer=model.task_semantics[stage-1];
    auto const& consumer=model.task_semantics[stage];
    std::cout << producer.op.name << '\t' << consumer.op.name << '\t'
              << producer.stage << '\t' << consumer.stage << '\t';
    int before=isl.ReferenceCount();
    try {
      auto candidate=tilemega::solver::DeriveLogicalFusionCandidate(model,configs,producer.op.name,consumer.op.name);
      auto const& access=candidate.accesses;
      if (!access.consumer_to_producer.Card().SumDomain().SemanticallyEqual(
              access.fanout.SumDomain(),{}))
        throw std::runtime_error("fusion wait/fanout conservation failed");
      auto binding=model.MetricBindings();
      std::cout << (producer.stage==consumer.stage ? "EXISTING_RUNTIME_FUSION" : "LEGAL_L_TASK")
                << '\t' << access.task_count.Eval(binding)
                << '\t' << access.recompute_tasks.Eval(binding)
                << "\t1\t" << access.task_count.ToString() << '\n';
      ++accepted;
    } catch (std::invalid_argument const& error) {
      std::cout << "REJECT\t\t\t\t" << error.what() << '\n';
      ++rejected;
    }
    if (isl.ReferenceCount()!=before) throw std::runtime_error("fusion candidate retained ISL references");
  }
  std::cerr << "FUSION_CANDIDATES accepted=" << accepted << " rejected=" << rejected
            << " remaining=" << isl.ReferenceCount() << '\n';
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n'; return 2;
}
