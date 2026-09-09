// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/IR/MLIRContext.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

int main(int argc,char** argv) try {
  tilemega::analysis::IslContext isl_context;
  if (argc!=4) throw std::invalid_argument("usage: tilemega-event-price EXPORT.json TARGET.json CTAS_PER_SM");
  using namespace tilemega;
  using namespace tilemega::solver;
  auto target=TargetSpec::FromJson(argv[2]);
  Residency residency{std::stoi(argv[3])};
  if (residency.ctas_per_sm<=0) throw std::invalid_argument("invalid residency");
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  frontend::ImportOptions import;
  import.rope_tile_per_block=import.kv_tile_per_block=true;
  import.activation_tile_per_block=import.combiner_tile_per_block=true;
  auto module=frontend::TorchExportImporter{}.Import(argv[1],context,nullptr,import);
  auto seed=codegen::ReadRuntimePlan(*module);
  frontend::GemmGranularity shape{128,128,16,3,1};
  import.gemms.assign(seed.gemms.size(),shape);
  module=frontend::TorchExportImporter{}.Import(argv[1],context,nullptr,import);
  auto plan=codegen::ReadRuntimePlan(*module);
  auto original=ModelDescription::FromCouplingGraph(*module,ModelDims::Symbolic("S",3),argv[1]);
  if (original.dtype!=ScalarType::kBF16) throw std::invalid_argument("A9 gate requires BF16 CG");
  std::vector<GemmConfig> configs(plan.gemms.size(),{shape.tile_m,shape.tile_n,shape.tile_k,shape.stages,shape.split_k});
  std::map<int,std::pair<long,double>> aggregate;
  bool failed=false;
  std::cout << std::setprecision(17)
            << "seq\tpast\tkappa\tstages\ttask_refs\twaits\tmax_worker_tasks\tevent_ns\tdelta_from_k0_ns\n";
  for (int kappa:{0,1,2}) {
    auto symbolic=original;
    RuntimeProjectionOptions projection{target.res.num_sms*residency.ctas_per_sm,
                                       kTensorBF16Threads,kappa};
    projection.partition_worker_counts=true; projection.split_count_periods=true;
    AttachRuntimeEventMetrics(symbolic,plan,projection);
    CostModelOptions options; options.l2_events=true; options.kappa=kappa;
    CostModel cost(target,ScalarType::kBF16,options);
    auto const& metrics=*symbolic.coupling_metrics.runtime;
    std::cerr << "KAPPA " << kappa << " refs=" << metrics.task_refs.ToString()
              << " waits=" << metrics.wait_entries.ToString() << '\n';
    for (int seq:{1,4,16,128,512,2048}) {
      analysis::ParamBinding theta; theta.Bind("S",seq);
      auto bound=symbolic.SubstituteParams(theta);
      auto const& work=*bound.coupling_metrics.runtime;
      double price=cost.EventNs(bound,configs,residency,work.stage_count);
      // Resolve through the direct concrete input as well as the explicitly
      // substituted input. The floating-point expression order is identical.
      auto direct=symbolic; direct.dims={seq,3,seq+3};
      double comparison=cost.EventNs(direct,configs,residency,work.stage_count);
      if (std::memcmp(&price,&comparison,sizeof(double)))
        throw std::runtime_error("parameter substitution changes event price bits");
      long waits=work.wait_entries.Eval({});
      if (!kappa) aggregate[seq]={waits,price};
      double delta=price-aggregate.at(seq).second;
      if (kappa==1 && (delta==0 || (delta>0)!=(waits>aggregate.at(seq).first))) failed=true;
      std::cout << seq << '\t' << 3 << '\t' << kappa << '\t' << work.stage_count
                << '\t' << work.task_refs.Eval({}) << '\t' << waits
                << '\t' << work.max_worker_task_refs.Eval({}) << '\t' << price << '\t' << delta << std::endl;
    }
  }
  if (failed) throw std::runtime_error("A9.3 local stop: zero or wrong-sign kappa price difference");
  std::cerr << "EVENT_PRICE_GATE PASS; candidate-specific DP integration not claimed\n";
} catch (std::exception const& e) {
  std::cerr << "event-price: " << e.what() << '\n'; return 2;
}
