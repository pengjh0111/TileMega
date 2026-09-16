// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <fstream>
#include <cmath>
int main(int argc,char** argv) try {
  if (argc!=4) throw std::runtime_error("check_body TARGET EXPORT OUT_JSON");
  tilemega::analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  auto module=tilemega::frontend::TorchExportImporter{}.Import(argv[2],context);
  auto model=tilemega::solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},"body-calibration-test");
  auto target=tilemega::TargetSpec::FromJson(argv[1]);
  target.ToJson(argv[3]);auto roundtrip=tilemega::TargetSpec::FromJson(argv[3]);
  auto const& a=target.calib_bf16.task_body;auto const& b=roundtrip.calib_bf16.task_body;
  // The existing JSON writer retains ten significant decimal digits.
  auto same=[](auto const& x,auto const& y) {
    if (x.size()!=y.size()) return false;
    for (std::size_t i=0;i<x.size();++i)
      if (std::abs(x[i]-y[i])>1e-9*std::max(1.,std::abs(x[i]))) return false;
    return true;
  };
  if (!same(a.fixed,b.fixed) || !same(a.loop_body,b.loop_body) || !same(a.loop_wait,b.loop_wait) ||
      !same(a.loop_fixed,b.loop_fixed) || a.samples!=b.samples || a.source!=b.source)
    throw std::runtime_error("TaskBody calibration JSON round trip changed");
  auto const& events=target.event_bf16;auto const& again=roundtrip.event_bf16;
  for (auto rates:{std::make_pair(events.task_publication,again.task_publication),
                  std::make_pair(events.task_wait,again.task_wait)}) {
    if (rates.first.ns.has_value()!=rates.second.ns.has_value() ||
        (rates.first.ns && std::abs(*rates.first.ns-*rates.second.ns)>1e-9*std::max(1.,*rates.first.ns)) ||
        rates.first.unit!=rates.second.unit) throw std::runtime_error("task event rate round trip changed");
  }
  if (events.task_source!=again.task_source || events.task_source_sha256!=again.task_source_sha256)
    throw std::runtime_error("task event provenance changed");
  using namespace tilemega::solver;
  CostModel cost(target,model.dtype);std::vector<DpCandidate> candidates;
  // Resource admission is deliberately restricted to two CTAs for this CPU
  // decomposition test; these register values are not occupancy evidence.
  for (int k:{16,32,64}) candidates.push_back({{32,16,k,2,1},255,TensorBF16SmemBytes(32,16,k,2)});
  ChainDpOptions options;options.mode=DpMode::kUniform;options.max_ctas_per_sm=2;
  ChainDpStats stats;auto selected=ChainDP(cost,candidates).Solve(model,options,&stats);
  std::cout << "CALIBRATED_DP feasible=" << selected.feasible
      << " decomposition_error_ns=" << stats.decomposition_error_ns << '\n';
  if (!selected.feasible || stats.decomposition_error_ns>1e-6)
    throw std::runtime_error("calibrated ChainDP differs from unified Evaluate");
  double total=cost.Evaluate(model,selected.configs,selected.residency).total_ns;
  if (std::abs(total-selected.cost.total_ns)>1e-6)
    throw std::runtime_error("calibrated selected plan cost differs");
  std::cout << "CALIBRATION_ROUNDTRIP PASS samples=" << a.samples << '\n'
      << "CHAIN_DP_EVALUATE PASS error_ns=" << stats.decomposition_error_ns
      << " selected_k=" << selected.configs.front().tile_k << " total_ns=" << total << '\n';
} catch (std::exception const& e) {std::cerr << e.what() << '\n';return 1;}
