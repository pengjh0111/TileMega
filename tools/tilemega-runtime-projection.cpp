// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
  tilemega::analysis::IslContext isl_context;
  if (argc != 12 && argc != 13)
    throw std::invalid_argument("usage: tilemega-runtime-projection EXPORT.json "
        "GRID THREADS KAPPA {PAST|symbolic} TILE_M TILE_N TILE_K STAGES SPLIT {tile|element} [CG_ORDER=0|1]");
  tilemega::solver::RuntimeProjectionOptions options{
      std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4])};
  if (argc==13) {
    if (std::string(argv[12])!="0" && std::string(argv[12])!="1")
      throw std::invalid_argument("CG_ORDER must be 0 or 1");
    options.cg_split_task_order = std::string(argv[12])=="1";
  }
  bool symbolic_past = std::string(argv[5]) == "symbolic";
  int past = symbolic_past ? 0 : std::stoi(argv[5]);
  tilemega::frontend::GemmGranularity shape{
      std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),
      std::stoi(argv[9]),std::stoi(argv[10])};
  std::string ownership = argv[11];
  if (ownership != "tile" && ownership != "element")
    throw std::invalid_argument("ownership must be tile or element");
  tilemega::frontend::ImportOptions import;
  import.rope_tile_per_block = import.kv_tile_per_block =
      import.activation_tile_per_block = import.combiner_tile_per_block = ownership == "tile";
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  auto module = tilemega::frontend::TorchExportImporter{}.Import(argv[1],context,nullptr,import);
  auto seed = tilemega::codegen::ReadRuntimePlan(*module);
  import.gemms.assign(seed.gemms.size(),shape);
  module = tilemega::frontend::TorchExportImporter{}.Import(argv[1],context,nullptr,import);
  auto plan = tilemega::codegen::ReadRuntimePlan(*module);
  auto dims = tilemega::solver::ModelDims::Symbolic("S",past);
  if (symbolic_past) dims.past_parameter = "P";
  auto model = tilemega::solver::ModelDescription::FromCouplingGraph(*module,dims,argv[1]);
  for (auto const& [name,range] : plan.parameter_ranges)
    std::cerr << "CG_PARAMETER " << name << "=[" << range.first << ',' << range.second << "]\n";
  std::cerr << "CG_ROLES seq=" << model.seq_metric_parameter
            << " past=" << model.past_metric_parameter << '\n';
  auto projection = tilemega::solver::ProjectRuntimeQueues(model,plan,options);
  std::cerr << "runtime_task_refs=" << projection.runtime_task_refs.ToString()
            << "\nruntime_wait_entries=" << projection.runtime_wait_entries.ToString()
            << "\nmax_worker_task_refs=" << projection.max_worker_task_refs.ToString() << '\n';
  std::cout << "seq\tpast\tsplit\tkappa\tstages\ttask_refs\twaits\tmax_worker_tasks\n";
  auto pasts = symbolic_past ? std::vector<int>{0,3,512} : std::vector<int>{past};
  for (int seq : {1,4,128,512,2048}) for (int value_past : pasts) {
    tilemega::analysis::ParamBinding theta; theta.Bind("S",seq).Bind("P",value_past);
    auto eval = [&](tilemega::analysis::QuasiPolynomial const& q) {
      return q.SubstituteParams(theta).Eval(theta);
    };
    std::cout << seq << '\t' << value_past << '\t' << shape.split_k << '\t' << options.kappa
              << '\t' << projection.stages.size() << '\t' << eval(projection.runtime_task_refs)
              << '\t' << eval(projection.runtime_wait_entries)
              << '\t' << eval(projection.max_worker_task_refs) << '\n';
  }
} catch (std::exception const& e) {
  std::cerr << "runtime-projection: " << e.what() << '\n'; return 2;
}
