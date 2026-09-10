// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/CostModel.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
  tilemega::analysis::IslContext isl_context;
  if (argc < 12 || argc > 18 || argc==16)
    throw std::invalid_argument("usage: tilemega-runtime-projection EXPORT.json "
        "GRID THREADS KAPPA {PAST|symbolic} TILE_M TILE_N TILE_K STAGES SPLIT {tile|element} [CG_ORDER=0|1] [PARTITION_WORKERS=0|1] [SPLIT_PERIODS=0|1] [ATTENTION_CHUNKS CHUNK_EXTENT [PLACEMENT_SEQ]]");
  tilemega::solver::RuntimeProjectionOptions options{
      std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4])};
  if (argc>=13) {
    if (std::string(argv[12])!="0" && std::string(argv[12])!="1")
      throw std::invalid_argument("CG_ORDER must be 0 or 1");
    options.cg_split_task_order = std::string(argv[12])=="1";
  }
  if (argc>=14) {
    if (std::string(argv[13])!="0" && std::string(argv[13])!="1")
      throw std::invalid_argument("PARTITION_WORKERS must be 0 or 1");
    options.partition_worker_counts = std::string(argv[13])=="1";
  }
  if (argc>=15) {
    if (std::string(argv[14])!="0" && std::string(argv[14])!="1")
      throw std::invalid_argument("SPLIT_PERIODS must be 0 or 1");
    options.split_count_periods = std::string(argv[14])=="1";
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
  if (argc>=17) {
    int chunks=std::stoi(argv[15]),extent=std::stoi(argv[16]);
    if (chunks<=0 || extent<=0) throw std::invalid_argument("attention chunks and extent must be positive");
    auto seed_model=tilemega::solver::ModelDescription::FromCouplingGraph(
        *module,tilemega::solver::ModelDims::Symbolic("S",past),argv[1]);
    for (std::size_t stage=0;stage<seed_model.stages.size();++stage)
      if (seed_model.stages[stage].kind==tilemega::solver::StageKind::kAttention)
        import.attention.push_back({static_cast<int>(stage),
            {static_cast<std::uint32_t>(chunks),static_cast<std::uint32_t>(extent)}});
    if (import.attention.empty()) throw std::invalid_argument("model has no attention stages");
  }
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
  if (argc==18) {
    int seq=std::stoi(argv[17]);
    if (seq<=0 || symbolic_past) throw std::invalid_argument("placement needs positive seq and concrete past");
    tilemega::analysis::ParamBinding theta; theta.Bind("S",seq);
    auto placed=tilemega::solver::BalanceProjectedQueues(projection,theta,options.grid);
    auto concrete=model.SubstituteParams(theta);
    tilemega::solver::AttachRuntimeEventMetrics(concrete,plan,options);
    auto target=tilemega::TargetSpec::FromJson("configs/targets/sm_89.json");
    tilemega::solver::CostModelOptions prices;
    prices.l2_events=true; prices.kappa=options.kappa;
    tilemega::solver::CostModel cost(target,concrete.dtype,prices);
    std::vector<tilemega::solver::GemmConfig> configs(concrete.gemms.size(),
        {shape.tile_m,shape.tile_n,shape.tile_k,shape.stages,shape.split_k});
    tilemega::solver::Residency residency{options.grid/target.res.num_sms};
    auto before=cost.EventNs(concrete,configs,residency,projection.stages.size());
    auto& metrics=*concrete.coupling_metrics.runtime;
    metrics.wait_entries=tilemega::analysis::QuasiPolynomial::Constant(placed.wait_entries);
    metrics.max_worker_task_refs=tilemega::analysis::QuasiPolynomial::Constant(placed.placement.max_queue);
    metrics.fence_free_producers=tilemega::analysis::QuasiPolynomial::Constant(placed.placement.fence_free_producers);
    auto after=cost.EventNs(concrete,configs,residency,projection.stages.size());
    std::cerr << "PRODUCTION_PLACEMENT seq=" << seq << " tasks=" << placed.task_ids.size()
              << " stages=" << projection.stages.size() << " max_queue=" << placed.placement.max_queue
              << " baseline_max_queue=" << projection.max_worker_task_refs.Eval(theta)
              << " same_worker_edges=" << placed.placement.same_worker_edges
              << " fence_free_producers=" << placed.placement.fence_free_producers
              << " max_worker_span=" << placed.placement.max_worker_span << " resident_only=1\n";
    std::cerr << "PLACEMENT_EVENT_PRICE baseline_ns=" << before << " balanced_ns=" << after
              << " balanced_waits=" << placed.wait_entries << " delta_ns=" << after-before << '\n';
  }
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
