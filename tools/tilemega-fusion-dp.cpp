// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <cstring>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=6 && argc!=7) throw std::invalid_argument("usage: tilemega-fusion-dp EXPORT TARGET PTXAS_LOG SEQ PAST [SELECTED.mlir]");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=frontend::TorchExportImporter{}.Import(argv[1],context);
  int seq=std::stoi(argv[4]),past=std::stoi(argv[5]);
  auto model=ModelDescription::FromCouplingGraph(*module,{seq,past,seq+past},argv[1]);
  auto target=TargetSpec::FromJson(argv[2]);
  FusionDpDomain domain;
  domain.plan=codegen::ReadRuntimePlan(*module);
  std::ifstream file(argv[3]);
  if (!file) throw std::invalid_argument("cannot read compiled register evidence");
  std::string log{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
  BackendCandidate compiled;
  if (!compiled.RecordPtxas(log,"tilemega_l2_kernel"))
    throw std::invalid_argument("L2 register evidence missing in ptxas log");
  int registers=*compiled.estimatedRegisters();
  int threads=model.dtype==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
  std::vector<GemmConfig> configs;
  for (auto const& g:domain.plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  for (auto const& stage:model.stages) {
    BackendTraits traits;
    if (stage.gemm>=0) {
      auto const& g=configs.at(stage.gemm);
      traits=model.dtype==ScalarType::kBF16 ? TensorBF16Traits(g.tile_m,g.tile_n,g.tile_k,g.stages)
          : SimtF32Traits(g.tile_m,g.tile_n,g.tile_k,g.stages);
    } else {
      traits.threads=threads;
      traits.smem_bytes=sizeof(float)*codegen::SimtSharedElements(static_cast<codegen::TaskKind>(stage.kind),
          threads,TILEMEGA_ATTENTION_MAX_TOTAL);
    }
    domain.stage_traits.push_back(traits);
    // The measured whole-kernel maximum is an upper bound on each phase;
    // it is not evidence of the future fused kernel's register allocation.
    domain.stage_registers.push_back(registers);
  }
  for (std::size_t index=1;index<model.task_semantics.size();++index) {
    auto const& p=model.task_semantics[index-1]; auto const& c=model.task_semantics[index];
    if (c.stage!=p.stage+1) continue;
    try {
      (void)DeriveLogicalFusionCandidate(model,configs,p.op.name,c.op.name);
      (void)DeriveModelFusionCandidate(model,configs,p.stage,c.stage);
    } catch (std::invalid_argument const& error) {
      std::cerr << "FUSION_EXCLUDED producer=" << p.op.name << " consumer=" << c.op.name
                << " reason=" << error.what() << '\n';
      continue;
    }
    domain.pairs.emplace_back(p.op.name,c.op.name);
  }
  CostModelOptions pricing; pricing.l2_events=true; pricing.kappa=1;
  CostModel cost(target,model.dtype,pricing);
  ChainDP dp(cost,{});
  ChainDpOptions options;
  std::vector<FusionDpAlternative> alternatives;
  auto best=dp.SolveFusionIntervals(model,domain,options,&alternatives);
  if (!best.feasible || alternatives.empty()) throw std::runtime_error("fusion DP found no feasible plan");
  std::cout << std::unitbuf << std::setprecision(17)
            << "fusion_count\tctas_per_sm\tshared\tregisters\ttask_refs\twaits\ttask_ns\tevent_ns\ttotal_ns\tpairs\n";
  bool checked_baseline=false;
  double minimum=std::numeric_limits<double>::infinity();
  for (auto const& alternative:alternatives) {
    auto const& s=alternative.solution;
    minimum=std::min(minimum,s.cost.total_ns);
    std::cout << s.fusion.size() << '\t' << s.residency.ctas_per_sm << '\t' << s.max_smem_bytes
              << '\t' << s.max_registers << '\t' << alternative.task_refs << '\t' << alternative.wait_entries
              << '\t' << s.cost.task_ns_sum << '\t' << s.cost.event_ns << '\t' << s.cost.total_ns << '\t';
    for (auto const& [p,c]:s.fusion) std::cout << p << ':' << c << ',';
    std::cout << '\n';
    if (s.fusion.empty()) {
      auto bound=model;
      AttachRuntimeEventMetrics(bound,domain.plan,{target.res.num_sms*s.residency.ctas_per_sm,threads,1});
      auto baseline=cost.Evaluate(bound,configs,s.residency);
      if (std::memcmp(&baseline.total_ns,&s.cost.total_ns,sizeof(double)))
        throw std::runtime_error("unfused interval branch is not bit-identical to Evaluate");
      checked_baseline=true;
    }
  }
  if (!checked_baseline || best.cost.total_ns!=minimum || isl.ReferenceCount())
    throw std::runtime_error("fusion interval minimum, control or reference audit failed");
  if (argc==7) {
    if (!best.fusion.empty()) dialect::FuseTaskPairs(*module,best.fusion);
    auto inputs=ReadFusedTaskInputs(*module);
    if (inputs.size()!=best.fusion.size()) throw std::runtime_error("selected interval writeback lost a fusion");
    std::error_code error;
    llvm::raw_fd_ostream output(argv[6],error,llvm::sys::fs::CD_CreateNew);
    if (error) throw std::runtime_error("cannot create selected fusion module: "+error.message());
    module->print(output); output.flush();
    if (output.has_error()) throw std::runtime_error("selected fusion module write failed");
  }
  std::cerr << "FUSION_DP alternatives=" << alternatives.size() << " selected=" << best.fusion.size()
            << " baseline_bits_equal=1 remaining=0 fused_registers_gpu_verified=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
