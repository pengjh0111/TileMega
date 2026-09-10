// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <limits>
#include <cstring>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc<3 || argc>5) throw std::invalid_argument("usage: tilemega-interface-probe REPO MODEL [--unified] [SEQ]");
  bool unified=argc>=4 && std::string(argv[3])=="--unified";
  if (argc>=4 && !unified) throw std::invalid_argument("expected --unified");
  int seq=argc==5 ? std::stoi(argv[4]) : 128;
  if (seq<=0) throw std::invalid_argument("sequence must be positive");
  analysis::IslContext context;
  std::string root=argv[1],name=argv[2];
  auto target=TargetSpec::FromJson(root+"/configs/targets/sm_89.json");
  mlir::MLIRContext mlir;
  mlir.getOrLoadDialect<dialect::CGDialect>();
  auto cg=frontend::TorchExportImporter{}.Import(
      root+"/docs/experiments/SEQSCAN/raw/export/"+name+".json",mlir);
  auto model=ModelDescription::FromCouplingGraph(*cg,{seq,3,seq+3},name);
  CostModelOptions historical;
  historical.cg_interface=false;
  historical.unified_task_cost=unified;
  CostModel cost(target,model.dtype,historical);
  std::cout << std::setprecision(17)
      << "model\ttile_m\ttile_n\tproducer\tconsumer\tproducer_gemm\tconsumer_gemm\twaits\tconsumers\tvolume\tinterface_ns\n";
  for (int m:{32,128}) for (int n:{32,128}) {
    std::vector<GemmConfig> configs(model.gemms.size(),{m,n,16,3,1});
    auto edges=InstantiateModelCouplings(model,configs);
    for (auto const& edge:edges) {
      auto bindings=model.MetricBindings();
      auto waits=edge.wait.SumDomain().SubstituteParams(bindings).Eval({});
      auto consumers=edge.relation.Reverse().ImageCard().SubstituteParams(bindings).Eval({});
      auto volume=edge.volume.SubstituteParams(bindings).Eval({});
      std::cout << name << '\t' << m << '\t' << n << '\t' << edge.producer_task << '\t'
          << edge.consumer_task << '\t' << model.stages.at(edge.producer).gemm << '\t'
          << model.stages.at(edge.consumer).gemm << '\t' << waits << '\t' << consumers
          << '\t' << volume << '\t' << cost.InterfaceEdgeNs(edge,model) << '\n';
    }
    std::cerr << "INTERFACE_PROBE model=" << name << " m=" << m << " n=" << n
              << " edges=" << edges.size() << " remaining=" << context.ReferenceCount() << '\n';
  }
  for (int pm:{32,128}) for (int cm:{32,128}) {
    std::vector<GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
    configs.at(3).tile_m=pm;
    configs.at(6).tile_m=cm;
    auto edges=InstantiateModelCouplings(model,configs);
    for (auto const& edge:edges) {
      if (model.stages.at(edge.producer).gemm!=3 || model.stages.at(edge.consumer).gemm!=6) continue;
      std::cerr << std::setprecision(17) << "INTERFACE_PAIR producer_gemm=3 consumer_gemm=6 producer_m="
                << pm << " consumer_m=" << cm << " ns=" << cost.InterfaceEdgeNs(edge,model) << '\n';
    }
  }
  std::map<std::string,int> registers;
  std::ifstream regfile(root+"/docs/experiments/ORACLE/raw_bf16/cost/registers_"+name+".tsv");
  std::string line;
  while (std::getline(regfile,line)) {
    if (line.empty() || line.front()=='#') continue;
    auto tab=line.find('\t');
    registers.emplace(line.substr(0,tab),std::stoi(line.substr(tab+1)));
  }
  std::vector<DpCandidate> candidates;
  for (int m:{32,128}) for (int n:{32,128}) for (int split:{1,2}) {
    auto traits=TensorBF16Traits(m,n,16,3);
    int smem=traits.smem_bytes;
    for (auto const& stage:model.stages)
      smem=std::max(smem,int(sizeof(float))*codegen::SimtSharedElements(
          static_cast<codegen::TaskKind>(stage.kind),traits.threads,TILEMEGA_ATTENTION_MAX_TOTAL));
    candidates.push_back({{m,n,16,3,split},registers.at(std::to_string(m)+"x"+
        std::to_string(n)+"x16s3"),smem});
  }
  CostModelOptions priced_options;
  priced_options.cg_interface=true;
  priced_options.unified_task_cost=unified;
  CostModel priced(target,model.dtype,priced_options);
  ChainDP dp(priced,candidates),old_dp(cost,candidates);
  ChainDpOptions dp_options;
  for (auto mode:{DpMode::kUniform,DpMode::kPerOperatorSplit,DpMode::kPerOperator}) {
    dp_options.mode=mode;
    ChainDpStats stats;
    auto result=dp.Solve(model,dp_options,&stats);
    auto old=old_dp.Solve(model,dp_options);
    if (!result.feasible || !old.feasible) throw std::runtime_error("probe DP has no solution");
    std::cerr << "INTERFACE_DP model=" << name << " mode=" << int(mode)
              << " old_ns=" << old.cost.total_ns << " new_ns=" << result.cost.total_ns
              << " interface_ns=" << result.cost.interface_ns
              << " spread_ns=" << stats.interface_spread_ns
              << " frontier=" << stats.interface_frontier_width
              << " decomposition_ns=" << stats.decomposition_error_ns
              << " transitions=" << stats.transitions << '\n';
  }
  // A non-adjacent residual edge supplies the two independent axes. All
  // other choices are fixed, so a complete Cartesian enumeration is cheap.
  dp_options.mode=DpMode::kPerOperator;
  dp_options.per_operator_candidates.assign(model.gemms.size(),{0});
  for (int i:{3,6}) dp_options.per_operator_candidates[i]={0,2,4,6};
  auto chosen=dp.Solve(model,dp_options);
  double oracle=std::numeric_limits<double>::infinity();
  for (int p:{0,2,4,6}) for (int c:{0,2,4,6}) {
    std::vector<GemmConfig> configs(model.gemms.size(),candidates.front().config);
    configs[3]=candidates[p].config; configs[6]=candidates[c].config;
    int resident=dp.CtasPerSm(candidates[0].smem_bytes,candidates[0].registers);
    for (int i:{p,c}) resident=std::min(resident,dp.CtasPerSm(candidates[i].smem_bytes,candidates[i].registers));
    oracle=std::min(oracle,priced.Evaluate(model,configs,{resident}).total_ns);
  }
  if (!chosen.feasible || std::memcmp(&oracle,&chosen.cost.total_ns,sizeof(double)))
    throw std::runtime_error("CG frontier DP differs from exhaustive non-adjacent two-axis oracle");
  std::cerr << "INTERFACE_EXHAUSTIVE model=" << name << " configurations=16 price_bits_equal=1"
            << " unified=" << unified << " seq=" << seq << '\n';
  int errors=0;
  auto reject=[&](auto&& action) {
    long before=context.ReferenceCount();
    bool failed=false;
    try { action(); } catch (std::exception const&) { failed=true; }
    if (!failed || context.ReferenceCount()!=before)
      throw std::runtime_error("interface rejection failed or retained isl objects");
    ++errors;
  };
  reject([&] { auto broken=model; broken.task_semantics.clear(); dp.Solve(broken,{}); });
  reject([&] { auto o=dp_options; o.general_transitions=false; dp.Solve(model,o); });
  reject([&] { auto o=dp_options; o.per_operator_candidates[0]={int(candidates.size())}; dp.Solve(model,o); });
  reject([&] { auto configs=chosen.configs; configs.pop_back(); InstantiateModelCouplings(model,configs); });
  reject([&] { auto edges=InstantiateModelCouplings(model,chosen.configs); auto edge=edges.front();
    edge.wait=analysis::QuasiPolynomial::Constant(-1); priced.InterfaceEdgeNs(edge,model); });
  std::cerr << "INTERFACE_ERRORS branches=" << errors << " reference_delta=0\n";
  if (context.ReferenceCount()) throw std::runtime_error("interface probe retained isl objects");
  std::cerr << "ISL_CONTEXT remaining=0\n";
} catch (std::exception const& error) {
  std::cerr << "tilemega-interface-probe: " << error.what() << '\n'; return 2;
}
