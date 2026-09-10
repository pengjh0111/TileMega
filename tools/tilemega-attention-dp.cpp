// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/AttentionWork.h>
#include <tilemega/Solver/ChainDP.h>
#include <mlir/IR/MLIRContext.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=4) throw std::invalid_argument("usage: tilemega-attention-dp REPO MODEL CANDIDATES.tsv");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  frontend::ImportOptions import;
  import.rope_tile_per_block=import.kv_tile_per_block=import.activation_tile_per_block=import.combiner_tile_per_block=false;
  auto module=frontend::TorchExportImporter{}.Import(std::string(argv[1])+
      "/docs/experiments/SEQSCAN/raw/export/"+argv[2]+".json",context,nullptr,import);
  auto model=ModelDescription::FromCouplingGraph(*module,{4,3,7},argv[2]);
  std::ifstream file(argv[3]); std::string line;
  if (!file || !std::getline(file,line) || line!="chunks\textent\tregisters\ttile_m\ttile_n\ttile_k\tstages\tsplit")
    throw std::invalid_argument("missing attention candidate resource table");
  std::vector<AttentionDpCandidate> plans;
  while (std::getline(file,line)) {
    unsigned chunks,extent; DpCandidate gemm; std::istringstream row(line);
    if (!(row>>chunks>>extent>>gemm.registers>>gemm.config.tile_m>>gemm.config.tile_n>>
          gemm.config.tile_k>>gemm.config.stages>>gemm.config.split_k) || !chunks || !extent)
      throw std::invalid_argument("malformed attention candidate resource row");
    gemm.smem_bytes=TensorBF16SmemBytes(gemm.config.tile_m,gemm.config.tile_n,gemm.config.tile_k,gemm.config.stages);
    AttentionDpCandidate candidate; candidate.choices.resize(model.stages.size()); candidate.gemms={gemm};
    for (std::size_t i=0;i<model.stages.size();++i)
      if (model.stages[i].kind==StageKind::kAttention) candidate.choices[i]={chunks,extent};
    plans.push_back(std::move(candidate));
  }
  auto target=TargetSpec::FromJson(std::string(argv[1])+"/configs/targets/sm_89.json");
  CostModel cost(target,model.dtype);
  ChainDP dp(cost,{});
  std::cout << std::setprecision(17) << "model\tseq\tcandidate\tchunks\tfeasible\tctas_per_sm\tshared_bytes\tl1_ns\tselected\n";
  for (int seq:{4,128}) {
    model.dims={seq,3,seq+3};
    ChainDpOptions options; options.interface_term=false;
    std::vector<ChainDpSolution> alternatives;
    auto best=dp.SolveAttentionPlans(model,plans,options,&alternatives);
    if (!best.feasible) throw std::runtime_error("all attention plans infeasible");
    double minimum=std::numeric_limits<double>::infinity();
    for (std::size_t i=0;i<plans.size();++i) {
      auto check=model; ApplyAttentionCostPlan(check,plans[i].choices,kTensorBF16Threads);
      auto const& solution=alternatives[i];
      if (solution.feasible) {
        double direct=cost.Evaluate(check,solution.configs,solution.residency).total_ns;
        if (direct!=solution.cost.total_ns) throw std::runtime_error("attention DP alternative changed direct price bits");
        minimum=std::min(minimum,direct);
      }
      unsigned chunks=1;
      for (auto choice:solution.attention) chunks=std::max(chunks,choice.chunks);
      std::cout << argv[2] << '\t' << seq << '\t' << i << '\t' << chunks << '\t' << solution.feasible
                << '\t' << solution.residency.ctas_per_sm << '\t' << solution.max_smem_bytes << '\t'
                << solution.cost.total_ns << '\t' << (solution.feasible && solution.cost.total_ns==best.cost.total_ns) << '\n';
    }
    if (best.cost.total_ns!=minimum) throw std::runtime_error("attention DP differs from exhaustive candidate minimum");
  }
  if (isl.ReferenceCount()) throw std::runtime_error("attention DP retained ISL references");
  std::cerr << "ATTENTION_DP exact_plan_minimum=PASS remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
