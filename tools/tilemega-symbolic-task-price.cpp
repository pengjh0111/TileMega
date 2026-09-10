// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/ChainDP.h>
#include <mlir/IR/MLIRContext.h>
#include <gmpxx.h>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc<3 || argc>6) throw std::invalid_argument("usage: tilemega-symbolic-task-price REPO MODEL [collective|scalar|dp] [MAX_SEQ] [COMPILED_REGISTERS]");
  std::string mode=argc>3 ? argv[3] : "collective";
  int max_seq=argc>4 ? std::stoi(argv[4]) : (mode=="scalar" ? 16 : 512);
  if ((mode!="scalar" && mode!="collective" && mode!="dp") || max_seq<1)
    throw std::invalid_argument("invalid symbolic price probe mode or domain");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=frontend::TorchExportImporter{}.Import(std::string(argv[1])+
      "/docs/experiments/SEQSCAN/raw/export/"+argv[2]+".json",context);
  auto model=ModelDescription::FromCouplingGraph(*module,{4,3,7},argv[2]);
  auto target=TargetSpec::FromJson(std::string(argv[1])+"/configs/targets/sm_89.json");
  CostModelOptions options; options.measured_cache_curve=true; options.cg_interface=mode=="dp";
  CostModel cost(target,model.dtype,options);
  if (mode=="dp") {
    if (argc!=6 || std::stoi(argv[5])<=0) throw std::invalid_argument("DP probe requires compiled registers");
    std::vector<DpCandidate> candidates;
    for (int split:{1,2,16}) candidates.push_back({{128,128,16,3,split},std::stoi(argv[5]),TensorBF16Traits(128,128,16,3).smem_bytes});
    ChainDP dp(cost,candidates);
    auto symbolic=model; symbolic.dims=ModelDims::Symbolic("S",3);
    ChainDpOptions dp_options; dp_options.mode=DpMode::kPerOperatorSplit;
    FiniteParameterDomain domain{"S",1,max_seq,{}};
    std::cerr << "symbolic DP constructing seq partitions\n";
    auto result=dp.SolveSymbolicParameter(symbolic,domain,dp_options);
    std::cout << std::unitbuf << std::setprecision(17) << "begin\tend\tresidency\tsplits\tpolynomial\n";
    for (auto const& piece:result.pieces) {
      std::cout << piece.begin << '\t' << piece.end << '\t' << piece.residency.ctas_per_sm << '\t';
      for (auto const& config:piece.configs) std::cout << config.split_k << ',';
      std::cout << '\t' << piece.total_ns.ToString() << '\n';
    }
    std::cerr << "symbolic DP finite control\n";
    auto finite=dp.SolveFiniteParameter(symbolic,domain,dp_options);
    int checks=0; double max_rounding=0;
    for (auto const& region:finite.pieces) for (int seq=region.begin;seq<=region.end;++seq) {
      auto const& numeric=region.points.at(seq-region.begin);
      auto piece=std::find_if(result.pieces.begin(),result.pieces.end(),[&](auto const& p) { return p.begin<=seq && seq<=p.end; });
      if (!numeric.feasible || piece==result.pieces.end() || numeric.residency.ctas_per_sm!=piece->residency.ctas_per_sm)
        throw std::runtime_error("symbolic DP feasibility or residency differs from finite control");
      for (std::size_t i=0;i<numeric.configs.size();++i) {
        auto const& a=numeric.configs[i]; auto const& b=piece->configs.at(i);
        if (std::tie(a.tile_m,a.tile_n,a.tile_k,a.stages,a.split_k)!=std::tie(b.tile_m,b.tile_n,b.tile_k,b.stages,b.split_k))
          throw std::runtime_error("symbolic DP choice differs from finite control at seq="+std::to_string(seq));
      }
      auto intervals=piece->total_ns.QuadraticIntervals(result.parameter,seq,seq);
      if (intervals.size()!=1) throw std::runtime_error("symbolic DP price domain mismatch");
      auto const& c=intervals.front().coefficients;
      mpq_class exact=mpq_class(c[0])+mpq_class(c[1])*seq+mpq_class(c[2])*seq*seq;
      double delta=std::abs(exact.get_d()-numeric.cost.total_ns);
      max_rounding=std::max(max_rounding,delta);
      if (delta>128*std::numeric_limits<double>::epsilon()*std::abs(numeric.cost.total_ns))
        throw std::runtime_error("symbolic DP price exceeds arithmetic rounding envelope at seq="+std::to_string(seq));
      ++checks;
    }
    if (isl.ReferenceCount()) throw std::runtime_error("symbolic DP leaked ISL references");
    std::cerr << "SYMBOLIC_DP choices=" << checks << " pieces=" << result.pieces.size()
        << " transitions=" << result.transitions << " max_rounding_ns=" << std::setprecision(17) << max_rounding
        << " remaining=0; L2/chunk/fusion/placement choices not asserted\n";
    return 0;
  }
  if (mode=="scalar") {
    GemmConfig config{128,128,16,3,1};
    auto graph=InstantiateModelTasks(model,std::vector<GemmConfig>(model.gemms.size(),config));
    std::cout << std::unitbuf << std::setprecision(17)
        << "model\tstage\top\tpieces\tchecks\tmax_absolute_rounding_ns\tpolynomial\n";
    int checks=0;
    for (auto const& task:model.task_semantics) {
      if (model.stages.at(task.stage).kind==StageKind::kGemm) continue;
      auto input=DeriveModelTaskInput(model,task,graph,nullptr);
      BackendTraits traits; traits.threads=kTensorBF16Threads;
      std::cerr << "symbolic scalar stage=" << task.stage << " op=" << task.op.name << '\n';
      auto price=cost.SymbolicScalarNs(input,traits,{2},model,model.seq_metric_parameter,1,max_seq);
      auto pieces=price.QuadraticIntervals(model.seq_metric_parameter,1,max_seq);
      double error=0;
      for (int seq=1;seq<=max_seq;++seq) {
        auto piece=std::find_if(pieces.begin(),pieces.end(),[&](auto const& p) { return p.begin<=seq && seq<=p.end; });
        if (piece==pieces.end()) throw std::runtime_error("symbolic scalar has a domain gap");
        mpq_class exact=mpq_class(piece->coefficients[0])+mpq_class(piece->coefficients[1])*seq+
            mpq_class(piece->coefficients[2])*seq*seq;
        model.dims={seq,3,seq+3};
        double numeric=cost.TaskCostNs(input,traits,{2},model,1);
        double delta=std::abs(exact.get_d()-numeric);
        error=std::max(error,delta);
        if (delta>64*std::numeric_limits<double>::epsilon()*std::abs(numeric)) {
          std::cerr << "SCALAR_MISMATCH seq=" << seq << " symbolic=" << exact.get_d() << " concrete=" << numeric << '\n';
          throw std::runtime_error("symbolic scalar differs beyond arithmetic rounding envelope");
        }
        ++checks;
      }
      std::cout << argv[2] << '\t' << task.stage << '\t' << task.op.name << '\t'
          << pieces.size() << '\t' << max_seq << '\t' << error << '\t' << price.ToString() << '\n';
    }
    if (isl.ReferenceCount()) throw std::runtime_error("symbolic scalar leaked ISL references");
    std::cerr << "SYMBOLIC_SCALAR points=" << checks << " remaining=0; full model DP not asserted\n";
    return 0;
  }
  auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
      [](auto const& task) { return task.op.kind==analysis::OperatorKind::kMatmul; });
  if (semantic==model.task_semantics.end()) throw std::runtime_error("missing semantic collective");
  std::cout << std::setprecision(17) << "model\tsplit\tpieces\tchecks\tmax_absolute_rounding_ns\tpolynomial\n";
  int checks=0;
  for (int split:{1,2,4,8,16}) {
    GemmConfig config{128,128,16,3,split};
    auto graph=InstantiateModelTasks(model,std::vector<GemmConfig>(model.gemms.size(),config));
    auto input=DeriveModelTaskInput(model,*semantic,graph,&config);
    auto traits=TensorBF16Traits(128,128,16,3);
    int chunks=cost.Chunks(model.gemms.at(model.stages[semantic->stage].gemm),config);
    auto price=cost.SymbolicCollectiveNs(input,traits,{2},model,chunks,model.seq_metric_parameter,1,max_seq);
    auto pieces=price.QuadraticIntervals(model.seq_metric_parameter,1,max_seq);
    double error=0; int points=0;
    for (int seq=1;seq<=max_seq;++seq) {
      auto piece=std::find_if(pieces.begin(),pieces.end(),[&](auto const& p) { return p.begin<=seq && seq<=p.end; });
      if (piece==pieces.end()) throw std::runtime_error("symbolic collective has a domain gap");
      mpq_class exact=mpq_class(piece->coefficients[0])+mpq_class(piece->coefficients[1])*seq+
          mpq_class(piece->coefficients[2])*seq*seq;
      model.dims={seq,3,seq+3};
      double numeric=cost.TaskCostNs(input,traits,{2},model,chunks);
      double delta=std::abs(exact.get_d()-numeric);
      error=std::max(error,delta);
      // Report algebraic-vs-IEEE rounding; this is not the A6 bit gate and
      // does not substitute for the eventual symbolic DP choice equality.
      if (delta>64*std::numeric_limits<double>::epsilon()*std::abs(numeric))
        throw std::runtime_error("symbolic collective differs beyond arithmetic rounding envelope");
      ++points; ++checks;
    }
    std::cout << argv[2] << '\t' << split << '\t' << pieces.size() << '\t' << points << '\t'
              << error << '\t' << price.ToString() << '\n';
  }
  if (isl.ReferenceCount()) throw std::runtime_error("symbolic collective leaked ISL references");
  std::cerr << "SYMBOLIC_COLLECTIVE points=" << checks << " remaining=0; full model DP not asserted\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
