// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/AttentionWork.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/IR/MLIRContext.h>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=6) throw std::invalid_argument("usage: tilemega-attention-plan-price REPO MODEL CHUNKS EXTENT REGISTERS");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  std::string repo=argv[1],name=argv[2];
  int chunks=std::stoi(argv[3]),extent=std::stoi(argv[4]),registers=std::stoi(argv[5]);
  if (chunks<1 || extent<1 || registers<1) throw std::invalid_argument("missing measured attention candidate resources");
  std::string source=repo+"/docs/experiments/SEQSCAN/raw/export/"+name+".json";
  frontend::ImportOptions import;
  // The frozen attention_models experiment uses element ownership.
  import.rope_tile_per_block=import.kv_tile_per_block=import.activation_tile_per_block=import.combiner_tile_per_block=false;
  auto module=frontend::TorchExportImporter{}.Import(source,context,nullptr,import);
  auto seed=ModelDescription::FromCouplingGraph(*module,{4,3,7},name);
  import.gemms.assign(seed.gemms.size(),{128,128,16,3,1});
  for (std::size_t i=0;i<seed.stages.size();++i)
    if (seed.stages[i].kind==StageKind::kAttention)
      import.attention.push_back({static_cast<int>(i),{static_cast<unsigned>(chunks),static_cast<unsigned>(extent)}});
  module=frontend::TorchExportImporter{}.Import(source,context,nullptr,import);
  auto model=ModelDescription::FromCouplingGraph(*module,{4,3,7},name);
  auto plan=codegen::ReadRuntimePlan(*module);
  auto target=TargetSpec::FromJson(repo+"/configs/targets/sm_89.json");
  CostModel cost(target,model.dtype);
  CostModelOptions l2_options; l2_options.l2_events=true;
  CostModel l2(target,model.dtype,l2_options);
  auto traits=TensorBF16Traits(128,128,16,3);
  DpCandidate candidate{{128,128,16,3,1},registers,traits.smem_bytes};
  ChainDP dp(cost,{candidate});
  int shared=std::max(traits.smem_bytes,model.NonGemmSharedBytes());
  int resident=dp.CtasPerSm(shared,registers);
  if (shared>target.res.max_dynamic_smem_per_cta) throw std::invalid_argument("attention candidate exceeds dynamic shared budget");
  RuntimeProjectionOptions projection; projection.grid=resident*target.res.num_sms;
  projection.threads=traits.threads; projection.kappa=1;
  std::cout << std::setprecision(17) << "model\tseq\tchunks\tctas_per_sm\tshared_bytes\tworkspace_bytes\tattention_ns\tl1_ns\tl2_ns\ttask_refs\twaits\tdp_decomposition_ns\n";
  for (int seq:{4,128}) {
    model.dims={seq,3,seq+3};
    AttachRuntimeEventMetrics(model,plan,projection);
    auto predicted=cost.Evaluate(model,candidate.config,{resident});
    auto event=l2.Evaluate(model,candidate.config,{resident});
    double attention=0;
    for (std::size_t stage=0;stage<model.stages.size();++stage)
      if (model.stages[stage].kind==StageKind::kAttention)
        attention+=cost.TaskStageNs(model,stage,candidate.config,{resident});
    ChainDpStats stats; ChainDpOptions options; options.interface_term=false;
    auto chosen=dp.Solve(model,options,&stats);
    if (!chosen.feasible || chosen.residency.ctas_per_sm!=resident || chosen.cost.total_ns!=predicted.total_ns)
      throw std::runtime_error("pinned chunk DP differs from direct model evaluation");
    auto known=model.MetricBindings(); auto const& metrics=*model.coupling_metrics.runtime;
    std::cout << name << '\t' << seq << '\t' << chunks << '\t' << resident << '\t' << shared << '\t'
              << model.attention_plan->workspace_bytes.Eval(known) << '\t' << attention << '\t'
              << predicted.total_ns << '\t' << event.total_ns << '\t' << metrics.task_refs.Eval(known)
              << '\t' << metrics.wait_entries.Eval(known) << '\t' << stats.decomposition_error_ns << '\n';
  }
  if (isl.ReferenceCount()) throw std::runtime_error("attention plan price retained ISL references");
  std::cerr << "ATTENTION_PLAN fixed_chunk_dp=PASS remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
