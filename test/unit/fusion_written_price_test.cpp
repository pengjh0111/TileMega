// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <cstring>
#include <iostream>

int main() try {
  using namespace tilemega;
  using namespace tilemega::solver;
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
  int cases=0,errors=0;
  for (auto model:{"gqa2","mha4"}) for (auto pair:{std::make_pair("l0.s05.rope","l0.s06.append"),
                                                std::make_pair("l0.s09.proj","l0.s09.add")}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+
        "/docs/experiments/SEQSCAN/raw/export/"+model+".json",context);
    auto description=ModelDescription::FromCouplingGraph(*module,{4,3,7},model);
    auto plan=codegen::ReadRuntimePlan(*module);
    std::vector<GemmConfig> configs;
    for (auto const& g:plan.gemms) configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
    auto before=DeriveLogicalFusionCandidate(description,configs,pair.first,pair.second);
    dialect::FuseTaskPair(*module,pair.first,pair.second);
    auto inputs=ReadFusedTaskInputs(*module);
    auto after=DeriveWrittenFusionCandidate(inputs.at(0),description,configs);
    auto traits=[&](std::size_t phase) {
      auto const& semantic=inputs[0].semantics[phase];
      auto const& stage=description.stages.at(semantic.stage);
      if (semantic.op.kind==analysis::OperatorKind::kMatmul) {
        auto const& config=configs.at(stage.gemm);
        return TensorBF16Traits(config.tile_m,config.tile_n,config.tile_k,config.stages);
      }
      BackendTraits result;
      result.threads=kTensorBF16Threads;
      auto kind=static_cast<codegen::TaskKind>(stage.kind);
      if (kind==codegen::TaskKind::kGemm) kind=codegen::TaskKind::kElementwise;
      result.smem_bytes=sizeof(float)*codegen::SimtSharedElements(kind,result.threads,TILEMEGA_ATTENTION_MAX_TOTAL);
      return result;
    };
    CostModel cost(target,description.dtype);
    auto a=PriceFusionTasks(before,cost,traits(0),traits(1),{2},description);
    auto b=PriceFusionTasks(after,cost,traits(0),traits(1),{2},description);
    auto values=[](auto const& p) { return std::vector<double>{p.separate_ns,p.fused_ns,p.recompute_ns,
        p.global_bytes_before,p.global_bytes_after,p.local_bytes}; };
    auto av=values(a),bv=values(b);
    if (std::memcmp(av.data(),bv.data(),av.size()*sizeof(double)) ||
        a.producer_tasks!=b.producer_tasks || a.consumer_tasks!=b.consumer_tasks ||
        a.recomputed_tasks!=b.recomputed_tasks || a.producer_waves!=b.producer_waves ||
        a.consumer_waves!=b.consumer_waves)
      throw std::runtime_error("written fusion price differs from selected candidate");
    auto reject=[&](auto action) {
      int refs=isl.ReferenceCount(); bool caught=false;
      try { action(); } catch (std::invalid_argument const&) { caught=true; }
      if (!caught || refs!=isl.ReferenceCount()) throw std::runtime_error("fused price rejection failed");
      ++errors;
    };
    reject([&] { (void)DeriveWrittenFusionCandidate(inputs[0],description,{}); });
    reject([&] { auto wrong=inputs[0]; wrong.semantics[0].stage=description.stages.size();
      (void)DeriveWrittenFusionCandidate(wrong,description,configs); });
    if (inputs[0].semantics[0].op.kind==analysis::OperatorKind::kMatmul) {
      reject([&] { auto wrong=configs; wrong[description.stages[inputs[0].semantics[0].stage].gemm].tile_m*=2;
        (void)DeriveWrittenFusionCandidate(inputs[0],description,wrong); });
    }
    ++cases;
    std::cout << "FUSION_WRITTEN_PRICE model=" << model << " producer=" << pair.first
              << " consumer=" << pair.second << " double_fields=6 bits_equal=1\n";
  }
  if (isl.ReferenceCount()) throw std::runtime_error("fused price reader leaked ISL references");
  std::cout << "FUSION_WRITTEN cases=" << cases << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
