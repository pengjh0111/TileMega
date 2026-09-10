// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<std::string> Fields(std::string const& line) {
  std::istringstream input(line); std::string field;
  std::vector<std::string> result;
  while (std::getline(input,field,'\t')) result.push_back(field);
  return result;
}
bool Bits(double a,double b) { return std::memcmp(&a,&b,sizeof(double))==0; }
}

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  analysis::IslContext context;
  if (argc!=2) throw std::invalid_argument("usage: tilemega-task-cost-gate REPO");
  std::string root=argv[1];
  auto target=TargetSpec::FromJson(root+"/configs/targets/sm_89.json");
  mlir::MLIRContext mlir;
  mlir.getOrLoadDialect<dialect::CGDialect>();
  std::cout << "dtype\tmodel\tconfig\tgemm_stages\tseq_points\tprice_bit_checks\tstatus\n"
            << std::setprecision(17);
  for (std::string dtype:{"bf16","f32"}) for (std::string name:{"gqa2","mha4"}) {
    std::string source=dtype=="bf16"
        ? root+"/docs/experiments/SEQSCAN/raw/export/"+name+".json"
        : root+"/docs/experiments/"+(name=="gqa2" ? "E2E_GEN" : "P3_GENERALIZATION")+
            "/raw/export_bridge.json";
    auto cg=frontend::TorchExportImporter{}.Import(source,mlir);
    auto model=ModelDescription::FromCouplingGraph(*cg,{4,3,7},name);
    if ((model.dtype==ScalarType::kBF16)!=(dtype=="bf16"))
      throw std::runtime_error("gate dtype differs from CG input");
    CostModel cost(target,model.dtype);
    std::ifstream oracle(root+"/docs/experiments/ORACLE/raw/screen_"+name+".tsv");
    if (!oracle) throw std::runtime_error("missing complete ORACLE configuration universe");
    std::string line; std::getline(oracle,line);
    int configurations=0;
    while (std::getline(oracle,line)) {
      auto row=Fields(line);
      if (row.size()<13) throw std::runtime_error("malformed ORACLE row");
      if (row[8]!="PASS") continue;
      GemmConfig config{std::stoi(row[0]),std::stoi(row[1]),std::stoi(row[2]),
                        std::stoi(row[3]),std::stoi(row[4])};
      std::vector<GemmConfig> configs(model.gemms.size(),config);
      auto graph=InstantiateModelTasks(model,configs);
      auto traits=model.dtype==ScalarType::kBF16
          ? TensorBF16Traits(config.tile_m,config.tile_n,config.tile_k,config.stages)
          : SimtF32Traits(config.tile_m,config.tile_n,config.tile_k,config.stages);
      int stages=0, checks=0;
      for (auto const& semantic:model.task_semantics) {
        if (semantic.op.kind!=analysis::OperatorKind::kMatmul) continue;
        auto const& stage=model.stages.at(semantic.stage);
        auto const& gemm=model.gemms.at(stage.gemm);
        auto input=DeriveModelTaskInput(model,semantic,graph,&config);
        for (int seq:{1,4,128,512,2048}) for (int resident:{1,2}) {
          model.dims={seq,3,seq+3};
          int chunks=0;
          double old=cost.GemmStageNs(gemm,config,{resident},model,&chunks);
          double current=cost.TaskCostNs(input,traits,{resident},model,chunks);
          if (!Bits(old,current)) {
            std::cerr << std::setprecision(17) << "TASK_PRICE_MISMATCH dtype=" << dtype
                      << " model=" << name << " stage=" << semantic.op.name << " seq=" << seq
                      << " residency=" << resident << " config=" << row[0] << 'x' << row[1]
                      << 'x' << row[2] << 's' << row[3] << 'k' << row[4]
                      << " old=" << old << " new=" << current << '\n';
            throw std::runtime_error("A6 GEMM stage-price bit gate failed; stop");
          }
          ++checks;
        }
        ++stages;
      }
      if (stages!=int(model.gemms.size())) throw std::runtime_error("incomplete GEMM stage coverage");
      std::cout << dtype << '\t' << name << '\t' << row[0] << 'x' << row[1] << 'x' << row[2]
                << 's' << row[3] << 'k' << row[4] << '\t' << stages << "\t5\t" << checks << "\tPASS\n";
      ++configurations;
    }
    if (configurations!=1077) throw std::runtime_error("incomplete 1077-configuration price gate");
    std::cerr << "TASK_PRICE_GATE dtype=" << dtype << " model=" << name
              << " configs=" << configurations << " status=PASS scalar_path=pending\n";
  }
  if (context.ReferenceCount()!=0) throw std::runtime_error("task price gate retained isl objects");
  std::cerr << "ISL_CONTEXT remaining=" << context.ReferenceCount() << '\n';
} catch (std::exception const& error) {
  std::cerr << "tilemega-task-cost-gate: " << error.what() << '\n'; return 2;
}
