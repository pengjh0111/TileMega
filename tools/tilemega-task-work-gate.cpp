// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
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
  using namespace tilemega::analysis;
  using namespace tilemega::frontend;
  IslContext context;
  if (argc!=2) throw std::invalid_argument("usage: tilemega-task-work-gate REPO");
  std::string root=argv[1];
  std::cout << "dtype\tmodel\tconfig\tgemm_stages\tseq_points\twork_bit_checks\tstatus\n";
  for (std::string dtype:{"bf16","f32"}) for (std::string model:{"gqa2","mha4"}) {
    std::string source=dtype=="bf16"
        ? root+"/docs/experiments/SEQSCAN/raw/export/"+model+".json"
        : root+"/docs/experiments/"+(model=="gqa2" ? "E2E_GEN" : "P3_GENERALIZATION")+
            "/raw/export_bridge.json";
    auto bridge=ReadExportBridge(source);
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
    if (plan.dtype!=dtype) throw std::runtime_error("export dtype differs from gate dtype");
    auto lifted=LiftSemantics(plan,{});
    std::ifstream oracle(root+"/docs/experiments/ORACLE/raw/screen_"+model+".tsv");
    if (!oracle) throw std::runtime_error("missing complete ORACLE configuration universe");
    std::string line; std::getline(oracle,line);
    int configurations=0;
    while (std::getline(oracle,line)) {
      auto row=Fields(line);
      if (row.size()<13) throw std::runtime_error("malformed ORACLE row");
      if (row[8]!="PASS") continue;
      GemmGranularity config{std::stoi(row[0]),std::stoi(row[1]),std::stoi(row[2]),
                             std::stoi(row[3]),std::stoi(row[4])};
      std::vector<GemmGranularity> configs(plan.gemms.size(),config);
      auto graph=Instantiate(lifted.sem,LaunchGranularity(lifted,plan,configs));
      int stages=0, checks=0;
      for (auto const& op:lifted.sem.ops) {
        if (op.kind!=OperatorKind::kMatmul) continue;
        auto const* task=graph.Find(op.name);
        if (!task) throw std::runtime_error("missing instantiated GEMM");
        TaskWorkOptions options;
        options.reduction_tiles.emplace(op.reduction.dim,ClosedForm::Constant(config.tile_k));
        auto work=DeriveTaskWork(op,*task,{},options);
        auto k=op.Dim(op.reduction.dim)->extent.Eval({},{});
        auto n=op.result.axes.back().extent.Eval({},{});
        double k_tiles=std::ceil(double(k)/config.tile_k);
        int chunks=std::max(1,std::min(config.split_k,static_cast<int>(k_tiles)));
        double iters=std::ceil(k_tiles/chunks);
        double bytes=dtype=="bf16" ? 2.0 : 4.0;
        double historical=bytes*config.tile_k*(config.tile_m+config.tile_n);
        for (int seq:{1,4,128,512,2048}) {
          ParamBinding theta;
          theta.Bind("S",seq).Bind("past",3);
          for (auto const& coordinate:task->Coordinates()) theta.Bind(coordinate,0);
          double expected=std::ceil(double(seq)/config.tile_m)*
                          std::ceil(double(n)/config.tile_n)*chunks;
          double actual_bytes=double(work.nominal_read_elements.SubstituteParams(theta).Eval({}))*bytes/iters;
          if (!Bits(double(work.task_count.SubstituteParams(theta).Eval({})),expected) ||
              !Bits(actual_bytes,historical) ||
              !Bits(double(work.nominal_task_reduce_extent.SubstituteParams(theta).Eval({}))/config.tile_k,iters))
            throw std::runtime_error("A3 GEMM work bit mismatch: "+dtype+" "+model+" "+op.name+
                " seq="+std::to_string(seq)+" config="+row[0]+"x"+row[1]+"x"+row[2]+"k"+row[4]);
          checks+=3;
        }
        ++stages;
      }
      if (stages!=static_cast<int>(plan.gemms.size()))
        throw std::runtime_error("not every production GEMM stage was checked");
      std::cout << dtype << '\t' << model << '\t' << row[0] << 'x' << row[1] << 'x' << row[2]
                << 's' << row[3] << 'k' << row[4] << '\t' << stages << "\t5\t" << checks << "\tPASS\n";
      ++configurations;
    }
    if (configurations!=1077) throw std::runtime_error("incomplete 1077-configuration work gate");
    std::cerr << "TASK_WORK_GATE dtype=" << dtype << " model=" << model
              << " configs=" << configurations << " status=PASS price_gate=pending\n";
  }
} catch (std::exception const& error) {
  std::cerr << "tilemega-task-work-gate: " << error.what() << '\n'; return 2;
}
