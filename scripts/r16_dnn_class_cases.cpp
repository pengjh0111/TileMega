// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Target/TargetSpec.h>
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
 if(argc!=4)return 2;
 try {
  auto target=tilemega::TargetSpec::FromJson(argv[2]);
  auto bridge=tilemega::frontend::ReadExportBridge(argv[1]);
  tilemega::frontend::DnnPlanOptions options;options.batch=1;
  options.deferred_layernorm=true;options.dwpw_fuse=true;options.memory_reuse="l2";
  options.workspace_budget_bytes=target.res.max_dynamic_smem_per_cta;
  options.memory_l2_budget_bytes=target.res.l2_bytes;
  if(target.calib.l2_knee_bytes>0)options.memory_l2_budget_bytes=std::min(options.memory_l2_budget_bytes,std::uint64_t(target.calib.l2_knee_bytes));
  auto plan=tilemega::frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  mlir::MLIRContext context;
  auto imported=tilemega::frontend::TorchExportImporter{}.ImportSemantics(argv[1],plan,context);
  auto classes=tilemega::solver::BuildOperatorClasses(imported);
  std::ofstream out(argv[3]);out<<"{\"cases\":[{\"kappa\":1,\"residency\":1,\"geometries\":[";
  for(std::size_t i=0;i<classes.size();++i){if(i)out<<',';out<<"{\"tile_m\":64,\"tile_n\":128,\"tile_k\":64,\"stages\":2,\"split_k\":1,\"impl\":\"mma16\"}";}
  out<<"]}]}\n";
  std::cout<<"{\"class_count\":"<<classes.size()<<",\"gemms\":"<<plan.gemms.size()<<",\"gpu_executed\":false}\n";
 }catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
}
