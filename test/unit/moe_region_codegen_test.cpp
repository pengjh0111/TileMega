// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/IR/Verifier.h>
#include <cassert>
#include <fstream>
#include <iostream>

namespace tilemega::tests::moe_region_codegen_test {
int TestMoeRegionCodegen(int argc,char** argv) {
  using namespace frontend;using namespace analysis;using namespace codegen;
  IslContext isl;unsigned cases=0;
  bool emit_group=(argc==3 && std::string(argv[1])=="--emit-group") ||
      (argc==4 && std::string(argv[1])=="--emit-group-split");
  unsigned selected_split=argc==4?std::stoul(argv[3]):1;
  assert(selected_split>=1 && selected_split<=3);
  for(bool grouped:{false,true})for(unsigned split:{1u,2u,3u}) {
    if(emit_group && (!grouped || split!=selected_split))continue;
    ScopedExactAnalysisMemo memo;
    mlir::MLIRContext context;
    auto path=std::string(TILEMEGA_SOURCE_DIR)+"/test/fixtures/moe/region_before.json";
    auto bridge=ReadExportBridge(path);
    for(auto& node:bridge.nodes)for(auto& d:node.shape) {
      if(d=="2048")d="64";else if(d=="768")d="32";
      else if(d=="1536")d="64";else if(d=="128")d="16";
    }
    MoeRegionOptions options;options.tokens=17;options.grouped=grouped;
    options.block_rows=32;options.combine_token_tile=16;
    auto plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
    ImportOptions geometry;geometry.phase_batch=1;
    geometry.gemms={{16,16,16,2,1},{16,32,16,2,int(split)},{16,16,16,2,int(split)}};
    auto module=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,geometry);
    assert(mlir::succeeded(mlir::verify(*module)));
    auto runtime=ReadRuntimePlan(*module);
    unsigned counted=0;
    for(auto edge:module->getOps<dialect::CouplingOp>()) {
      auto contract=dialect::ReadBoundCountedScatter(edge,runtime.task_binding);
      if(contract) {
        assert(grouped && contract->expected.size()==8);
        for(unsigned id=0;id<8;++id)assert(contract->expected[id]==(id/4?8u:128u));
        ++counted;
      }
    }
    assert(counted==unsigned(grouped));
    solver::ModelDims dims;dims.seq=dims.total=17;dims.batch=1;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"moe-region");
    auto projected=solver::ProjectRuntimeQueues(model,runtime,{4,128,0});
    assert(projected.runtime_counted.size()==counted);
    assert(model.stages.back().width==16 && model.stages.back().dm_workspace_bytes==16*16*4);
    assert(model.stages.back().moe.step==DmMoeStep::kCombine);
    auto cu=CouplingGraphToCUDA{}.LowerVariants({{*module,17,17}});
    assert(cu.find("RunMoe<")!=std::string::npos && cu.find("TILEMEGA_SERVING_PHASE 2")!=std::string::npos);
    assert((cu.find("StageDependency::Map::kCounted")!=std::string::npos)==grouped);
    if(((argc==3 && std::string(argv[1])=="--emit") || emit_group) &&
        grouped && split==selected_split) {
      std::ofstream out(argv[2]);assert(out);out<<cu;
    }
    ++cases;
    std::cout<<"MoE region case: grouped="<<grouped<<" split="<<split<<" PASS"<<std::endl;
  }
  assert(cases==(emit_group?1u:6u));
  std::cout<<"MoE region codegen: "<<cases<<" graphs, counted contracts, TN16 combine, projection and CUDA generation PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_region_codegen_test
