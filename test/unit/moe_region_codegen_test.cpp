// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Target/TargetSpec.h>
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
  auto mode=argc>1?std::string(argv[1]):std::string();
  bool tokens_pinned=argc==4 && (mode=="--emit-group-tokens" || mode=="--emit-slot-tokens" ||
      mode=="--emit-group-pages-tokens" || mode=="--emit-slot-pages-tokens");
  bool pages=(argc==3 || tokens_pinned) && (mode=="--emit-group-pages" ||
      mode=="--emit-slot-pages" || mode=="--emit-group-pages-tokens" || mode=="--emit-slot-pages-tokens");
  bool emit_group=(argc==3 && std::string(argv[1])=="--emit-group") ||
      (argc==4 && std::string(argv[1])=="--emit-group-split") || pages || tokens_pinned;
  unsigned selected_split=argc==4 && !tokens_pinned?std::stoul(argv[3]):1;
  unsigned tokens=tokens_pinned?std::stoul(argv[3]):17;
  assert(tokens>=1 && tokens<=4096);
  assert(selected_split>=1 && selected_split<=3);
  for(bool grouped:{false,true})for(unsigned split:{1u,2u,3u}) {
    bool selected_group=mode.find("--emit-slot")!=0;
    if(emit_group && (grouped!=selected_group || split!=selected_split))continue;
    ScopedExactAnalysisMemo memo;
    mlir::MLIRContext context;
    auto path=std::string(TILEMEGA_SOURCE_DIR)+"/test/fixtures/moe/region_before.json";
    auto bridge=ReadExportBridge(path);
    for(auto& node:bridge.nodes)for(auto& d:node.shape) {
      if(d=="2048")d="64";else if(d=="768")d="32";
      else if(d=="1536")d="64";else if(d=="128")d="16";
    }
    MoeRegionOptions options;options.tokens=tokens;options.grouped=grouped;
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
        assert(grouped && contract->expected.size()==4*((tokens+15)/16));
        for(unsigned id=0;id<contract->expected.size();++id)
          assert(contract->expected[id]==8*std::min(16u,tokens-(id/4)*16));
        ++counted;
      }
    }
    assert(counted==unsigned(grouped));
    solver::ModelDims dims;dims.seq=dims.total=tokens;dims.batch=1;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"moe-region");
    auto projected=solver::ProjectRuntimeQueues(model,runtime,{4,128,0});
    assert(projected.runtime_counted.size()==counted);
    assert(model.stages.back().width==16 && model.stages.back().dm_workspace_bytes==16*16*4);
    assert(model.stages.back().moe.step==DmMoeStep::kCombine);
    if(pages) {
      auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
      ConfigureServingPages(*module,target,8192);
      ResolveServingWeightPacking(*module);
      auto packed=ReadRuntimePlan(*module);
      assert(packed.gemms.size()==3);
      auto attr=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
      auto buffers=attr.getAs<mlir::ArrayAttr>("buffers"),gemms=attr.getAs<mlir::ArrayAttr>("gemms");
      for(unsigned index=1;index<3;++index) {
        auto gemm=mlir::cast<mlir::DictionaryAttr>(gemms[index]);
        unsigned buffer=gemm.getAs<mlir::IntegerAttr>("b").getInt();
        auto weight=mlir::cast<mlir::DictionaryAttr>(buffers[buffer]);
        assert(buffer==plan.gemms[index].b);
        assert(weight.getAs<mlir::IntegerAttr>("constant").getInt()==
            16u*plan.gemms[index].n*plan.gemms[index].k);
      }
      ResolveServingWeightPacking(*module);
      assert((*module)->getAttr("tilemega.model_plan")==attr);
      assert(mlir::succeeded(mlir::verify(*module)));
    }
    auto cu=CouplingGraphToCUDA{}.LowerVariants({{*module,tokens,tokens}});
    assert(cu.find("RunMoe<")!=std::string::npos && cu.find("TILEMEGA_SERVING_PHASE 2")!=std::string::npos);
    assert((cu.find("StageDependency::Map::kCounted")!=std::string::npos)==grouped);
    if(((argc==3 && std::string(argv[1])=="--emit") || emit_group) &&
        grouped==selected_group && split==selected_split) {
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
