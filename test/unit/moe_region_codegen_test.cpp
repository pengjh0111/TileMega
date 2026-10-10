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
  if(mode=="--emit-decoder" && argc==7) {
    auto bridge=ReadExportBridge(argv[2]);unsigned seq=0;
    for(auto const& node:bridge.nodes)if(node.name=="input_ids")seq=std::stoul(node.shape.at(1));
    unsigned batch=std::stoul(argv[4]);bool deferred=std::stoul(argv[5]),pages=std::stoul(argv[6]);
    ServingOptions options;options.seq=seq;options.moe_batch=batch;options.capacity=192;
    options.kv_block=64;options.query_rows=seq==1?8:16;options.argmax_tile_n=32;
    options.phase=seq==1?ServingOptions::Phase::kDecode:ServingOptions::Phase::kPrefill;
    options.moe_grouped=batch*seq>2;options.moe_block_rows=32;options.deferred_norm=deferred;
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
    assert(plan.dm && plan.serving && !plan.forward);
    mlir::MLIRContext context;ImportOptions geometry;geometry.phase_batch=batch;
    geometry.gemms.assign(plan.gemms.size(),GemmGranularity{16,32,32,2,1});
    for(unsigned index=0;index<plan.gemms.size();++index) {
      auto const& gemm=plan.gemms[index];
      if(gemm.chain.side_count)geometry.gemms[index]={16,16,32,2,1};
      if(gemm.epilogue==PlanGemm::Epilogue::kArgmaxPartial)geometry.gemms[index]={16,32,32,2,1};
    }
    auto module=TorchExportImporter{}.ImportPlan(argv[2],plan,context,nullptr,geometry);
    assert(mlir::succeeded(mlir::verify(*module)));
    // This fixture bypasses the solver, which normally binds the ABI past range.
    (*module)->setAttr("tmexec.solved_past",mlir::IntegerAttr::get(
        mlir::IntegerType::get(&context,64),seq==1?3:0));
    if(pages) {
      auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
      ConfigureServingPages(*module,target,8192);ResolveServingWeightPacking(*module);
      assert(mlir::succeeded(mlir::verify(*module)));
    }
    auto runtime=ReadRuntimePlan(*module);
    assert(runtime.gemms.size()==plan.gemms.size());
    solver::ModelDims dims;dims.batch=batch;dims.seq=seq;dims.past=seq==1?3:0;
    dims.total=dims.seq+dims.past;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"moe-decoder");
    auto projected=solver::ProjectRuntimeQueues(model,runtime,{4,128,0});
    unsigned combined=0;
    for(auto const& stage:plan.stages)combined+=stage.kind==PlanTaskKind::kMoECombine;
    assert(projected.runtime_counted.size()==(options.moe_grouped?combined:0));
    auto cu=CouplingGraphToCUDA{}.LowerVariants({{*module,seq,seq}});
    assert(cu.find("RunMoe<")!=std::string::npos && cu.find("TILEMEGA_SERVING_QPERKV 8")!=std::string::npos);
    std::ofstream out(argv[3]);assert(out);out<<cu;
    std::cout<<"MoE decoder CUDA: batch="<<batch<<" seq="<<seq<<" deferred="<<deferred
             <<" pages="<<pages<<" stages="<<plan.stages.size()<<" PASS\n";
    return 0;
  }
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
      // The router, too, has no row-major consumer. Packed-only plans must
      // not expose an unused second copy to the weight loader.
      assert(buffers.size()==plan.buffers.size());
      for(unsigned index=0;index<3;++index) {

        auto gemm=mlir::cast<mlir::DictionaryAttr>(gemms[index]);
        unsigned buffer=gemm.getAs<mlir::IntegerAttr>("b").getInt();
        auto weight=mlir::cast<mlir::DictionaryAttr>(buffers[buffer]);
        assert(buffer==plan.gemms[index].b);
        assert(weight.getAs<mlir::IntegerAttr>("constant").getInt()==
            (index?16u:1u)*plan.gemms[index].n*plan.gemms[index].k);
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
