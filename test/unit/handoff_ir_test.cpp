// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/ExecOps.h>
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Target/TargetSpec.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/SymbolTable.h>
#include <cassert>
#include <iostream>
#include <map>
namespace tilemega::tests::handoff_ir_test {
int TestHandoffIr(int argc,char** argv) try {
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  if(argc==5 && std::string(argv[1])=="splitk_import") {
    auto bridge=frontend::ReadExportBridge(argv[2]);
    frontend::ServingOptions serving;serving.seq=1;
    serving.phase=frontend::ServingOptions::Phase::kDecode;
    auto model=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,serving);
    frontend::ImportOptions options;
    bool split=std::string(argv[3])!="one";
    options.gemms.assign(model.gemms.size(),{16,128,64,2,split?4:1});
    options.gemms.back().split_k=1;
    auto module=frontend::TorchExportImporter{}.ImportPlan(argv[2],model,context,nullptr,options);
    bool nonpaged=std::string(argv[3])=="nonpaged_attention";
    if(!nonpaged)codegen::ConfigureServingPages(*module,TargetSpec::FromJson(argv[4]),16384);
    bool escape=std::string(argv[3])=="escape";
    if(escape) {
      mlir::Operation* combine=nullptr;
      for(auto task:module->getOps<dialect::TileSpaceOp>()) {
        if(task->hasAttr("split_access_semantic")) {
          auto name=task->getAttrOfType<mlir::StringAttr>("operator_name");
          if(name && name.getValue().ends_with(".combine")){combine=task;break;}
        }
      }
      assert(combine);
      auto cname=mlir::cast<dialect::TileSpaceOp>(combine).getSymName();
      dialect::CouplingOp edge;
      for(auto e:module->getOps<dialect::CouplingOp>())if(e.getDst()==cname){edge=e;break;}
      assert(edge);
      auto* reader=combine->clone();reader->setAttr("sym_name",mlir::StringAttr::get(&context,"unsafe_partial_reader"));
      module->getBody()->push_back(reader);
      auto* extra=edge->clone();extra->setAttr("sym_name",mlir::StringAttr::get(&context,"unsafe_partial_edge"));
      extra->setAttr("dst",mlir::FlatSymbolRefAttr::get(&context,"unsafe_partial_reader"));
      module->getBody()->push_back(extra);
    }
    unsigned mask=nonpaged?8:(std::string(argv[3])=="off"?0:4);
    try {
      auto selected=dialect::SelectServingHandoffs(*module,mask);
      assert(!escape);
      if(nonpaged) {
        int merges=0;
        auto desc=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
        for(auto stage:desc.getAs<mlir::ArrayAttr>("stages")) {
          auto d=mlir::cast<mlir::DictionaryAttr>(stage);
          if(d.getAs<mlir::StringAttr>("kind").getValue()=="kAttentionMerge")++merges;
        }
        assert(merges>0 && selected.last_arriver==merges);
      }else assert(selected.last_arriver==(split && mask?model.gemms.size()-1:0));
      std::cout<<"SPLIT_HANDOFF case="<<argv[3]<<" selected="<<selected.last_arriver<<" PASS\n";
    } catch(std::invalid_argument const& e) {
      if(!escape || std::string(e.what()).find("partial workspace")==std::string::npos)throw;
      std::cout<<"SPLIT_HANDOFF unsafe_partial_reader rejected PASS\n";
    }
    return 0;
  }
  auto const imported_case=argc<=1 || std::string(argv[1])=="audit" ||
      std::string(argv[1])=="smem_direct";
  auto module=!imported_case?mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context):
    frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",context);
  assert(module);
  if(argc>2 && (std::string(argv[2])=="select_serving" ||
                 std::string(argv[2])=="select_splitk" ||
                 std::string(argv[2])=="splitk_off")) {
    unsigned classes=std::string(argv[2])=="select_splitk"?4:
        std::string(argv[2])=="splitk_off"?0:7;
    auto selected=dialect::SelectServingHandoffs(*module,classes);
    if(classes==0){assert(selected.last_arriver==0);return 0;}
    assert(selected.recompute+selected.last_arriver>0);
    assert(!(*module)->hasAttr("tilemega.handoff_pending_lowering"));
    assert((*module)->hasAttr("tmexec.runtime_handoff_lowering"));
    auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1u,1u}});
    assert(!source.empty());
    std::cout<<"HANDOFF_SERVING recompute="<<selected.recompute
             <<" last_arriver="<<selected.last_arriver
             <<" cuda_bytes="<<source.size()<<'\n';
    return 0;
  }
  mlir::OpBuilder b(&context);b.setInsertionPointToEnd(module->getBody());
  mlir::OperationState gs(b.getUnknownLoc(),dialect::GraphOp::getOperationName());
  gs.addAttribute("sym_name",b.getStringAttr("graph"));gs.addRegion()->push_back(new mlir::Block);
  auto graph=mlir::cast<dialect::GraphOp>(b.create(gs));
  std::vector<mlir::Operation*> move;
  for(auto& op:*module->getBody())if(mlir::isa<dialect::TileSpaceOp,dialect::CouplingOp,dialect::EventTensorOp>(op))move.push_back(&op);
  for(auto op:move)op->moveBefore(&graph.getBody().front(),graph.getBody().front().end());
  mlir::OperationState ps(b.getUnknownLoc(),dialect::PlanOp::getOperationName());
  ps.addAttribute("sym_name",b.getStringAttr("plan"));ps.addAttribute("graph",mlir::FlatSymbolRefAttr::get(&context,"graph"));ps.addRegion()->push_back(new mlir::Block);
  auto plan=mlir::cast<dialect::PlanOp>(b.create(ps));
  std::vector<mlir::Operation*> decisions;
  for(auto& op:*module->getBody())if(mlir::isa<dialect::PlacementOp,dialect::ImplementationOp>(op))decisions.push_back(&op);
  for(auto op:decisions)op->moveBefore(&plan.getBody().front(),plan.getBody().front().end());
  b.setInsertionPointToEnd(&plan.getBody().front());
  std::string choice=argc>2?argv[2]:(argc>1 && imported_case?argv[1]:"recompute");
  if(choice=="smem_direct") {
    // c11 is a one-to-one relation in the reference fixture. Give its two
    // task spaces adjacent positions on one worker, then ask the real access
    // verifier and ApplyHandoffs pass to transform that two-stage edge.
    auto edge=mlir::dyn_cast_or_null<dialect::CouplingOp>(
        mlir::SymbolTable::lookupSymbolIn(graph,"c11"));
    assert(edge && edge.getRelation().getMap().IsSingleValued() &&
           edge.getRelation().getMap().Reverse().IsSingleValued());
    auto position=[](int offset) {
      return analysis::CouplingRelation::FromIslText(
          "[s11] -> { [m,n] -> [worker,slot] : worker=0 and "
          "slot=2*(4*m+n)+"+std::to_string(offset)+
          " and m>=0 and 128*m<s11 and 0<=n<=3 }");
    };
    int positioned=0;
    for(auto placement:plan.getBody().front().getOps<dialect::PlacementOp>()) {
      int offset=placement.getTask()==edge.getSrc()?0:
                 placement.getTask()==edge.getDst()?1:-1;
      if(offset<0)continue;
      placement->setAttr("position",dialect::CouplingMapAttr::get(&context,position(offset)));
      ++positioned;
    }
    assert(positioned==2 && mlir::succeeded(mlir::verify(*module)));
    mlir::OperationState hs(edge.getLoc(),dialect::HandoffOp::getOperationName());
    hs.addAttribute("coupling",mlir::FlatSymbolRefAttr::get(&context,edge.getSymName()));
    hs.addAttribute("kind",b.getStringAttr("smem_direct"));
    auto handoff=mlir::cast<dialect::HandoffOp>(b.create(hs));
    auto proof=dialect::VerifyHandoffAccess(handoff);
    assert(proof.consumer_to_producer.IsSingleValued());
    dialect::ApplyHandoffs(*module);
    assert(mlir::succeeded(mlir::verify(*module)));
    assert((*module)->hasAttr("tilemega.handoff_pending_lowering"));
    std::cout<<"HANDOFF_DIRECT two_stage_access=PASS ir_rewrite=PASS\n";
    return 0;
  }
  if(choice=="audit") {
    std::map<std::string,int> counts;
    for(auto edge:graph.getBody().front().getOps<dialect::CouplingOp>()) {
      for(char const* kind:{"recompute","last_arriver","smem_direct"}) {
        mlir::OperationState hs(edge.getLoc(),dialect::HandoffOp::getOperationName());
        hs.addAttribute("coupling",mlir::FlatSymbolRefAttr::get(&context,edge.getSymName()));
        hs.addAttribute("kind",b.getStringAttr(kind));
        auto handoff=mlir::cast<dialect::HandoffOp>(b.create(hs));
        try {
          (void)dialect::VerifyHandoffAccess(handoff);
          ++counts[kind];
          std::cout<<"HANDOFF_ELIGIBLE\t"<<kind<<'\t'<<edge.getSymName().str()
                   <<'\t'<<edge.getSrc().str()<<'\t'<<edge.getDst().str()<<'\n';
        }catch(std::invalid_argument const&) {}
        handoff.erase();
      }
    }
    std::cout<<"HANDOFF_AUDIT\t"<<counts["recompute"]<<'\t'
             <<counts["last_arriver"]<<'\t'<<counts["smem_direct"]<<'\n';
    return 0;
  }
  bool const multiple=choice=="recompute_multi";
  int proofs=0,rejected=0;bool runtime_wired=false,applied=false;
  for(auto edge:graph.getBody().front().getOps<dialect::CouplingOp>()) {
    auto p=mlir::dyn_cast_or_null<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getSrc()));
    auto c=mlir::dyn_cast_or_null<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getDst()));
    // Select normalization edges as test cases; the verifier has no kind matcher.
    if(!p || !c)continue;
    if((choice=="recompute" || multiple) && (p.getArithmetic().value_or("")!="rmsnorm" || c.getKind().getValue().getValue()!="gemm"))continue;
    if(choice=="last_arriver" && c.getKind().getValue().getValue()!="attention_merge")continue;
    mlir::OperationState hs(edge.getLoc(),dialect::HandoffOp::getOperationName());
    hs.addAttribute("coupling",mlir::FlatSymbolRefAttr::get(&context,edge.getSymName()));hs.addAttribute("kind",b.getStringAttr(multiple?"recompute":choice));
    auto handoff=mlir::cast<dialect::HandoffOp>(b.create(hs));
    bool valid=false;
    try {auto proof=dialect::VerifyHandoffAccess(handoff);assert(!proof.composed.reads.empty());++proofs;valid=true;}
    catch(std::invalid_argument const& e){std::cerr<<edge.getSymName().str()<<": "<<e.what()<<'\n';++rejected;}
    if(multiple && !valid){handoff.erase();continue;}
    handoff.setKindAttr(b.getStringAttr("invalid"));bool caught=false;
    try {dialect::VerifyHandoffAccess(handoff);}catch(std::invalid_argument const&){caught=true;}assert(caught);
    if(proofs>=1) {
      handoff.setKindAttr(b.getStringAttr(multiple?"recompute":choice));
      if(multiple && proofs<4)continue;
      dialect::ApplyHandoffs(*module);
      applied=true;
      assert(mlir::succeeded(mlir::verify(*module)));
      bool lowering_rejected=false;
      try { (void)codegen::CouplingGraphToCUDA{}.Lower(*module); }
      catch(std::invalid_argument const& e) {
        lowering_rejected=std::string(e.what()).find("runtime stage replanning")!=std::string::npos;
      }
      assert(lowering_rejected && "handoff CG must not silently use the old runtime stage table");
      lowering_rejected=false;
      try { (void)codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1u,4u}}); }
      catch(std::invalid_argument const& e) {
        lowering_rejected=std::string(e.what()).find("runtime stage replanning")!=std::string::npos;
      }
      assert(lowering_rejected && "variant lowering must reject pending handoff stage replanning");
      if(argc>1 && (choice=="recompute" || multiple)) {
        auto model=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
        assert(model && "serving handoff must retain its runtime model");
        bool wired=false;
        for(auto entry:mlir::cast<mlir::ArrayAttr>(model.get("gemms"))) {
          auto gemm=mlir::cast<mlir::DictionaryAttr>(entry);
          if(gemm.get("norm_input") || gemm.get("norm_weight")) {
            assert(gemm.get("norm_input") && gemm.get("norm_weight"));
            wired=true;
          }
        }
        assert(wired && "norm recompute must reach serving GEMM operands");
        runtime_wired=wired;
      }
      auto active=module->getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("tmexec.active_graph");assert(active);
      auto result=mlir::cast<dialect::GraphOp>(mlir::SymbolTable::lookupSymbolIn(*module,active.getValue()));
      int fused=0;for(auto f:result.getBody().front().getOps<dialect::FusedTileSpaceOp>()) {
        ++fused;dialect::VerifyWrittenHandoff(f);
        auto saved=f.getReads();f->setAttr("reads",b.getDictionaryAttr({}));bool rejected=false;
        try{dialect::VerifyWrittenHandoff(f);}catch(std::invalid_argument const&){rejected=true;}
        assert(rejected);f->setAttr("reads",saved);
        if(choice=="last_arriver") {
          auto ticket=mlir::SymbolTable::lookupNearestSymbolFrom<dialect::EventTensorOp>(f,f->getAttrOfType<mlir::FlatSymbolRefAttr>("ticket"));
          auto triggers=ticket->getAttr("triggers");ticket->setAttr("triggers",dialect::MetricAttr::get(&context,mlir::cast<dialect::MetricAttr>(triggers).getValue().Scale(0)));
          bool bad=false;try{dialect::VerifyWrittenHandoff(f);}catch(std::invalid_argument const&){bad=true;}assert(bad);
          ticket->setAttr("triggers",triggers);
        }
      }
      assert(fused==proofs);break;
    }
    handoff.erase();
  }
  assert(proofs>0 && applied);
  std::cout<<"HANDOFF_IR kind="<<choice<<" proofs="<<proofs<<" unsupported="<<rejected
           <<" invalid_kind_rejected=1 norm_runtime_wired="<<runtime_wired<<"\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
