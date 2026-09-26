// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/ExecOps.h>
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/SymbolTable.h>
#include <cassert>
#include <iostream>
namespace tilemega::tests::handoff_ir_test {
int TestHandoffIr(int argc,char** argv) try {
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto module=argc>1?mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context):
    frontend::TorchExportImporter{}.Import(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",context);
  assert(module);
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
  std::string choice=argc>2?argv[2]:"recompute";
  int proofs=0,rejected=0;bool runtime_wired=false;
  for(auto edge:graph.getBody().front().getOps<dialect::CouplingOp>()) {
    auto p=mlir::dyn_cast_or_null<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getSrc()));
    auto c=mlir::dyn_cast_or_null<dialect::TileSpaceOp>(mlir::SymbolTable::lookupSymbolIn(graph,edge.getDst()));
    // Select normalization edges as test cases; the verifier has no kind matcher.
    if(!p || !c)continue;
    if(choice=="recompute" && (p.getArithmetic().value_or("")!="rmsnorm" || c.getKind().getValue().getValue()!="gemm"))continue;
    if(choice=="last_arriver" && c.getKind().getValue().getValue()!="attention_merge")continue;
    mlir::OperationState hs(edge.getLoc(),dialect::HandoffOp::getOperationName());
    hs.addAttribute("coupling",mlir::FlatSymbolRefAttr::get(&context,edge.getSymName()));hs.addAttribute("kind",b.getStringAttr(choice));
    auto handoff=mlir::cast<dialect::HandoffOp>(b.create(hs));
    try {auto proof=dialect::VerifyHandoffAccess(handoff);assert(!proof.composed.reads.empty());++proofs;}
    catch(std::invalid_argument const& e){std::cerr<<edge.getSymName().str()<<": "<<e.what()<<'\n';++rejected;}
    handoff.setKindAttr(b.getStringAttr("invalid"));bool caught=false;
    try {dialect::VerifyHandoffAccess(handoff);}catch(std::invalid_argument const&){caught=true;}assert(caught);
    if(proofs>=1) {
      handoff.setKindAttr(b.getStringAttr(choice));
      dialect::ApplyHandoffs(*module);
      assert(mlir::succeeded(mlir::verify(*module)));
      if(argc>1 && choice=="recompute") {
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
      assert(fused==1);break;
    }
    handoff.erase();
  }
  assert(proofs>0);
  std::cout<<"HANDOFF_IR kind="<<choice<<" proofs="<<proofs<<" unsupported="<<rejected
           <<" invalid_kind_rejected=1 norm_runtime_wired="<<runtime_wired<<"\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
