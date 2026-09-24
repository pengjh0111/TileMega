// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <stdexcept>
using namespace tilemega;
void require(bool ok,char const* why) {if(!ok)throw std::runtime_error(why);}
std::string dump(mlir::ModuleOp m) {std::string s;llvm::raw_string_ostream o(s);m.print(o);return s;}
int main(int argc,char** argv) {
 try {
  analysis::IslContext isl;mlir::MLIRContext context;
  std::string path=argc>1 ? argv[1] : std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/E2E_GEN/raw/export_bridge.json";
  auto bridge=frontend::ReadExportBridge(path);
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  frontend::TorchExportImporter importer;
  auto imported=importer.ImportSemantics(path,plan,context);
  auto original=imported.lifted.sem.ops.front(),renamed=original;
  renamed.name="another_layer";renamed.result.name="other_tensor";
  require(analysis::SemanticSignature(original)==analysis::SemanticSignature(renamed),"names changed SemSig");
  int collisions=0;
  for(auto const& op:imported.lifted.sem.ops) {
    if(op.operands.empty())continue;
    if(collisions==0) {
      auto changed=op;
      changed.operands[0].map.results[0].offset=changed.operands[0].map.results[0].offset+analysis::ClosedForm::Constant(1);
      require(analysis::SemanticSignature(op)!=analysis::SemanticSignature(changed),"index mapping collision");
      ++collisions;
    }
    if(!op.element_reads.empty()) {
      auto changed=op;changed.element_reads.front().nonnegative.push_back(analysis::IndexResult::Dim(op.domain.front().name));
      require(analysis::SemanticSignature(op)!=analysis::SemanticSignature(changed),"element_reads collision");
      ++collisions;break;
    }
  }
  require(collisions==2,"missing adversarial pairs");
  analysis::CouplingCache cache;
  for(int split:{1,2}) {
    frontend::ImportOptions options;options.gemms.resize(plan.gemms.size());
    for(std::size_t i=0;i<options.gemms.size();++i)options.gemms[i]={32,int(i%2 ? 32:16),32,2,split};
    options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
    auto baseline=importer.ImportPlan(path,plan,context,nullptr,options);
    analysis::ScopedExactAnalysisMemo memo;
    auto cold=importer.InstantiateForGranularity(imported,context,options,&cache);
    auto warm=importer.InstantiateForGranularity(imported,context,options,&cache);
    require(dump(*baseline)==dump(*cold),"cold cache CG differs from original importer");
    require(dump(*baseline)==dump(*warm),"warm cache CG differs from original importer");
    auto edge=*warm->getOps<dialect::CouplingOp>().begin();auto saved=edge.getWaitAttr();
    edge.setWaitAttr(dialect::MetricAttr::get(&context,analysis::QuasiPolynomial::Constant(999999)));
    require(mlir::failed(mlir::verify(*warm)),"memoized calculation hid an invalid coupling metric");
    edge.setWaitAttr(saved);
    require(mlir::succeeded(mlir::verify(*warm)),"restored coupling metric is invalid");
    auto symbolic=analysis::QuasiPolynomial::FromIslText("[N] -> { N }");
    for(int n:{1,2,1,2}) {
      analysis::ParamBinding known;known.Bind("N",n);
      require(symbolic.SemanticallyEqual(analysis::QuasiPolynomial::Constant(1),known)==(n==1),
          "memoized equality ignored a changed binding");
    }
    require(memo.memo.hits>0,"exact calculation memo never hit");
    std::cout<<"EXACT_MEMO hit="<<memo.memo.hits<<" miss="<<memo.memo.misses<<" mutation_rejected=PASS\n";
    std::cout<<"CACHE_EQ split="<<split<<" PASS\n";
  }
  require(cache.hits>0,"cache never hit");
  std::cout<<"SEMSIG_COLLISIONS pairs="<<collisions<<" PASS\nCACHE hit="<<cache.hits<<" miss="<<cache.misses<<" PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
