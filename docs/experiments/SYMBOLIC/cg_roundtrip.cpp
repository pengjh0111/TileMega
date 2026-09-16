// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/ParametricPlacementIO.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <optional>
using namespace tilemega;
static analysis::CouplingRelation read(std::filesystem::path p){std::ifstream f(p);if(!f)throw std::invalid_argument("missing "+p.string());std::stringstream s;s<<f.rdbuf();return analysis::CouplingRelation::FromIslText(s.str());}
int main(int argc,char** argv)try{
 if(argc!=4)throw std::invalid_argument("cg_roundtrip INPUT_CG PROOF_DIR OUT_CG");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto m=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!m)throw std::invalid_argument("invalid CG");
 auto resident=m->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_grid");if(!resident)throw std::invalid_argument("missing residency witness");
 std::vector<mlir::Attribute> attrs;std::filesystem::path root=argv[2];
 for(std::string family:{"legacy_grid_stride","rotate","band","wavefront"}){
  std::optional<solver::ParametricPlacement> united;
  for(int grid:{256,340}){
   auto stem=family+"_g"+std::to_string(grid);
   solver::ParametricPlacement p{read(root/(stem+".tasks.isl")),read(root/(stem+".dependencies.isl")),read(root/(stem+".pi_sigma.isl")),read(root/(stem+".rank.isl")),analysis::CouplingRelation::FromIslText("[S,G] -> { [] -> [G,"+std::to_string(resident.getInt())+"] : 1<=S<=128 and G="+std::to_string(grid)+" }"),family};
   if(!united)united=p;else{united->tasks=united->tasks.Union(p.tasks);united->dependencies=united->dependencies.Union(p.dependencies);united->pi_sigma=united->pi_sigma.Union(p.pi_sigma);united->rank=united->rank.Union(p.rank);united->grid_limit=united->grid_limit.Union(p.grid_limit);}
  }
  attrs.push_back(dialect::ParametricPlacementAttr(&context,*united));
 }
 mlir::Builder b(&context);m->getOperation()->setAttr("tilemega.parametric_alternatives",b.getArrayAttr(attrs));
 std::string text;llvm::raw_string_ostream stream(text);m->print(stream);stream.flush();std::ofstream(argv[3])<<text<<'\n';
 auto back=mlir::parseSourceString<mlir::ModuleOp>(text,&context);if(!back)throw std::invalid_argument("CG did not round trip");
 auto copied=back->getOperation()->getAttrOfType<mlir::ArrayAttr>("tilemega.parametric_alternatives");
 for(std::size_t i=0;i<attrs.size();++i){
  auto a=dialect::ReadParametricPlacement(llvm::cast<mlir::DictionaryAttr>(attrs[i]));auto p=dialect::ReadParametricPlacement(llvm::cast<mlir::DictionaryAttr>(copied[i]));
  for(int g:{256,340})for(int s:{1,32,64,96,128}){
   analysis::ParamBinding theta;theta.Bind("S",s).Bind("G",g);
   auto expected=a.pi_sigma.BindParams(theta).Points(),actual=p.pi_sigma.BindParams(theta).Points();
   if(expected!=actual)throw std::runtime_error("round trip changed symbolic evaluation");
   std::cout<<"CG_SYMBOLIC_ROUNDTRIP family="<<p.family<<" grid="<<g<<" seq="<<s<<" nodes="<<actual.size()<<" identical=1\n";
  }
 }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
