// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/Parser/Parser.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace tilemega;
int main(int argc,char** argv)try {
  if(argc!=3 && argc!=4)throw std::invalid_argument("interval_check CG TARGET [TABLE_PREFIX]");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  if(!module)throw std::invalid_argument("invalid interval CG");
  std::vector<dialect::PlacementTable> tables;std::string error;
  if(!dialect::ReadPlacementIntervalTables(*module,&tables,&error) || tables.size()!=5)
    throw std::runtime_error("missing five-point interval: "+error);
  int compared=0;
  for(auto const& table:tables) {
    auto copy=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>((*module)->clone()));
    dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[2]);
    options.dims={int(table.seq),int(table.past),int(table.seq+table.past)};
    options.residency=1;options.requested_grid=table.grid;options.kappa=1;
    std::string why;
    if(!solver::HopCurve::FromTsv("docs/experiments/SIMULATOR/hop_ns.tsv",&options.hop,&why))throw std::runtime_error(why);
    auto point=dialect::SolveAndWritePlacement(*copy,options);auto const& plan=point.candidates.front().plan;
    std::vector<int> worker,slot;
    for(std::size_t s=0;s<plan.owner.size();++s) {
      worker.insert(worker.end(),plan.owner[s].begin(),plan.owner[s].end());
      slot.insert(slot.end(),plan.slot[s].begin(),plan.slot[s].end());
    }
    if(worker!=table.worker || slot!=table.slot)throw std::runtime_error("interval changed point winner");
    if(argc==4) {
      std::ofstream out(std::string(argv[3])+std::to_string(table.seq)+".tsv");
      out<<"node\tworker\tslot\n";
      for(std::size_t n=0;n<worker.size();++n)out<<n<<'\t'<<worker[n]<<'\t'<<slot[n]<<'\n';
    }
    ++compared;std::cout<<"INTERVAL_POINT seq="<<table.seq<<" nodes="<<worker.size()<<" diff_bytes=0\n";
  }
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,5}});
  std::string text;llvm::raw_string_ostream stream(text);(*module).print(stream);stream.flush();
  auto parsed=mlir::parseSourceString<mlir::ModuleOp>(text,&context);
  if(!parsed || codegen::CouplingGraphToCUDA{}.LowerVariants({{*parsed,1,5}})!=source)
    throw std::runtime_error("interval serialization changed emitted CUDA");
  mlir::OpBuilder b(&context);auto root=(*parsed)->getAttrOfType<mlir::DictionaryAttr>(dialect::kPlacementTableAttr);
  auto entries=root.getAs<mlir::ArrayAttr>("interval");std::vector<mlir::Attribute> bad(entries.begin(),entries.end());
  int rejected=0;
  for(auto const* key:{"seq","past","grid"}) {
    bad.assign(entries.begin(),entries.end());
    mlir::NamedAttrList gap(llvm::cast<mlir::DictionaryAttr>(bad[2]));
    gap.set(key,b.getI64IntegerAttr(99));bad[2]=gap.getDictionary(&context);
    mlir::NamedAttrList malformed(root);malformed.set("interval",b.getArrayAttr(bad));
    (*parsed)->setAttr(dialect::kPlacementTableAttr,malformed.getDictionary(&context));
    if(dialect::ReadPlacementIntervalTables(*parsed,&tables,&error))throw std::runtime_error("inconsistent interval accepted");
    ++rejected;
  }
  std::cout<<"INTERVAL_PASS points="<<compared<<" serialization=byte_identical invalid_metadata_rejected="<<rejected<<" variants=1\n";

}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
