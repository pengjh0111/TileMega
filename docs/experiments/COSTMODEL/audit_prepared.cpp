// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <mlir/Parser/Parser.h>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv)try {
  if(argc!=4)throw std::invalid_argument("audit_prepared CG TARGET OUTPUT.tsv");
  analysis::IslContext isl;mlir::MLIRContext ctx;ctx.getOrLoadDialect<dialect::CGDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&ctx);
  if(!module)throw std::invalid_argument("invalid CG");
  auto value=[&](char const* key){auto a=module->getOperation()->getAttrOfType<mlir::IntegerAttr>(key);if(!a)throw std::invalid_argument(key);return int(a.getInt());};
  int seq=value("tilemega.solved_seq"),past=value("tilemega.solved_past");
  dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[2]);
  options.dims={seq,past,seq+past};options.residency=value("tilemega.solved_residency");
  options.kappa=value("tilemega.solved_kappa");options.requested_grid=value("tilemega.solved_grid");
  options.verified_resident_limit=options.residency;
  auto p=dialect::PreparePlacementProblem(*module,options);
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"runtime_stage\tlogical_stage\tlogical_task\tcombine\tprice_ns\tactive_ctas_per_sm\n";
  for(std::size_t s=0;s<p.counts.size();++s)for(int t=0;t<p.counts[s];++t)
    out<<s<<'\t'<<p.projection.stages[s].logical_stage<<'\t'<<t<<'\t'<<p.projection.stages[s].combine<<'\t'<<p.task_ns[p.graph.stage_offsets[s]+t]<<'\t'<<options.residency<<'\n';
  std::cout<<"PREPARED_PRICE_AUDIT tasks="<<p.task_ns.size()<<" stages="<<p.counts.size()<<'\n';
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
