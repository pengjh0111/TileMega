// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Support/Json.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <fstream>
#include <iostream>
namespace tilemega::commands::request_floor {
int RunRequestFloor(int argc,char** argv) {
  if(argc!=8)throw std::runtime_error(
      "usage: tilemega inspect request-floor CG TARGET B PAST_LO PAST_HI OUTPUT.json STEPS.tsv");
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  if(!module)throw std::runtime_error("cannot read serving CG");
  auto info=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  if(!info)throw std::runtime_error("request floor needs a serving CG");
  int seq=info.getAs<mlir::IntegerAttr>("seq").getInt();
  int batch=std::stoi(argv[3]),lo=std::stoi(argv[4]),hi=std::stoi(argv[5]);
  if(batch<=0 || lo<0 || hi<lo)throw std::runtime_error("invalid floor interval");
  solver::ModelDims dims;dims.seq=seq;dims.past=lo;dims.total=lo+seq;dims.batch=batch;
  auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"serving-floor");
  auto target=TargetSpec::FromJson(argv[2]);auto floor=solver::DeriveModelDramFloor(*module,model,target,"");
  std::ofstream steps(argv[7]);steps<<"past\tdram_ns\tcompute_ns\tfloor_ns\n";
  steps.precision(17);double sum=0;json::Array points;
  for(int past=lo;past<=hi;++past) {
    model.dims.past=past;model.dims.total=past+seq;
    auto value=floor.Evaluate(model.MetricBindings());sum+=value.floor_ns;
    steps<<past<<'\t'<<value.dram_ns<<'\t'<<value.compute_ns<<'\t'<<value.floor_ns<<'\n';
    if(past==lo || past==(lo+hi)/2 || past==hi)
      points.push_back(json::Object{{"past",past},{"dram_ns",value.dram_ns},
          {"compute_ns",value.compute_ns},{"floor_ns",value.floor_ns}});
  }
  json::Value report(json::Object{{"batch",batch},{"seq",seq},{"past_lo",lo},{"past_hi",hi},
      {"sum_floor_seconds",sum/1e9},{"dram_polynomial",floor.dram_ns.ToString()},
      {"compute_polynomial",floor.compute_ns.ToString()},{"points",points},{"cg",argv[1]}});
  std::ofstream(argv[6])<<report.Dump()<<'\n';std::cout<<report.Dump()<<'\n';return 0;
}
}
