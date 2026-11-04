// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <fstream>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;

int main(int argc,char** argv) {try {
  if(argc!=5)throw std::invalid_argument("usage: cache-replay target root cells output");
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  if(std::ifstream(argv[4]))throw std::invalid_argument("retain previous cache replay");
  std::ofstream output(argv[4]);
  if(!output)throw std::runtime_error("cannot open cache replay output");
  auto cells=json::ParseFile(argv[3]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto receipt=json::ParseFile(cell.At("receipt").AsString("receipt"));
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90NativeCtgCache cache;
    // Algorithm 1 invokes CTG analysis once per E entry, then once per
    // shortlist entry. Coordinate memo hits never invoke this callback.
    for(auto phase:{"evaluated","shortlist"}) {
      int index=0;
      for(auto const& candidate:receipt.At(phase).AsArray(phase)) {
        auto c=Sm90DenseCoordinates::FromJson(candidate.At("coordinates"));
        auto hits=cache.coupling.hits;
        auto result=Sm90NativeCtgScore(imported,work,c,cost,context,cache);
        json::Value row=json::Object{{"input",cell},{"phase",phase},{"index",index++},
          {"pool_ns",result.step_ns},{"derivations",double(result.derivations)},
          {"coupling_hits",double(cache.coupling.hits-hits)},
          {"coupling_hits_cumulative",double(cache.coupling.hits)},
          {"scope","cold replay of every A0 analysis callback in receipt order"}};
        output<<row.Dump(0);output.flush();
      }
    }
    std::cout<<cell.At("id").AsString("id")<<" callback replay complete\n"<<std::flush;
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
