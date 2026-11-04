// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Analysis/ISLContext.h>
#include <fstream>
#include <iostream>
#include <set>
using namespace tilemega;
using namespace tilemega::solver;

int main(int argc,char** argv) {try {
  if(argc!=4)throw std::invalid_argument("usage: tn-sweep target cells output");
  analysis::IslContext isl;
  auto target_json=json::ParseFile(argv[1]);auto target=TargetSpec::FromJson(argv[1]);
  if(std::ifstream(argv[3]))throw std::invalid_argument("retain previous TN domain");
  std::ofstream output(argv[3]);
  if(!output)throw std::runtime_error("cannot open TN sweep output");
  auto cells=json::ParseFile(argv[2]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
    space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
    space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
    auto family=unsigned(cell.At("family").AsNumber("family"));
    if((family!=0 && family!=6) || (family==6 && !seed.expert_fuse))
      throw std::invalid_argument("I3 requires fused M56 or fallback qkv");
    if(!space.Legal(seed))throw std::invalid_argument("A0 winner is not legal");
    std::set<int> partitions;
    for(auto const& g:space.FamilyCandidates(family,seed.residency))partitions.insert(g.tile_n);
    json::Array variants;
    for(int tn:partitions) {
      auto c=seed;auto& g=c.gemms[family];int tk=g.TileK();
      g.tile_n=tn;g.slot_bytes=2*tn*tk;
      bool legal=space.Legal(c);
      variants.emplace_back(json::Object{{"tile_n",tn},{"legal",legal},{"coordinates",c.ToJson()},
        {"reason",legal?"":"illegal with the other A0 coordinates fixed"}});
    }
    json::Value row=json::Object{{"input",cell},{"variants",variants},
      {"scope","all family TN values; TK, warps, slots, register stages and all other A0 coordinates fixed; slot_bytes follows 2*TN*TK"}};
    output<<row.Dump(0);output.flush();
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
