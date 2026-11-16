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
  if(argc!=4)throw std::invalid_argument("usage: partition-domains target.json cells.json output.jsonl");
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);auto cells=json::ParseFile(argv[2]);
  if(std::ifstream(argv[3]))throw std::invalid_argument("retain previous partition output");
  std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot open output");
  for(auto const& cell:cells.AsArray("cells")) {
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
    space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
    space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
    if(work.tp!=1)throw std::invalid_argument("AB5 partition count is TP1 only");
    json::Array structures;
    for(int residency:{1,2}) {
      auto c=seed;c.residency=residency;c.cluster=1;c.group_wait=false;c.embed_la=false;c.argmax_stage=true;
      c.expert_fuse=false;c.route=work.moe?Sm90DenseCoordinates::Route::kStage:Sm90DenseCoordinates::Route::kLocal;
      c.combine_stage=work.moe;
      std::map<unsigned,std::vector<Sm90DenseGemm>> geometries;
      std::map<unsigned,std::set<int>> partitions;
      auto attention=space.AttentionCandidates(residency);
      bool complete=!attention.empty();
      for(unsigned f:c.Active()) {
        geometries[f]=space.FamilyCandidates(f,residency);
        complete&=!geometries[f].empty();
        if(!geometries[f].empty())c.gemms[f]=geometries[f].front();
      }
      std::set<std::pair<int,int>> expert_pairs;
      std::set<int> chunks;
      if(complete) {
        c.attention=attention.front();c.attention.merge_stage=true;
        Sm90DenseSearchSpace::Structure(c);
        // With aggregate TP1 boundaries all constraints factor by family,
        // except the shared ring of the expert pair. Realization choices can
        // restrict these domains, but cannot admit a new partition axis.
        if(work.moe) {
          Sm90DenseCoordinates witness;bool found=false;
          for(auto gu:geometries[6])for(auto dn:geometries[7]) {
            auto trial=c;trial.gemms[6]=gu;trial.gemms[7]=dn;
            bool legal=space.Legal(trial);
            if(!legal) {
              trial.expert_fuse=true;
              for(int slots=8;slots>=2 && !legal;--slots) {
                trial.gemms[6].slots=trial.gemms[7].slots=slots;legal=space.Legal(trial);
              }
            }
            if(legal) {
              expert_pairs.emplace(gu.tile_n,dn.tile_n);
              if(!found) {witness=trial;found=true;}
            }
          }
          complete=found;
          if(found)c=witness;
          for(auto [gu,dn]:expert_pairs) {partitions[6].insert(gu);partitions[7].insert(dn);}
        }else if(!space.Legal(c))throw std::runtime_error("independent dense domain lacks a legal witness");
        if(complete) {
          for(unsigned f:c.Active())if(f<6)for(auto g:geometries[f]) {
            auto trial=c;trial.gemms[f]=g;Sm90DenseSearchSpace::Structure(trial);
            if(space.Legal(trial))partitions[f].insert(g.tile_n);
          }
          for(auto a:attention) {
            auto trial=c;trial.attention=a;trial.attention.merge_stage=true;
            if(space.Legal(trial))chunks.insert(a.chunks);
          }
          for(unsigned f:c.Active())complete&=!partitions[f].empty();
          complete&=!chunks.empty();
        }
      }
      json::Object families;
      for(unsigned f:seed.Active()) {
        json::Array values;if(complete)for(int n:partitions[f])values.emplace_back(n);
        families.emplace_back(kSm90DenseFamilies[f],values);
      }
      json::Array pairs,attention_values;
      if(complete) {
        for(auto [gu,dn]:expert_pairs)pairs.emplace_back(json::Array{gu,dn});
        for(int n:chunks)attention_values.emplace_back(n);
      }
      structures.emplace_back(json::Object{{"residency",residency},{"grid",space.Grid(residency)},
          {"has_legal_program",complete},{"families",families},{"expert_pairs",pairs},
          {"attention_chunks",attention_values}});
    }
    json::Value row=json::Object{{"input",cell},{"layers",work.layers},{"moe",work.moe},
        {"residencies",structures},
        {"scope","distinct family-tied task partitions: TN per GEMM and attention chunks; body-only geometry and realization multiplicity excluded"}};
    out<<row.Dump(0);out.flush();std::cout<<cell.At("id").AsString("id")<<" partition domains passed\n"<<std::flush;
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
